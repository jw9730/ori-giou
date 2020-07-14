"""
Utilities for oriented bounding box manipulation and GIoU.
"""
import torch
import math
import time
import torch.multiprocessing as mp

parallelism = 8

def box_center_to_corners(b):
    """
    Converts a set of oriented bounding boxes from
    centered representation (x_c, y_c, w, h, theta) to corner representation (x0, y0, ..., x3, y3).

    Arguments:
        b (Tensor[N, 5]): boxes to be converted. They are
            expected to be in (x_c, y_c, w, h, theta) format, where -pi <= theta < pi.

    Returns:
        c (Tensor[N, 8]): converted boxes in (x0, y0, ..., x3, y3) format, where
            the corners are sorted counterclockwise.
    """
    x_c, y_c, w, h, theta = b.unbind(-1)  # [N,]
    center = torch.stack([x_c, y_c], dim=-1).repeat(1, 4)  # [N, 8]

    assert (theta >= -math.pi).all()
    assert (theta < math.pi).all()

    dx = 0.5 * w
    dy = 0.5 * h
    cos = theta.cos()
    sin = theta.sin()
    dxcos = dx * cos
    dxsin = dx * sin
    dycos = dy * cos
    dysin = dy * sin

    dxy = [- dxcos + dysin, - dxsin - dycos,
             dxcos + dysin,   dxsin - dycos,
             dxcos - dysin,   dxsin + dycos,
           - dxcos - dysin, - dxsin + dycos]

    return center + torch.stack(dxy, dim=-1)  # [N, 8]


def box_corners_to_center(c):
    """
    Arguments:
        c (Tensor[N, 8]): boxes to be converted. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.

    Returns:
        b (Tensor[N, 5]): converted boxes in centered
            (x_c, y_c, w, h, theta) format, where -pi <= theta < pi.
    """
    x0,y0, x1,y1, x2,y2, x3,y3 = c.unbind(-1)

    x_c = (x0 + x2) / 2
    y_c = (y0 + y2) / 2

    wsin, wcos, hsin, hcos = (y1 - y0,
                              x1 - x0,
                              x0 - x_c + x1 - x_c,
                              y_c - y0 + y_c - y1)
    theta = torch.atan2(wsin, wcos)

    assert (theta >= -math.pi).all()
    assert (theta < math.pi).all()

    b = [x_c, y_c,
         (wsin ** 2 + wcos ** 2) ** 0.5,
         (hsin ** 2 + hcos ** 2) ** 0.5,
         theta]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    x0,y0, x1,y1, x2,y2, x3,y3 = boxes.unbind(-1)  # [N,]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 *\
           ((x3 - x0) ** 2 + (y3 - y0) ** 2) ** 0.5  # [N,]


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):  # v1, v2: [2,]
        self.a = v2[1] - v1[1]  # scalar
        self.b = v1[0] - v2[0]
        self.c = v2[0] * v1[1] - v2[1] * v1[0]

    def __call__(self, p):
        return self.a * p[0] + self.b * p[1] + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplementedError
        w = self.a * other.b - self.b * other.a
        return torch.stack([
            (self.b * other.c - self.c * other.b) / w,
            (self.c * other.a - self.a * other.c) / w], dim=-1)  # [2,]


def box_inter(box1, box2):
    """
    Finds intersection convex polygon using sequential cut, then computes its area by counterclockwise cross product.
    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241

       Arguments:
           box1, box2 (Tensor[8], Tensor[8]): boxes to compute area of intersection. They are
               expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.

       Returns:
           inter: torch.float32
    """
    intersection = box1.reshape([4, 2]).unbind(-2)  # [2,]
    box2_corners = box2.reshape([4, 2]).unbind(-2)  # [2,]

    for p, q in zip(box2_corners, box2_corners[1:] + box2_corners[:1]):
        new_intersection = list()

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        line_values = [line(t) for t in intersection]

        for s, t, s_value, t_value in zip(
                intersection, intersection[1:] + intersection[:1],
                line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return torch.tensor(0, dtype=torch.float).to(box1.device)

    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


def cut(polygon, p, q):
    # cut given convex polygon with a line defined by (p, q)
    new_polygon = list()

    # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
    # any point p with line(p) > 0 is on the "outside".
    line = Line(p, q)
    v = [line(t) for t in polygon]

    for s, t, s_v, t_v in zip(polygon, polygon[1:] + polygon[:1], v, v[1:] + v[:1]):
        if s_v <= 0:
            new_polygon.append(s)
        if s_v * t_v < 0:
            # Points are on opposite sides.
            # Add the intersection of the lines to new_intersection.
            intersection_point = line.intersection(Line(s, t))
            new_polygon.append(intersection_point)

    return new_polygon

"""
def cuts(polygons, sizes, p, q):
    vectorized polygon cut

    Arguments:
        polygons (Tensor[N, k, 2])
        sizes (Tensor[N,])
        p (Tensor[N, 2])
        q (Tensor[N, 2])

    Returns:
        new_polygons (Tensor[N, k+1, 2])
        new_sizes (Tensor[N,])
    new_polygons = list()

    # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
    # any point p with line(p) > 0 is on the "outside".
    line = Line(p, q)
    v = [line(t) for t in polygon]

    for s, t, s_v, t_v in zip(polygon, polygon[1:] + polygon[:1], v, v[1:] + v[:1]):
        if s_v <= 0:
            new_polygon.append(s)
        if s_v * t_v < 0:
            # Points are on opposite sides.
            # Add the intersection of the lines to new_intersection.
            intersection_point = line.intersection(Line(s, t))
            new_polygon.append(intersection_point)
    return new_polygons
"""

def box_inter_t(boxes1, boxes2):
    """
    Finds intersection convex polygon using sequential cut, then computes its area by counterclockwise cross product.
    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241

       Arguments:
           boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute area of intersection. They are
               expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.

       Returns:
           inter (Tensor[N, M]) pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    N = boxes1.shape[-2]
    M = boxes2.shape[-2]

    boxes1 = boxes1.reshape([-1, 4, 2])
    boxes2 = boxes2.reshape([-1, 4, 2])

    inter_len = torch.zeros([N, M]).to(boxes1.device).long()  # [N, M]
    inter_xy = torch.zeros([N, M, 8, 2]).to(boxes1.device).fill_(1e10)  # [N, M, 8, 2]

    # vectorized intersection computation
    #polygons = boxes1.unsqueeze(1).expand(-1, M, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    #cutter = boxes2.unsqueeze(0).expand(N, -1, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    #polygons = cuts(polygons, cutter[:, :, 0, :], cutter[:, :, 1, :])  # [N * M, 5, 2]
    #polygons = cuts(polygons, cutter[:, :, 1, :], cutter[:, :, 2, :])  # [N * M, 6, 2]
    #polygons = cuts(polygons, cutter[:, :, 2, :], cutter[:, :, 3, :])  # [N * M, 7, 2]
    #polygons = cuts(polygons, cutter[:, :, 3, :], cutter[:, :, 0, :])  # [N * M, 8, 2]

    for n in range(N):
        for m in range(M):
            # compute intersection
            polygon = boxes1[n, :, :].unbind(-2)  # [2,]
            polygon = cut(polygon, boxes2[m, 0, :], boxes2[m, 1, :])
            if len(polygon) <= 2: continue
            polygon = cut(polygon, boxes2[m, 1, :], boxes2[m, 2, :])
            if len(polygon) <= 2: continue
            polygon = cut(polygon, boxes2[m, 2, :], boxes2[m, 3, :])
            if len(polygon) <= 2: continue
            polygon = cut(polygon, boxes2[m, 3, :], boxes2[m, 0, :])
            if len(polygon) <= 2: continue

            inter_len[n, m] = len(polygon)
            inter_xy[n, m, :len(polygon), :] = torch.stack(polygon)

    # compute area
    inter_abc = inter_xy.reshape([N * M, 8, 2])  # [N*M, 8, 2]
    inter_bca = inter_abc.clone()
    inter_bca[:, :-1, :] = inter_abc[:, 1:, :]
    inter_bca[torch.arange(N * M), inter_len.flatten()-1, :] = inter_abc[:, 0, :]
    inter = inter_abc[:, :, 0] * inter_bca[:, :, 1] - \
            inter_abc[:, :, 1] * inter_bca[:, :, 0]

    inter_len = inter_len.flatten().unsqueeze(-1).expand([-1, 8])  # [N*M, 8]
    #inter[inter_len <= 2] = 0
    inter[inter_len <= torch.arange(8).unsqueeze(0)] = 0

    return 0.5 * inter.reshape([N, M, -1]).sum(dim=-1)  # [N, M]


def box_convex_hull(box1, box2):
    """
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
       Arguments:
           box1, box2 (Tensor[8], Tensor[8]): boxes to compute convex hull area. They are
               expected to be in (x0,y0, ..., x3,y3) format, where the corners are sorted counterclockwise.

       Returns:
           area: torch.float32
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    box1 = box1.reshape(4, 2).unbind(-2)
    box2 = box2.reshape(4, 2).unbind(-2)
    points = sorted(box1 + box2, key=lambda x:[x[0], x[1]])

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    convex_hull = lower[:-1] + upper[:-1]

    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in
                     zip(convex_hull, convex_hull[1:] + convex_hull[:1]))


def box_iou(boxes1, boxes2):
    """
    Arguments:
        boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute IoU. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.

    Returns:
        iou: [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
        union: [N, M] pairwise matrix
    """
    area1 = box_area(boxes1)  # [N,]
    area2 = box_area(boxes2)  # [M,]

    #######################################parallelization target#######################################

    tic = time.time()
    inter = torch.zeros([boxes1.shape[-2], boxes2.shape[-2]]).to(boxes1.device)  # [N, M]
    for n in range(boxes1.shape[-2]):
        for m in range(boxes2.shape[-2]):
            inter[n, m] = box_inter(boxes1[n, :], boxes2[m, :])
    print(inter.sum())
    toc = time.time()
    print(f"looped time: {toc - tic} s")

    tic = time.time()
    inter = box_inter_t(boxes1, boxes2)
    print(inter.sum())
    toc = time.time()
    print(f"torch time: {toc - tic} s")

    #######################################parallelization target#######################################

    union = area1.unsqueeze(-1) + area2.unsqueeze(-2) - inter  # [N, M]

    iou = inter / union  # [N, M]
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in corners [x0,y0, ... x3,y3] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """

    iou, union = box_iou(boxes1, boxes2)

    #######################################parallelization target#######################################
    # looped version
    hull = torch.zeros([boxes1.shape[-2], boxes2.shape[-2]]).to(boxes1.device)
    for n in range(boxes1.shape[-2]):
        for m in range(boxes2.shape[-2]):
            hull[n, m] = box_convex_hull(boxes1[n, :], boxes2[m, :])
    #######################################parallelization target#######################################

    return iou - (hull - union) / hull
