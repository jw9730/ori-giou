"""
Test oriented GIoU.
"""
import torch
import torch.optim as optim
import time
import numpy as np
import cv2
from util import box_ops
from util import matching


def randombox(n=1):
    boxes = np.random.rand(n, 6)
    boxes[:, 4:6] = np.random.randn(n, 2)
    return boxes


def draw_boxes(boxes, img, color=(0, 0, 0)):
    corners = 256 * box_ops.box_center_to_corners(boxes.clone().detach())  # [N, 8]
    for b in range(boxes.shape[0]):
        c = corners[b, :].unbind(-1)
        box = np.asarray([[c[0], c[1]],
                          [c[2], c[3]],
                          [c[4], c[5]],
                          [c[6], c[7]]], np.int32)
        img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=1)
    return img


def optimization_test(tgt_boxes, src_boxes, lr=1e-3, max_iter=int(1e5)):

    img_array = list()

    optimizer = optim.Adam([src_boxes], lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss_giou = 1 - box_ops.generalized_box_iou(
            box_ops.box_center_to_corners(tgt_boxes),
            box_ops.box_center_to_corners(src_boxes)
        )
        loss = loss_giou.sum()

        if i % 10 == 0:
            img = np.ones((256, 256, 3), np.uint8) * 255
            img = draw_boxes(src_boxes, img, (0, 0, 255))
            img = draw_boxes(tgt_boxes, img, (255, 0, 0))
            img = cv2.putText(img, f'lr={lr}, step: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            img_array.append(img)

            print(loss.item())

        loss.backward()
        optimizer.step()

    out = cv2.VideoWriter('box_test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (256, 256))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def get_matched_loss(tgt_boxes, src_boxes, cost_matcher):
    outputs = dict()
    outputs["pred_boxes"] = src_boxes.unsqueeze(0)  # (1, N, 5)

    target = dict()
    target["boxes"] = tgt_boxes  # (N, 5)
    targets = [target]

    indices = cost_matcher(outputs, targets)
    src_idx, tgt_idx = indices[0]  # batch=1

    src_boxes = src_boxes[src_idx]
    tgt_boxes = tgt_boxes[tgt_idx]

    loss_giou = 1 - box_ops.generalized_box_iou(
        box_ops.box_center_to_corners(tgt_boxes),
        box_ops.box_center_to_corners(src_boxes)
    )
    loss = torch.diag(loss_giou).sum()
    return loss


def matching_test(tgt_boxes, src_boxes, cost_bbox=0, cost_giou=1, lr=1e-3, max_iter=int(1e5)):

    img_array = list()

    matcher = matching.HungarianMatcher(cost_bbox, cost_giou)
    optimizer = optim.Adam([src_boxes], lr=lr)
    for i in range(max_iter):
        optimizer.zero_grad()

        loss = get_matched_loss(tgt_boxes, src_boxes, matcher)

        if i % 1 == 0:
            img = np.ones((256, 256, 3), np.uint8) * 255
            img = draw_boxes(tgt_boxes, img, (255, 0, 0))
            img = draw_boxes(src_boxes, img, (0, 0, 255))
            img = cv2.putText(img, f'lr={lr}, step: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            img_array.append(img)

            print(loss.item())

        loss.backward()
        optimizer.step()

    out = cv2.VideoWriter('box_set_test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (256, 256))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)

    # bounding box representation:
    # [x_c, y_c, w, h, c, s]
    # (x_c, y_c, w, h) are assumed to be in range 0-1 for visualization (other values are fine for computation)
    # (c, s) are un-normalized cosine / sine values, only illegal case is c == s == 0

    # box optimization test
    """
    np.random.seed(777)
    boxes1 = randombox(1)
    boxes2 = randombox(1)
    target = torch.tensor(boxes1, dtype=torch.float32)
    source = torch.tensor(boxes2, dtype=torch.float32, requires_grad=True)
    optimization_test(target, source, lr=1e-3, max_iter=int(1e3))
    """

    # box set optimization test
    np.random.seed(123)
    m = 3
    max_iter = int(350)

    boxes1 = np.zeros((m ** 2, 6))
    for i in range(m):
        for j in range(m):
            boxes1[m * i + j, :] = [(1 + i) / (1 + m), (1 + j) / (1 + m), 0.1 * (0.1 + 2 * np.random.rand()), 0.1 * (0.1 + 2 * np.random.rand()), np.random.randn(), np.random.randn()]
    boxes1 = randombox(m ** 2)
    boxes2 = randombox(m ** 2)
    target = torch.tensor(boxes1, dtype=torch.float32)
    source = torch.tensor(boxes2, dtype=torch.float32, requires_grad=True)

    tic = time.time()
    matching_test(target, source, cost_bbox=0, cost_giou=1, lr=1e-2, max_iter=max_iter)
    toc = time.time()

    # in local, 500-step optimization of 36 x 36 input took 2-3 min
    print(f"matching_test: {toc - tic} sec taken for {max_iter}-step optimization of {m**2} x {m**2} input")
