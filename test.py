"""
Test oriented GIoU.
"""
import torch
import torch.optim as optim
import math
import numpy as np
import cv2
from util import box_ops
from util import matching


def randombox(n=1):
    boxes = np.random.rand(n, 5)
    boxes[:, 4] = (boxes[:, 4] - 0.5) * math.pi / 2
    return boxes


def draw_boxes(boxes, img, color=(0, 0, 0)):
    corners = 512 * box_ops.box_center_to_corners(boxes.clone().detach())  # [N, 8]
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

        if i % 100 == 0:
            img = np.ones((512, 512, 3), np.uint8) * 255
            img = draw_boxes(src_boxes, img, (0, 0, 255))
            img = draw_boxes(tgt_boxes, img, (255, 0, 0))
            img = cv2.putText(img, f'lr={lr}, step: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            img_array.append(img)

            print(loss.item())

        loss.backward()
        optimizer.step()

    out = cv2.VideoWriter('opt_test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 512))
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

    """
    print(src_idx)
    print(src_boxes[src_idx])
    print(tgt_idx)
    print(tgt_boxes[tgt_idx])
    """

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
            img = np.ones((512, 512, 3), np.uint8) * 255
            img = draw_boxes(src_boxes, img, (0, 0, 255))
            img = draw_boxes(tgt_boxes, img, (255, 0, 0))
            img = cv2.putText(img, f'lr={lr}, step: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            img_array.append(img)

            print(loss.item())

        loss.backward()
        optimizer.step()

    out = cv2.VideoWriter('match_test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (512, 512))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    # x_c, y_c, w, h, theta (-pi <= theta < pi)
    # scale: 0-1

    # optimization test
    #boxes1 = [[.7, .7, .2, .1, math.pi/7]]
    #boxes2 = [[.2, .2, .3, .2, math.pi/3]]

    """
    np.random.seed(123)
    boxes1 = randombox(1)
    boxes2 = randombox(1)
    target = torch.tensor(boxes1, dtype=torch.float32)
    source = torch.tensor(boxes2, dtype=torch.float32, requires_grad=True)
    optimization_test(target, source, lr=1e-3, max_iter=int(2e3))
    """

    # matching test
    #np.random.seed(123)
    boxes1 = np.zeros((16, 5))
    for i in range(4):
        for j in range(4):
            boxes1[4 * i + j, :] = [(1 + i)/5, (1 + j)/5, 0.1, 0.1, np.random.rand() - 0.5]
    #boxes1 = randombox(16)
    boxes2 = randombox(16)
    target = torch.tensor(boxes1, dtype=torch.float32)
    source = torch.tensor(boxes2, dtype=torch.float32, requires_grad=True)
    matching_test(target, source, cost_bbox=0, cost_giou=1, lr=2e-2, max_iter=int(1e2))
