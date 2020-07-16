# GIoU for oriented bounding boxes

This code is a PyTorch-based, vectorized, and differentiable implementation of [Generalized union over intersection (GIoU)](https://giou.stanford.edu/) loss for rotated bounding boxes.

The base code for axis-aligned box manipulation and optimal matching was taken from [DETR](https://github.com/facebookresearch/detr) by Facebook Research.

For use cases, see `test.py`.

# Demo

Red boxes are optimized by gradient descent in respect to blue boxes.

1. Single box

![single](demo/single.gif)

2. Regular boxes on grid

![grid_regular](demo/grid_regular.gif)

3. Irregular boxes on grid

![grid_irregular](demo/grid_irregular.gif)

4. Random boxes

![randombox1](demo/random.gif)