from scipy.spatial import distance as dist
import numpy as np 
import math

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="int32")

def order_points_quadrangle(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    vector_0 = np.array(bl-tl)
    vector_1 = np.array(rightMost[0]-tl)
    vector_2 = np.array(rightMost[1]-tl)
    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]
    return np.array([tl, tr, br, bl], dtype="int32")

from functools import reduce
import operator
import math
def order_points_tuple(pts):
    pts = pts.tolist()
    coords = []
    for elem in pts:
        coords.append(tuple(elem))
    center = tuple(map(operator,truediv, reduce(lambda x, y:map(operator.add, x, y), coords), [len(coords)] * 2))
    output = sorted(coords, key=lambda coords: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coords, center))[::-1]))) % 360, reverse=True)
    res = []
    for elem in output:
        res.append(list(elem))
    return np.array(res, dtype="int32")
points = np.array([[54,20],[39,48],[117,52],[121,21]])
print(order_points(points))
pt = np.array([703,211,754,283,756,223,747,212]).reshape(4,2)
print(order_points(pt))
print(order_points_tuple(pt))