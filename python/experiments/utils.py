from collections import defaultdict
import math
import random
from multiprocessing import Pool

import vbfdml2 as vbfdml

from measurement import Measurement
from verify_prediction import compute_ray_distance, is_point_inside, prediction_error
import itertools
import numpy as np
from sklearn.cluster import DBSCAN

AVG_RHO = 300

def error_upper_bound(n):
    return 2 * (3 * AVG_RHO / (4 * math.pi)) ** (1/3) / n

def generate_measurements(segments, p, k, dtheta, d_obstacles=None):
    measurements = []
    delta_theta = 0
    for _ in range(k):
        dist, hit_d_osbt = compute_ray_distance(p.x, p.y, p.theta + delta_theta, segments, d_obstacles=d_obstacles)
        m = vbfdml.Measurement(0, 0, delta_theta, dist, hit_d_osbt)
        measurements.append(m)
        delta_theta += dtheta
    return measurements

def load_polygon(filename):
    points = []
    minx, maxx, miny, maxy = None, None, None, None
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            x, y = line.split()
            x = float(x); y = float(y)
            if minx is None or x < minx:
                minx = x
            if maxx is None or x > maxx:
                maxx = x
            if miny is None or y < miny:
                miny = y
            if maxy is None or y > maxy:
                maxy = y
            points.append((x,y))
    points.append(points[0])

    segments = []
    for i in range(len(points) - 1):
        segments.append(
            vbfdml.Segment(*points[i], *points[i+1])
        )
    
    width = maxx - minx
    height = maxy - miny
    depth = 2.01 * math.pi
    x0 = (minx + maxx) / 2
    y0 = (miny + maxy) / 2
    theta0 = 0
    be = vbfdml.BoxExtent(width, height, depth, x0, y0, theta0)

    return segments, be


def sample_random_point(segments, be):
    while True:
        x = random.uniform(be.x0 - be.width / 2, be.x0 + be.width / 2)
        y = random.uniform(be.y0 - be.height / 2, be.y0 + be.height / 2)
        theta = random.uniform(be.theta0 - be.depth / 2, be.theta0 + be.depth / 2)
        if is_point_inside(x, y, segments):
            p = vbfdml.Prediction()
            p.x = x; p.y = y; p.theta = theta
            return p
        
def generate_dynamic_obsticales(segments, be, p, d_obst_num, radius=0.1):

    def generate_random_path(x, y, path_len=10):
        path = []
        # for _ in range(path_len):
        #     x += random.uniform(-0.1, 0.1)
        #     y += random.uniform(-0.1, 0.1)
        #     while not is_point_inside(x, y, segments):
        #         x += random.uniform(-0.1, 0.1)
        #         y += random.uniform(-0.1, 0.1)
        #     path.append((x, y))
        return path
    
    d_obst_arr = []
    while len(d_obst_arr) < d_obst_num:
        x = random.uniform(be.x0 - be.width / 2, be.x0 + be.width / 2)
        y = random.uniform(be.y0 - be.height / 2, be.y0 + be.height / 2)
        if is_point_inside(x, y, segments) and math.dist((x, y), (p.x, p.y)) > 2 * radius:
            tmp = vbfdml.DynamicObstacle(x, y, radius, generate_random_path(x, y)) 
            d_obst_arr.append(tmp)
            
    return d_obst_arr
    # return None

def run_vbfdml(segments, be, measurements, n, bUseGPU=True):
    preprocess, mask = vbfdml.preprocess_scene(n, be, segments, bUseGPU)
    ms = []
    for measurement in measurements:
        m = vbfdml.marching_voxels(preprocess, be, measurement.dx, measurement.dy, measurement.dtheta, measurement.dist, bUseGPU)
        m = vbfdml.conv3d(m, bUseGPU)
        ms.append(m)
    
    # divide to all permutations of measurements of size k choose m
    # intersect all subset of 4 manifolds and take union of all intersections (mass centers)
    # from voxels take measurements in the same manner as before, and compare to the original measurements
    # find the measurements that have the largest error from the original measuremnets
    # remove manifolds created from the false original maeasurements

    isect = ms[0]
    for m in ms[1:]:
        isect = vbfdml.do_intersect(isect, m, bUseGPU)
    isect = vbfdml.do_intersect(isect, mask, bUseGPU)

    return vbfdml.predict(isect, be)
    # return vbfdml.run_vbfdml(segments, be, n, measurements, bUseGPU)

def run_improved_vbfdml(segments, be, measurements, n, k, bUseGPU=True):
    l = 6
    epsilon = 0.5 * 4
    preds_union = vbfdml.run_improved_vbfdml(segments, be, measurements, n, l, epsilon)

    if not preds_union:
        return []
    
    # remove predictions that are close in value
    preds_union = np.array([(pred.x, pred.y, pred.theta) for pred in preds_union])

    # Apply DBSCAN algorithm
    clustering = DBSCAN(eps=epsilon / 20.0 * 50.0 / n, min_samples=1).fit(preds_union)

    # Get the labels of the clusters
    labels = clustering.labels_

    # Calculate the center of each cluster
    cluster_centers = []
    for label in set(labels):
        if label != -1:  # Ignore noise (if any)
            cluster_points = preds_union[labels == label]
            cluster_center = cluster_points.mean(axis=0)
            cluster_centers.append(cluster_center)

    # Convert cluster centers back to predictions
    cluster_center_preds = [vbfdml.Prediction(x=center[0], y=center[1], theta=center[2]) for center in cluster_centers]

    return cluster_center_preds

def pred_dist_3d(p1, p2, be):
    return (
        ((p1.x - p2.x) / be.width) ** 2 +
        ((p1.y - p2.y) / be.height) ** 2 + 
        ((p1.theta - p2.theta) / be.depth) ** 2
    ) ** 0.5


def guess_best_prediction(preds, segments, measurements):
    best_pred = None
    best_err = None
    for pred in preds:
        err = prediction_error(pred, segments, measurements)
        if err is None:
            continue
        if best_err is None or err < best_err:
            best_err = err
            best_pred = pred
    return best_pred, best_err

def actual_best_prediction(preds, gt, be):
    best_pred = None
    best_err = None
    for pred in preds:
        err = pred_dist_3d(pred, gt, be)
        if best_err is None or err < best_err:
            best_err = err
            best_pred = pred
    return best_pred, best_err



def generate_random_polygon(n, angle_err=0.1, radius_err=0.2, radius_mean=1.0):
    """
    Creates a random (simple) polygon
    Adapted from: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    """
    # Generate angles (with errors), where the mean in 2pi/n
    angles = []
    for _ in range(n):
        angle = random.uniform(2*math.pi / n - angle_err, 2 * math.pi / n + angle_err)
        angles.append(angle)
    
    # Normalize angles so that their sum is 2pi
    sum_angles = sum(angles)
    for i in range(n):
        angles[i] = angles[i] * 2 * math.pi / sum_angles 
    
    # Generate the points
    points = []
    theta = 0
    for i in range(n):
        theta += angles[i]
        radius = random.gauss(radius_mean, radius_err)
        if radius > 2 * radius_mean:
            radius = 2 * radius_mean
        points.append((radius * math.cos(theta), radius * math.sin(theta)))
    points.append(points[0])

    segments = []
    for i in range(len(points) - 1):
        segments.append(
            vbfdml.Segment(*points[i], *points[i+1])
        )
    be = vbfdml.BoxExtent(2.2, 2.2, math.pi * 2, 0.0, 0.0, 0.0)

    return segments, be
    
