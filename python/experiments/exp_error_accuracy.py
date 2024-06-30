import os
import math
import time
import argparse

import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import *

NUM_EXPERIMENTS = 1
USE_GPU = True
RUN_IMPROVED_VBFDML = True
WITH_DYNAMIC_OBSTACLES = True
DYNAMIC_OBSTACLES_RADIUS = 0.1
# NUM_OF_DYNAMIC_OBSTACLES = 10
DRAW_SIM = False

def run_test(segments, be, n, k, dtheta, num_of_dynamic_obstacles, bUseGPU=True, draw_sim=False):
    p = sample_random_point(segments, be)
    d_obstacles = generate_dynamic_obsticales(segments, be, p, num_of_dynamic_obstacles, DYNAMIC_OBSTACLES_RADIUS) if WITH_DYNAMIC_OBSTACLES else None
    measurements = generate_measurements(segments, p, k, dtheta, d_obstacles) 
    false_measurements = sum([1 for m in measurements if m.d_obstacle == True])

    if draw_sim:
        draw_simulation(p, segments, measurements, d_obstacles)

    preds = run_improved_vbfdml(segments, be, measurements, n, k, bUseGPU) if RUN_IMPROVED_VBFDML else run_vbfdml(segments, be, measurements, n, bUseGPU)

    # if draw_sim:
    #     draw_simulation(p, segments, measurements, d_obstacles, f_measurements_id)

    if len(preds) == 0:
        return {
            'hueristic_success': False,
            'best_success': False,
            'hueristic_err': False,
            'best_err': False,
            'avg_cnt': 0,
            'false_measurements': false_measurements
        }

    # avg_cnt = 0
    # for pred in preds:
    #     avg_cnt += pred.cnt
    # avg_cnt /= len(preds)
    avg_cnt = len(preds)

    heuristic_pred, _ = guess_best_prediction(preds, segments, measurements)
    best_pred, _ = actual_best_prediction(preds, p, be)

    heuristic_err = pred_dist_3d(p, heuristic_pred, be)
    best_err = pred_dist_3d(p, best_pred, be)

    eps_thresh = error_upper_bound(n) * 2
    return {
        # 'hueristic_success': heuristic_err <= eps_thresh,
        'best_success': best_err <= eps_thresh,
        'hueristic_err': heuristic_err,
        'best_err': best_err,
        'avg_cnt': avg_cnt,
        'false_measurements': false_measurements
    }

def draw_simulation(p: vbfdml.Predict, segments: list[vbfdml.Segment], measurements: list[vbfdml.Measurement], d_obstacles: list[vbfdml.DynamicObstacle] = None, f_measurements_id: list[int] = None):
    # Draw the polygon 
    fig, ax = plt.subplots()
    for segment in segments:
        ax.plot([segment.x1, segment.x2], [segment.y1, segment.y2], 'k-')
    ax.set_aspect('equal')

    # Add arrows of measurements
    for i, m in enumerate(measurements):
        if f_measurements_id is not None and i in f_measurements_id:
            ax.arrow(p.x, p.y, m.dist * math.cos(p.theta + m.dtheta), m.dist * math.sin(p.theta + m.dtheta), head_width=0.005, head_length=0.001, fc='g', ec='g')
        else:
            ax.arrow(p.x, p.y, m.dist * math.cos(p.theta + m.dtheta), m.dist * math.sin(p.theta + m.dtheta), head_width=0.005, head_length=0.001, fc='r', ec='r')

    # Draw dynamic obstacles
    if d_obstacles is not None:
        for obstacle in d_obstacles:
            ax.add_artist(plt.Circle((obstacle.x, obstacle.y), obstacle.radius, color='r', fill=True))
    ax.plot(p.x, p.y, 'bo')

    plt.show()

def run_all_tests(segments, be, Ns, k, dtheta, num_of_dynamic_obstacles):
    d = {'n': [], 'best_success': [], 'hueristic_err': [], 'best_err': [], 'avg_cnt': [], 'avg_time': [], 'avg_false_measurements': []}
    for n in Ns:
        print(n)
        hueristic_success = 0
        best_success = 0
        hueristic_err = 0
        best_err = 0
        false_measurements = 0
        avg_cnt = 0
        t0 = time.time()
        for _ in tqdm.tqdm(range(NUM_EXPERIMENTS)):
            res = run_test(segments, be, n, k, dtheta, num_of_dynamic_obstacles, USE_GPU, draw_sim=DRAW_SIM)
            # if res['hueristic_success']:
            #     hueristic_success += 1
            hueristic_err += res['hueristic_err']
            if res['best_success']:
                best_success += 1
            best_err += res['best_err']
            avg_cnt += res['avg_cnt']
            false_measurements += res['false_measurements']
        t1 = time.time()
        
        d['n'].append(n)
        # d['hueristic_success'].append(hueristic_success / NUM_EXPERIMENTS)
        d['best_success'].append(best_success / NUM_EXPERIMENTS)
        d['hueristic_err'].append(hueristic_err / NUM_EXPERIMENTS)
        d['best_err'].append(best_err / NUM_EXPERIMENTS)
        d['avg_cnt'].append(avg_cnt / NUM_EXPERIMENTS)
        d['avg_time'].append((t1 - t0) / NUM_EXPERIMENTS)
        d['avg_false_measurements'].append(false_measurements / NUM_EXPERIMENTS)
    
    return pd.DataFrame(d)
    
def normalize_map_polygon(maps_polygons, workspace, scale):
    segments, be = maps_polygons[workspace]
    for seg in segments:
        seg.x1 *= scale
        seg.x2 *= scale
        seg.y1 *= scale
        seg.y2 *= scale
    be.width *= scale
    be.height *= scale
    maps_polygons[workspace] = (segments, be)


if __name__ == "__main__":
    
    maps_filenames = ["resources/maps/lab_lidar.poly", "resources/maps/checkpoint.poly"]
    workspaces = ["lab", "floor-plan"]
    maps_polygons = {workspace: load_polygon(filename) for workspace, filename in zip(workspaces, maps_filenames)}
    maps_polygons["random"] = generate_random_polygon(70)

    # normalize too small/too big scenes
    normalize_map_polygon(maps_polygons, "floor-plan", 0.25)
    normalize_map_polygon(maps_polygons, "random", 2)

    num_of_dynamic_obstacles = [10, 30]

    for num in num_of_dynamic_obstacles:
        for workspace, polygon in maps_polygons.items():
            segments, be = polygon
            Ns = [50, 75, 100, 125, 150, 175, 200]
            k = 10
            dtheta = 2 * math.pi / k
            df = run_all_tests(segments, be, Ns, k, dtheta, num)
            df.insert(0, 'workspace', workspace)

            # Write results to csv file
            folder = f"python/experiments/results/{k}_samples"
            file_name = f"exp_d_obstacles_{workspace}_{num}" if WITH_DYNAMIC_OBSTACLES else "exp_no_d_obstacles_{workspace}"
            file_name += "_improved" if RUN_IMPROVED_VBFDML else ""
            file_name += ".csv"
            os.makedirs(folder, exist_ok=True)
            outfile = os.path.join(folder, file_name)
            df.to_csv(outfile, index=False)
    
    print("Done")

        
