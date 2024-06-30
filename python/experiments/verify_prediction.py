import math

import vbfdml2 as vbfdml
import numpy as np

from measurement import Measurement


def compute_ray_distance(x, y, theta, segments, return_cnt=False, d_obstacles=None):

    # if theta > math.pi:
    #     theta -= 2 * math.pi
    # if theta < -math.pi:
    #     theta += 2 * math.pi

    ray_distance = None
    cnt = 0
    ray_hit_d_obstacle = False

    # check for collision with dynamic obstacles
    if d_obstacles is not None:
        for obstacle in d_obstacles:
            isects = ray_circle_intersection(x, y, theta, obstacle.x, obstacle.y, obstacle.radius) # TODO check in accordance to obstacles path
            for isect in isects:
                if isect is not None:
                    d = np.linalg.norm(np.array([x, y]) - isect)
                    if ray_distance is None or d < ray_distance:
                        ray_distance = d
                        ray_hit_d_obstacle = True
    
    for segment in segments:
        q = np.array([x, y])
        v3 = np.array([-math.sin(theta), math.cos(theta)])

        s1 = np.array([segment.x1, segment.y1])
        s2 = np.array([segment.x2, segment.y2])

        v1 = q - s1
        v1_ = np.array([v1[1], -v1[0]])
        v2 = s2 - s1

        v2_dot_v3 = np.dot(v2, v3)
        if (abs(v2_dot_v3) < 0.001):
            continue

        t1 = np.dot(v2, v1_) / v2_dot_v3
        t2 = np.dot(v1, v3) / v2_dot_v3
        if t1 < 0 or t2 < 0 or t2 > 1:
            continue

        cnt += 1
        isect = s1 + t2 * v2
        d = np.linalg.norm(q - isect)
        if ray_distance is None or d < ray_distance:
            ray_distance = d
            ray_hit_d_obstacle = False

    if return_cnt:
        return ray_distance, cnt
    
    if ray_distance is None:
        ray_distance = np.inf
    
    return ray_distance, ray_hit_d_obstacle

def ray_circle_intersection(x, y, theta, h, k, r):
    # Define the line
    p1 = np.array([x, y])
    dp = np.array([math.cos(theta), math.sin(theta)])

    # Calculate the vector from the circle's center to the line's origin
    f = p1 - np.array([h, k])

    # Calculate the quadratic formula coefficients
    a = np.dot(dp, dp)
    b = 2 * np.dot(f, dp)
    c = np.dot(f, f) - r**2

    # Solve the quadratic equation for t
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None  # No intersection
    else:
        t1 = (-b - np.sqrt(discriminant)) / (2*a)
        t2 = (-b + np.sqrt(discriminant)) / (2*a)

        # If t is positive, the intersection point is in the direction of the line
        if t1 > 0:
            intersection1 = p1 + t1 * dp
        else:
            intersection1 = None

        if t2 > 0:
            intersection2 = p1 + t2 * dp
        else:
            intersection2 = None

        return intersection1, intersection2
    

def is_point_inside(x, y, segments):
    _, cnt = compute_ray_distance(x, y, 0, segments, True)
    return (cnt % 2) == 1


def prediction_error(pred, segments, measurements):
    error = 0
    for measurement in measurements:
        # Transform pose to odometry frame
        dx = measurement.dx; dy = measurement.dy; dtheta = measurement.dtheta
        x = pred.x + dy * math.cos(pred.theta + dtheta) - dx * math.sin(pred.theta + dtheta)
        y = pred.y + dy * math.sin(pred.theta + dtheta) + dx * math.cos(pred.theta + dtheta)
        theta = pred.theta + dtheta
    
        # Compute ray-segment intersection
        ray_distance, hit_d_obst = compute_ray_distance(x, y, theta, segments)
        if ray_distance is None:
            return None
        
        # Append MSE distance error
        error += (ray_distance - measurement.dist) ** 2
        # error += abs(ray_distance - measurement.dist)
    
    return error