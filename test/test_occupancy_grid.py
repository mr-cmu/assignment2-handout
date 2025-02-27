import os
import unittest
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import cv2
import yaml

import os
qs_path = os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.append(qs_path)

from quadrotor_simulator_py.map_tools import OccupancyGrid, Cell, Point
from quadrotor_simulator_py.sensor_simulator import SensorSimulator
from quadrotor_simulator_py.utils import Pose
from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.visualizer.visualizer import *

def common_elements_count_loop(list1, list2):
    return sum(1 for elem in set(list1) if elem in list2)

def generate_circle_inward_poses(radius=3.5, num_points=36):
    """
    Generate positions (x, y, z) around a circle in the XY-plane
    and orientations (yaw, pitch, roll) such that each pose 
    points inward (toward the circle center) using ZYX Euler angles.
    
    :param radius: Radius of the circle (default 3.5 for diameter=7).
    :param num_points: How many discrete waypoints around the circle.
    :return: Two lists:
             positions (list of (x, y, z)), 
             orientations (list of (yaw, pitch, roll)).
    """
    positions = []
    orientations = []

    for i in range(num_points):
        # Angle from 0 to 2*pi
        theta = 2 * np.pi * i / num_points

        # Circle coordinates (XY plane, Z=0)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0
        positions.append((x, y, z))

        # ZYX Euler angles:
        # yaw around Z, pitch around Y, roll around X
        # Face inward => yaw = theta + pi
        yaw = theta + np.pi
        pitch = 0.0
        roll = 0.0
        orientations.append((yaw, pitch, roll))

    return positions, orientations

def test_conversions(config):
    og = OccupancyGrid(config)

    correct_point1 = Point(-28.0, -35.0, 20.0)
    correct_cell1 = Cell(0, 14, 44)
    correct_index1 = 1444

    student_point1 = og.index2point(correct_index1)
    student_cell1 = og.index2cell(correct_index1)
    student_index1 = og.cell2index(correct_cell1)

    assert(student_point1 == correct_point1)
    assert(student_cell1 == correct_cell1)
    assert(student_index1 == correct_index1)

    correct_point2 = Point(26.5, 2.5, 20.5)
    correct_cell2 = Cell(75, 123, 45)
    correct_index2 = 1512345

    student_point2 = og.index2point(correct_index2)
    student_cell2 = og.index2cell(correct_index2)
    student_index2 = og.point2index(correct_point2)

    assert(student_point2 == correct_point2)
    assert(student_cell2 == correct_cell2)
    assert(student_index2 == correct_index2)

    print('Map conversions success')

def test_occupancy_grid_map(DATA_DIR, config, environment_name, meshfile, visualize=True):
    sensor_simulator = SensorSimulator(config, meshfile)
    og = OccupancyGrid(config)
    viz_elements = []
    
    with open(config, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YamlError as exc:
            print(exc)
    trimmed_range_max = data['depth_camera']['trimmed_range_max']
    
    # generate poses in a circle
    circle_radius = 14.0
    num_waypoints = 20
    xyz, rpy = generate_circle_inward_poses(circle_radius, num_waypoints)
    
    # iterate over the poses, generate teh sensor observations, and raytrace
    # through the occupancy grid map
    for idx in range(0, len(xyz)):
    
        Twb = Pose()
        t = np.array([xyz[idx][0], xyz[idx][1], xyz[idx][2]+5.0])
        R = Rot3().from_euler_zyx([rpy[idx][2], rpy[idx][1], rpy[idx][0]])
        Twb.set_translation(t)
        Twb.set_rotation(R.R)
    
        world_frame_points = sensor_simulator.ray_mesh_intersect(Twb)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(world_frame_points)
    
        Twc = Twb.compose(sensor_simulator.Tbc)
   
        for w in world_frame_points:
            og.add_ray(Point().from_numpy(Twc.translation()),
                       Point().from_numpy(w), trimmed_range_max)

        if visualize:
            viz_elements.append(point_cloud)
            triad = visualize_sensor_triad(Twc)
            viz_elements.append(triad)
            line_list = line_segments_numpy(Twc.translation(), world_frame_points)
            viz_elements = viz_elements + line_list
 
    
    # Evaluation and visualization
    pts_occ, indices_occ = og.get_occupied_pointcloud()
    pts_free, indices_free = og.get_free_pointcloud()

    if visualize:
        mesh = visualize_mesh(meshfile)
        viz_elements.append(mesh)
        o3d.visualization.draw_geometries(viz_elements)
 
        voxels = numpy_to_voxel_grid(pts_occ, og.resolution)
        viz_elements.append(voxels)
        o3d.visualization.draw_geometries(viz_elements)

        voxels = numpy_to_voxel_grid(pts_free, og.resolution)
        viz_elements.append(voxels)
        o3d.visualization.draw_geometries(viz_elements)
    
    #np.savez(environment_name + "_solutions.npz", correct_occ_indices=indices_occ,
    #         correct_free_indices=indices_free)
    
    data = np.load(qs_path+"/data/" + environment_name + "_solutions.npz")
    
    common_occ_indices = common_elements_count_loop(data["correct_occ_indices"], indices_occ)
    common_free_indices = common_elements_count_loop(data["correct_free_indices"], indices_free)
    
    score = 0.0
    score += (common_occ_indices / len(data["correct_occ_indices"])) * 0.5
    score += (common_free_indices / len(data["correct_free_indices"])) * 0.5
    print("Score of " + str(round(score, 1)*100) + "%")

if __name__ == "__main__":
    test_conversions(qs_path+"/config/map.yaml")
    test_occupancy_grid_map(qs_path+"/data/", qs_path+"/config/map.yaml", "environment1", qs_path+"/mesh/environment1.ply")
    test_occupancy_grid_map(qs_path+"/data/", qs_path+"/config/map.yaml", "environment2", qs_path+"/mesh/environment2.ply")
