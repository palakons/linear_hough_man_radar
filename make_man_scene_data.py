import pickle, os

from requests import get
import numpy as np
import matplotlib.pyplot as plt
import time
from pyquaternion import Quaternion
from truckscenes import TruckScenes
import pypcd4, torch
from tqdm import trange
from pytorch3d.transforms import axis_angle_to_matrix


def get_camera_token(trucksc, seq_id=0, i_frame=5, camera_channel="CAMERA_LEFT_FRONT"):
    i = 0

    sample_token = trucksc.scene[seq_id]["first_sample_token"]

    while i < i_frame:
        sample_token = trucksc.get("sample", sample_token)["next"]
        i += 1

    sample_record = trucksc.get("sample", sample_token)
    camera_token = sample_record["data"][camera_channel]
    return camera_token


def get_rtk_man_ego(camera_token):
    """
    Get the rotation, translation, intrinsics and image filename from a camera sample data record.
    The Dataset schema says 'All extrinsic parameters are given with respect to the ego vehicle body frame.'

    Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle. All extrinsic parameters are given with respect to the ego vehicle body frame. All camera images come undistorted and rectified.

    calibrated_sensor {
        "token":                   <str> -- Unique record identifier.
        "sensor_token":            <str> -- Foreign key pointing to the sensor type.
        "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
        "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
        "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
    }
    """
    # 1. forward or backward transform?
    cam = trucksc.get("sample_data", camera_token)
    ego_pose = trucksc.get(
        "ego_pose", cam["ego_pose_token"]
    )  # Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system

    cs_record = trucksc.get(
        "calibrated_sensor", cam["calibrated_sensor_token"]
    )  # All extrinsic parameters are given with respect to the ego vehicle body frame.
    r = Quaternion(
        cs_record["rotation"]
    ).rotation_matrix  # Quaternion() accepts [w, x, y, z] format
    # print("translation:", cs_record['translation']) #row vector
    t = np.array(cs_record["translation"])  #

    r_ego = Quaternion(ego_pose["rotation"]).rotation_matrix
    t_ego = np.array(ego_pose["translation"])

    k = np.array(cs_record["camera_intrinsic"])
    img_file = cam["filename"]
    return {
        "rotation": r,
        "translation": t,
        "intrinsics": k,
        "image_file": img_file,
        "rotation_ego": r_ego,
        "translation_ego": t_ego,
    }


# go thorugh each seq_id and frame_id and save the r,t, filename for all 6 radars to a json file

trucksc_file_root = "/data/palakons/new_dataset/MAN/mini/man-truckscenes"
json_output_file = "/home/palakons/linear_hough_man_radar/public/man_scene_data.json"
trucksc = TruckScenes("v1.0-mini", trucksc_file_root, True)

data = []
for seq_i, scene in enumerate(trucksc.scene):
    print(f"Processing sequence {seq_i}")

    for i_frame in range(40):
        print(f"Processing frame {i_frame}")

        rtk_data = {"seq_id": seq_i, "frame_id": i_frame}

        for radar_ch in [
            "RADAR_LEFT_FRONT",
            "RADAR_LEFT_BACK",
            "RADAR_RIGHT_FRONT",
            "RADAR_RIGHT_BACK",
            "RADAR_LEFT_SIDE",
            "RADAR_RIGHT_SIDE",
        ]:
            camera_token = get_camera_token(trucksc, seq_i, i_frame, radar_ch)
            rtk_data[radar_ch] = get_rtk_man_ego(camera_token)
            for k in rtk_data[radar_ch]:
                if isinstance(rtk_data[radar_ch][k], np.ndarray):
                    rtk_data[radar_ch][k] = rtk_data[radar_ch][k].tolist()

        for cam_ch in [
            "CAMERA_LEFT_FRONT",
            "CAMERA_LEFT_BACK",
            "CAMERA_RIGHT_FRONT",
            "CAMERA_RIGHT_BACK",
        ]:
            camera_token = get_camera_token(trucksc, seq_i, i_frame, cam_ch)
            rtk_data[cam_ch] = get_rtk_man_ego(camera_token)
            for k in rtk_data[cam_ch]:
                if isinstance(rtk_data[cam_ch][k], np.ndarray):
                    rtk_data[cam_ch][k] = rtk_data[cam_ch][k].tolist()
        data.append(rtk_data)

# save to json
with open(json_output_file, "w") as f:
    import json

    json.dump(data, f, indent=4)
