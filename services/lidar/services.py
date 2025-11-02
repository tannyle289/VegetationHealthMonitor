import json
import os
from typing import Callable, Any

import torch
import open3d as o3d
import cv2
import numpy as np
from scipy.spatial import ConvexHull

from config.config import Config
from os.path import join as pjoin

depth_model = Callable[..., Any]
transform = Callable[..., Any]
_config = Config()


def load_model():
    global depth_model
    global transform
    model_type = "DPT_Large"
    depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
    depth_model.eval()

    # Set up dpt transform
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    return model_type


def _estimate_depth(img, mask):
    mask_bin = (mask > 127).astype(np.uint8)  # 1 for tree, 0 for background

    input_tensor = transform(img)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        depth = depth_model(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = depth.cpu().numpy()
    # Apply mask to remove background
    depth_masked = depth * mask_bin

    return depth_masked


def _produce_point_cloud(img, depth_masked, save_dir):
    save_path = pjoin(save_dir, "tree_cropped.pcd")

    h, w = img.shape[:2]
    fx = fy = 700.0
    cx, cy = w / 2.0, h / 2.0

    ys, xs = np.where(depth_masked > 0)  # only tree pixels
    z = depth_masked[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float32)

    colors = (img[ys, xs] / 255.0).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

    return points, save_path


def _extract_3d_features(points: np.ndarray):
    features = {}

    # Case there's no point cloud
    if len(points) == 0:
        return {
            "height_max": 0.0,
            "height_mean": 0.0,
            "tree_volume_d3": 0.0,
            "point_density": 0.0,
        }

    # Height features
    z_coords = points[:, 2]
    features["height_max"] = float(np.max(z_coords))
    features["height_mean"] = float(np.mean(z_coords))

    # Volume metrics
    # Need to compute 3D convex hull since point clouds are usually unevenly sampled
    try:
        if len(points) >= 4:  # ConvexHull requires at least 4 points
            hull = ConvexHull(points)
            features["tree_volume_d3"] = float(hull.volume)
        else:
            features["tree_volume_d3"] = 0.0
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        features["tree_volume_d3"] = 0.0

    # Point density metrics
    volume = features["tree_volume_d3"]
    if volume > 0:
        features["point_density"] = float(len(points) / volume)
    else:
        features["point_density"] = 0.0

    return features


def process(
    image_cropped_path: str,
    mask_cropped_path: str,
    save_dir: str,
    save_json=False,
    tree_id=0,
    time_stamp="",
):
    if depth_model is None or transform is None:
        raise ModuleNotFoundError("Model not loaded")

    if not save_dir:
        save_dir = pjoin(_config.DATA_PATH, "processed", f"{tree_id}-{time_stamp}")

    # Read cropped img
    img_bgr = cv2.imread(image_cropped_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read {image_cropped_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Read cropped mask
    mask = cv2.imread(mask_cropped_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read {mask_cropped_path}")

    # Estimate depth
    depth_masked = _estimate_depth(img, mask)

    # Convert into point cloud
    points, pcd_path = _produce_point_cloud(img, depth_masked, save_dir)
    features = _extract_3d_features(points)

    result = {
        "tree_id": tree_id,
        "timestamp": time_stamp,
        "save_dir": str(save_dir),
        "point_cloud_path": pcd_path,
        "features_3d": features,
    }

    if save_json:
        json_path = os.path.join(save_dir, "result_3d.json")
        with open(json_path, "w") as f:
            json.dump(result, f)

    return result
