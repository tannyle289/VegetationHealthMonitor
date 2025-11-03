import json
import os

import cv2
import numpy as np


from config.config import Config
from os.path import join as pjoin
from ultralytics import YOLO

seg_model: YOLO | None = None
_config = Config()


def _extract_2d_features(mask: np.ndarray, bbox: list, image: np.ndarray) -> dict:
    features = {}
    x1, y1, x2, y2 = bbox

    h, w = max(1, y2 - y1), max(1, x2 - x1)
    # Basic Geometric features
    features["tree_area_px2"] = int(np.sum(mask))
    features["ar"] = float(h / w)

    # Color features
    # Standardized to [0, 1]
    masked_region = image[mask == 1].astype(np.float32) / 255.0

    # Case there's no mask
    if len(masked_region) == 0:
        return {
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "ngrdi": 0.0,
            "vari": 0.0,
            "exg": 0.0,
        }

    # Mean RGB
    mean_r = float(np.mean(masked_region[:, 2]))  # cv2 uses BGR
    mean_g = float(np.mean(masked_region[:, 1]))
    mean_b = float(np.mean(masked_region[:, 0]))

    features["mean_r"] = mean_r
    features["mean_g"] = mean_g
    features["mean_b"] = mean_b

    # NGRDI = (G - R) / (G + R + esp)
    features["ngrdi"] = float((mean_g - mean_r) / (mean_g + mean_r + 1e-6))

    # VARI = (G − R) / (G + R − B + esp)
    features["vari"] = float((mean_g - mean_r) / (mean_g + mean_r - mean_b + 1e-6))

    # ExG = 2G - R - B
    features["exg"] = float(2 * mean_g - mean_r - mean_b)

    return features


def _crop_image(mask_binary, org_img, box, save_dir):
    x1, y1, x2, y2 = box
    masked_img = np.zeros_like(org_img)
    masked_img[mask_binary == 1] = org_img[mask_binary == 1]

    cropped_masked_img = masked_img[y1:y2, x1:x2]
    cropped_mask = mask_binary[y1:y2, x1:x2]

    # Save cropped image
    crop_filename = "tree_cropped.png"
    crop_path = pjoin(save_dir, crop_filename)
    cv2.imwrite(crop_path, cropped_masked_img)

    # Save cropped mask
    """
    TODO: Personally think this logic is not optimized,
    my idea was just using cropped mask but if so,
    the cropped is not clean rectangle (which is weird)
    -> gotta be improved later
    """
    mask_filename = "mask_cropped.png"
    mask_path = pjoin(save_dir, mask_filename)
    cv2.imwrite(mask_path, (cropped_mask * 255).astype(np.uint8))

    return crop_path, mask_path


def load_model():
    global seg_model
    seg_model = YOLO(_config.SEG_MODEL_CHECKPOINT_PATH)
    return seg_model.model_name


def start_training():
    model = YOLO("../../models/yolo11n-seg.pt")
    model.train(
        data="../../datasets/urban-street_tree_YOLO/data_config.yaml",
        epochs=_config.SEG_EPOCHS,
        batch=_config.BATCH_SIZE,
        workers=_config.WORKERS,
        project=_config.SEG_FOLDER_PATH,
        name=_config.SEG_MODEL_TRAINING_PATH,
    )


def infer_with_tracking(input_path: str, save_dir: str = None):
    if seg_model is None:
        raise ModuleNotFoundError("Model not loaded")

    if not save_dir:
        save_dir = pjoin(_config.SEG_FOLDER_PATH, "runs", "tracks")

    results = seg_model.track(source=input_path, save=True, save_dir=save_dir)

    return {
        "result_path": results[0].path,
        "save_dir": save_dir,
        "tracked_frames_count": len(results),
    }


def predict(
    input_path: str, save_dir: str = None, tree_id=0, time_stamp="", save_json=False
):
    if seg_model is None:
        raise ModuleNotFoundError("Model not loaded")

    if not save_dir:
        save_dir = pjoin(_config.DATA_PATH, "processed", f"{tree_id}-{time_stamp}")

    results = seg_model.predict(
        source=input_path,
        save=True,
        project=_config.DATA_PATH,
        name=pjoin("processed", f"{tree_id}-{time_stamp}"),
    )

    pred = results[0]

    # Crop img
    org_img = pred.orig_img
    h, w = org_img.shape[:2]

    # Get boxes and masks
    boxes = pred.boxes.xyxy.cpu().numpy()
    scores = pred.boxes.conf.cpu().numpy()
    classes = pred.boxes.cls.cpu().numpy()

    # Assume that the ByteTrack does correctly -> it returns only 1 detection
    # For simplicity I get box with the highest conf
    best_idx = scores.argmax()
    box = boxes[best_idx]

    if hasattr(pred, "masks") and pred.masks is not None:
        masks = pred.masks.xy
    else:
        masks = []

    predictions = {}

    x1, y1, x2, y2 = box.astype(int).tolist()

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    # Case invalid box after clipping
    if x2 <= x1 or y2 <= y1:
        return {"box": None, "mask": None}

    predictions["box"] = [x1, y1, x2, y2]
    predictions["score"] = float(scores[best_idx])
    predictions["class_id"] = int(classes[best_idx])
    predictions["class_name"] = pred.names[predictions["class_id"]]
    cropped_image_path, cropped_mask_path = None, None
    features = {}

    if masks:
        mask_list = masks[best_idx].tolist()
        predictions["mask"] = mask_list

        mask = pred.masks.data[best_idx].cpu().numpy()

        # Resize just in case, YOLO may return masks at the model's inference shape
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST
            )  # cv2 uses format w,h

        mask_binary = (mask > 0.5).astype(np.uint8)

        # Crop to get only ROI tree
        cropped_image_path, cropped_mask_path = _crop_image(
            mask_binary=mask_binary,
            org_img=org_img,
            save_dir=save_dir,
            box=[x1, y1, x2, y2],
        )

        # Extract 2D features from the full mask and original image
        features = _extract_2d_features(
            mask=mask_binary, bbox=[x1, y1, x2, y2], image=org_img
        )
    else:
        predictions["mask"] = None

    result = {
        "result_path": pred.path,
        "tree_id": tree_id,
        "timestamp": time_stamp,
        "save_dir": str(save_dir),
        "predictions": predictions,
        "cropped_image_path": cropped_image_path,
        "cropped_mask_path": cropped_mask_path,
        "features_2d": features,
    }

    if save_json:
        json_path = os.path.join(save_dir, "result_2d.json")
        with open(json_path, "w") as f:
            json.dump(result, f)

    return result
