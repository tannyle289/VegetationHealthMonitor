from config.config import Config
from os.path import join as pjoin
from ultralytics import YOLO

seg_model = None
_config = Config()


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


def predict(input_path: str, save_dir: str = None):
    if seg_model is None:
        raise ModuleNotFoundError("Model not loaded")

    if not save_dir:
        save_dir = pjoin(_config.SEG_FOLDER_PATH, "runs", "predictions")

    results = seg_model.predict(source=input_path, save=True, save_dir=save_dir)

    predictions = []
    for frame_res in results:
        frame_data = {
            "boxes": [],
            "scores": [],
            "classes": [],
            "masks": [],
        }
        # Boxes, scores, classes
        boxes = frame_res.boxes.xyxy.cpu().numpy()
        scores = frame_res.boxes.conf.cpu().numpy()
        classes = frame_res.boxes.cls.cpu().numpy()

        # Masks
        if hasattr(frame_res, "masks") and frame_res.masks is not None:
            # Get mask polygons
            mask_polygons = frame_res.masks.xy
            for p in mask_polygons:
                poly = p.tolist()
                frame_data["masks"].append(poly)
        else:
            frame_data["masks"] = []

        # Fill in frame_data
        frame_data["boxes"] = boxes.tolist()
        frame_data["scores"] = scores.tolist()
        frame_data["classes"] = classes.tolist()

        predictions.append(frame_data)

    return {
        "result_path": results[0].path,
        "save_dir": results[0].save_dir,
        "predictions": predictions,
    }
