from fastapi import APIRouter, HTTPException
from .services import load_model, start_training, infer_with_tracking, predict
from fastapi import BackgroundTasks


router = APIRouter()

model_state = {
    "model_type": "Unknown",
    "ready": False,
}


@router.get("/health")
def health():
    return {"status_code": 200, "status": "ok"}


@router.get("/ready")
def ready():
    return {"ready": model_state["ready"], "model_type": model_state["model_type"]}


@router.post("/model/load")
def load_model_endpoint():
    try:
        model_type = load_model()
        model_state["model_type"] = model_type
        model_state["ready"] = True
    except Exception as e:
        return {"status_code": 500, "message": str(e)}
    return {"status_code": 200, "model_type": model_state["model_type"]}


@router.post("/model/train")
def train_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training)
    return {"status": "training started"}  # TODO Add status code for this state


@router.post("/model/track")
def track_endpoint(input_path: str, save_dir: str = ""):
    try:
        results = infer_with_tracking(input_path=input_path, save_dir=save_dir)
    except FileNotFoundError and ModuleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status_code": 200, "results": results}


@router.post("/model/predict")
def predict_endpoint(
    input_path: str, save_dir: str = "", tree_id=0, time_stamp="", save_json=False
):
    try:
        results = predict(
            input_path=input_path,
            save_dir=save_dir,
            tree_id=tree_id,
            time_stamp=time_stamp,
            save_json=save_json,
        )
    except FileNotFoundError and ModuleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status_code": 200, "results": results}
