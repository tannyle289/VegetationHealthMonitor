from fastapi import APIRouter, HTTPException
from .models import load_model, start_training, infer_with_tracking, predict
from fastapi import BackgroundTasks


router = APIRouter()

model_state = {
    "model_version": "best",
    "ready": False,
}


@router.get("/health")
def health():
    return {"status_code": 200, "status": "ok"}


@router.get("/ready")
def ready():
    return {"ready": model_state["ready"]}


@router.post("/model/load")
def load_model_route():
    try:
        version = load_model()
        model_state["model_version"] = version
        model_state["ready"] = True
    except Exception as e:
        return {"status_code": 500, "message": str(e)}
    return {"status_code": 200, "model_version": model_state["model_version"]}


@router.post("/model/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_training)
    return {"status": "training started"}  # TODO Add status code for this state


@router.post("/model/track")
def track_route(input_path: str, save_dir: str = ""):
    try:
        results = infer_with_tracking(input_path=input_path, save_dir=save_dir)
    except FileNotFoundError and ModuleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status_code": 200, "results": results}


@router.post("/model/predict")
def predict_route(input_path: str, save_dir: str = ""):
    try:
        results = predict(input_path=input_path, save_dir=save_dir)
    except FileNotFoundError and ModuleNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status_code": 200, "results": results}
