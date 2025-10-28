from fastapi import FastAPI
from services.segmentation_service.routers import router as seg_router

app = FastAPI(title="My Tree Monitoring System", version="1.0.0")

app.include_router(seg_router, prefix="/segmentation", tags=["Segmentation"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
