from fastapi import FastAPI
from services.segmentation import router as seg_router
from services.lidar import router as lidar_router

app = FastAPI(title="My Tree Monitoring System", version="1.0.0")

app.include_router(seg_router, prefix="/segmentation", tags=["Segmentation"])
app.include_router(lidar_router, prefix="/lidar", tags=["LiDAR"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
