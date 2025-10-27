from ultralytics import YOLO
import os

print(os.getcwd())
print(os.path.exists("../datasets/urban-street_tree_YOLO/data_config.yaml"))


model = YOLO("../models/yolo11n-seg.pt")

# Train the model
model.train(
    data="../datasets/urban-street_tree_YOLO/data_config.yaml",
    epochs=400,
    batch=4,
    workers=2,
    project="../models/yolo11_seg_tree",
    name="yolo11n_seg_tree_training",
)
