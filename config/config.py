import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        # load .env variables
        load_dotenv()

        # Feature parameters
        self.DATASET_NAME = os.getenv("DATASET_NAME")

        # self.LOGGING = os.getenv("LOGGING") == "True"
        # Model parameters
        self.SEG_MODEL_CHECKPOINT = os.getenv("SEG_MODEL_CHECKPOINT")

        # YOLO parameters
        if (
            os.getenv("CLASSIFICATION_FPS") is not None
            and os.getenv("CLASSIFICATION_FPS") != ""
        ):
            self.CLASSIFICATION_FPS = int(os.getenv("CLASSIFICATION_FPS", "15"))
        if (
            os.getenv("CLASSIFICATION_THRESHOLD") is not None
            and os.getenv("CLASSIFICATION_THRESHOLD") != ""
        ):
            self.CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD"))
        if (
            os.getenv("MAX_NUMBER_OF_PREDICTIONS") is not None
            and os.getenv("CLASSIFICATION_FPS") != ""
        ):
            self.MAX_NUMBER_OF_PREDICTIONS = int(
                os.getenv("MAX_NUMBER_OF_PREDICTIONS", "50")
            )
        if os.getenv("MIN_DISTANCE") is not None and os.getenv("MIN_DISTANCE") != "":
            self.MIN_DISTANCE = int(os.getenv("MIN_DISTANCE", "500"))
        if (
            os.getenv("MIN_DETECTIONS") is not None
            and os.getenv("MIN_DETECTIONS") != ""
        ):
            self.MIN_DETECTIONS = int(os.getenv("MIN_DETECTIONS", "5"))
        self.FRAMES_SKIP_AFTER_DETECT = int(os.getenv("FRAMES_SKIP_AFTER_DETECT", "50"))
        self.IOU = float(os.getenv("IOU", "0.85"))
