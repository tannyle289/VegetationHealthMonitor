from services.segmentation import services as seg_service
from services.lidar import services as lidar_service


def phase1_2_single_shot(input_path: str, save_dir: str, time_stamp: str, tree_id: int):
    try:
        model1_type = seg_service.load_model()

        result_2d = seg_service.predict(
            input_path=input_path,
            save_dir=save_dir,
            tree_id=tree_id,
            time_stamp=time_stamp,
        )
    except Exception as e:
        raise Exception(e)

    cropped_image_path, cropped_mask_path = (
        result_2d["cropped_image_path"],
        result_2d["cropped_mask_path"],
    )

    try:
        model2_type = lidar_service.load_model()

        result_3d = lidar_service.process(
            image_cropped_path=cropped_image_path,
            mask_cropped_path=cropped_mask_path,
            save_dir=save_dir,
            tree_id=tree_id,
            time_stamp=time_stamp,
        )
    except Exception as e:
        raise Exception(e)

    print("------Phase 1------")
    print(f"Model used: {model1_type}")
    print(f"Tree predictions (bbox and masks): {result_2d['predictions']}")
    print(f"2d features extracted: {result_2d['features_2d']}")

    print("------Phase 2------")
    print(f"Model used: {model2_type}")
    print(f"3d features extracted: {result_3d['features_3d']}")


if __name__ == "__main__":
    _input_path = "your/test/img/path"
    _save_dir = ""  # Can be blank, it will automatically save to DATA_PATH/processed
    # check .env to modify DATA_PATH value
    _tree_id = 420
    _timestamp = "02112025"

    phase1_2_single_shot(
        input_path=_input_path,
        save_dir=_save_dir,
        time_stamp=_timestamp,
        tree_id=_tree_id,
    )
