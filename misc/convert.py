from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/home/sankalp/flake_classification/GMMDetectorDatasets/Graphene/annotations",
    save_dir="/home/sankalp/flake_classification/YOLO_ready/Graphene",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False
)

convert_coco(
    labels_dir="/home/sankalp/flake_classification/GMMDetectorDatasets/WSe2/annotations",
    save_dir="/home/sankalp/flake_classification/YOLO_ready/WSe2",
    use_segments=True,
    use_keypoints=False,
    cls91to80=False
)
