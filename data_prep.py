# data_prep_bbox.py: Prepare crops using COCO bboxes only (parsing category names correctly)
import json
import re
import cv2
import numpy as np
from pathlib import Path

# === CONFIG ===
RAW_ROOT = Path("/home/sankalp/flake_classification/GMMDetectorDatasets")
OUT_ROOT = Path("/home/sankalp/flake_classification/GMMClassifier_bbox")
MATERIALS = ["Graphene", "WSe2"]
SPLITS    = ["train", "test"]

# loop splits & materials
for split in SPLITS:
    for mat in MATERIALS:
        # load COCO JSON (_300 suffix)
        ann_path = RAW_ROOT / mat / "annotations" / f"{split}.json"
        with open(ann_path) as f:
            coco = json.load(f)

        # build category ID -> layer number mapping from JSON categories
        # category names like '1-Layer', '2-Layer', etc.
        cat_map = {}
        for cat in coco.get("categories", []):
            name = cat.get("name", "")
            m = re.match(r"(\d+)", name)
            if m:
                layer_num = int(m.group(1))
            else:
                layer_num = cat.get("id")
            cat_map[cat.get("id")] = layer_num

        # build image_id -> list of annotations
        by_img = {}
        for a in coco.get("annotations", []):
            by_img.setdefault(a.get("image_id"), []).append(a)

        # iterate over images
        for img_info in coco.get("images", []):
            img_id = img_info.get("id")
            fname  = img_info.get("file_name")
            img_path = RAW_ROOT / mat / f"{split}_images" / fname

            # load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️  Cannot load image: {img_path}")
                continue

            # crop by bbox for each annotation
            for i, a in enumerate(by_img.get(img_id, [])):
                cid = a.get("category_id")
                layer = cat_map.get(cid, cid)  # map category ID to actual layer
                x, y, w, h = map(int, a.get("bbox", [0,0,0,0]))

                # simple bbox crop
                crop = img[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                # prepare output directory and filename
                out_dir = OUT_ROOT / split / str(layer)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{mat}_{img_id:04d}_ann{i}_L{layer}.png"

                # save crop
                cv2.imwrite(str(out_dir / out_name), crop)

print("Done! Your bbox crops are in:\n", OUT_ROOT)
