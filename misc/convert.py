#!/usr/bin/env python3
import json
from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split
from ultralytics.data.converter import convert_coco

SOURCES = [
    {
        "json_train": "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/Graphene/annotations/train.json",
        "json_test":  "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/Graphene/annotations/test.json",
        "img_train":  "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/Graphene/train_images",
        "img_test":   "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/Graphene/test_images"
    },
    {
        "json_train": "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/WSe2/annotations/train.json",
        "json_test":  "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/WSe2/annotations/test.json",
        "img_train":  "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/WSe2/train_images",
        "img_test":   "/home/sankalp/flake_classification/datasets/GMMDetectorDatasets/WSe2/test_images"
    }
]
OUTPUT_DIR = Path("/home/sankalp/flake_classification/datasets/YOLO_ready/AllFlakes")
TRAIN_RATIO = 0.8
SEED = 42

def load_and_validate_coco(json_path):
    """Load and validate COCO JSON file"""
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
        
        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco:
                raise ValueError(f"Missing required field '{field}' in {json_path}")
        
        return coco
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        raise

def merge_categories(all_categories):
    """Merge and deduplicate categories from multiple datasets"""
    merged_cats = []
    seen_names = set()
    
    for categories in all_categories:
        for cat in categories:
            if cat['name'] not in seen_names:
                # Reassign IDs to ensure they're sequential starting from 0
                new_cat = cat.copy()
                new_cat['id'] = len(merged_cats)
                merged_cats.append(new_cat)
                seen_names.add(cat['name'])
    
    return merged_cats

print("Loading and merging COCO datasets...")
merged = {"info": None, "licenses": None, "categories": [], "images": [], "annotations": []}
all_categories = []
img_id_counter = 1
ann_id_counter = 1

for src_idx, src in enumerate(SOURCES):
    print(f"Processing source {src_idx + 1}/{len(SOURCES)}")
    
    for split in ("train", "test"):
        json_path = src[f"json_{split}"]
        img_dir = src[f"img_{split}"]
        
        if not Path(json_path).exists():
            print(f"Warning: {json_path} does not exist, skipping...")
            continue
        if not Path(img_dir).exists():
            print(f"Warning: {img_dir} does not exist, skipping...")
            continue
            
        coco = load_and_validate_coco(json_path)
        
        if src_idx == 0 and split == "train":
            all_categories.append(coco["categories"])
            merged["info"] = coco.get("info", None)
            merged["licenses"] = coco.get("licenses", None)
        
        old_to_new_img_id = {}
        for img in coco["images"]:
            old_img_id = img["id"]
            new_img = img.copy()
            new_img["id"] = img_id_counter
            new_img["file_name"] = Path(img["file_name"]).name
            
            old_to_new_img_id[old_img_id] = img_id_counter
            merged["images"].append(new_img)
            img_id_counter += 1
        
        for ann in coco["annotations"]:
            if ann["image_id"] in old_to_new_img_id:
                new_ann = ann.copy()
                new_ann["id"] = ann_id_counter
                new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
                merged["annotations"].append(new_ann)
                ann_id_counter += 1

merged["categories"] = merge_categories(all_categories)
print(f"Merged dataset: {len(merged['images'])} images, {len(merged['annotations'])} annotations, {len(merged['categories'])} categories")

print(f"Splitting dataset {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%} train/val...")
all_imgs = merged["images"]
train_imgs, val_imgs = train_test_split(
    all_imgs, 
    train_size=TRAIN_RATIO, 
    random_state=SEED, 
    shuffle=True
)

train_ids = {im["id"] for im in train_imgs}
train_anns = [a for a in merged["annotations"] if a["image_id"] in train_ids]
val_anns = [a for a in merged["annotations"] if a["image_id"] not in train_ids]

print(f"Train split: {len(train_imgs)} images, {len(train_anns)} annotations")
print(f"Val split: {len(val_imgs)} images, {len(val_anns)} annotations")

common = {k: merged[k] for k in ("info", "licenses", "categories") if merged[k] is not None}

print("Creating output directories...")
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
# Use temp directory for intermediate COCO files
import tempfile
temp_dir = Path(tempfile.mkdtemp())
coco_splits = temp_dir / "coco_splits"
coco_splits.mkdir(exist_ok=True)

print("Saving COCO split files...")
train_json = coco_splits / "instances_train2017.json"
val_json = coco_splits / "instances_val2017.json"

train_coco = {**common, "images": train_imgs, "annotations": train_anns}
val_coco = {**common, "images": val_imgs, "annotations": val_anns}

with open(train_json, 'w') as f:
    json.dump(train_coco, f, indent=2)
with open(val_json, 'w') as f:
    json.dump(val_coco, f, indent=2)

print(f"Saved: {train_json}")
print(f"Saved: {val_json}")

print("Building image lookup and copying images...")
# Build lookup for image files
img_lookup = {}
for src in SOURCES:
    for key in ("img_train", "img_test"):
        img_dir = Path(src[key])
        if img_dir.exists():
            for p in img_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                    img_lookup[p.name] = p

print(f"Found {len(img_lookup)} images in source directories")

missing_images = []
for im in train_imgs:
    fn = im["file_name"]
    if fn in img_lookup:
        copy2(img_lookup[fn], OUTPUT_DIR / "images/train" / fn)
    else:
        missing_images.append(fn)

for im in val_imgs:
    fn = im["file_name"]
    if fn in img_lookup:
        copy2(img_lookup[fn], OUTPUT_DIR / "images/val" / fn)
    else:
        missing_images.append(fn)

if missing_images:
    print(f"Warning: {len(missing_images)} images not found in source directories:")
    for img in missing_images[:10]:  # Show first 10
        print(f"  - {img}")
    if len(missing_images) > 10:
        print(f"  ... and {len(missing_images) - 10} more")

print("Converting COCO to YOLO format...")
try:
    temp_output = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_temp"
    
    convert_coco(
        labels_dir=str(coco_splits),
        save_dir=str(temp_output),
        use_segments=True,
        use_keypoints=False,
        cls91to80=False
    )
    
    import shutil
    if (temp_output / "labels").exists():
        if (OUTPUT_DIR / "labels").exists():
            shutil.rmtree(OUTPUT_DIR / "labels")
        # Move the generated labels
        shutil.move(str(temp_output / "labels"), str(OUTPUT_DIR / "labels"))
    
    if temp_output.exists():
        shutil.rmtree(temp_output)
    
    shutil.rmtree(temp_dir)
    
    print("✅ COCO to YOLO conversion completed successfully!")
    print("🧹 Cleaned up temporary files")
    
except Exception as e:
    print(f"❌ Error during COCO to YOLO conversion: {e}")
    print("You may need to check the ultralytics version or manually convert the annotations")

yaml_content = f"""# YOLO dataset configuration
path: {OUTPUT_DIR}
train: images/train
val: images/val

# Classes
nc: {len(merged['categories'])}
names:
"""
for cat in merged['categories']:
    yaml_content += f"  {cat['id']}: {cat['name']}\n"

yaml_path = OUTPUT_DIR / "dataset.yaml"
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n🎉 Done! YOLO dataset structure:")
print(f"📁 {OUTPUT_DIR}")
print(f"├── images/train/     ({len(train_imgs)} images)")
print(f"├── images/val/       ({len(val_imgs)} images)")
print(f"├── labels/train/     ({len(train_imgs)} label files)")
print(f"├── labels/val/       ({len(val_imgs)} label files)")
print(f"└── dataset.yaml      (YOLO config)")
print(f"\n📊 Dataset summary:")
print(f"   Total images: {len(merged['images'])}")
print(f"   Total annotations: {len(merged['annotations'])}")
print(f"   Categories: {len(merged['categories'])}")
print(f"   Train/Val split: {len(train_imgs)}/{len(val_imgs)} ({TRAIN_RATIO:.1%}/{1-TRAIN_RATIO:.1%})")