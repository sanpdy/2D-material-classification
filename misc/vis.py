import os
import sys
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def visualize_predictions_and_gt(
    image_path: str,
    detections: list,
    gt_annotations: list,
    save_path: str,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️  Cannot load image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Ground Truth")
    axes[1].imshow(img_rgb)
    axes[1].set_title("Predictions")
    for ax in axes:
        ax.axis('off')

    for ann in gt_annotations:
        x, y, w, h = ann['bbox']
        rect = Rectangle((x, y), w, h,
                         linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y-6, f"Layer {ann['category_id']}",
                     color='red', fontsize=10, weight='bold')

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-6, f"Layer {det['layer']}",
                     color='lime', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # python vis.py <IMG_DIR> <DET_JSON> <GT_JSON> [<OUTPUT_DIR>]

    if len(sys.argv) < 4:
        print("Usage: python vis.py <IMG_DIR> <DET_JSON> <GT_JSON> [<OUTPUT_DIR>]")
        sys.exit(1)

    IMG_DIR   = sys.argv[1]
    DET_JSON  = sys.argv[2]
    GT_JSON   = sys.argv[3]
    OUTPUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "vis"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DET_JSON) as f:
        det_results = json.load(f)

    gt_data = json.load(open(GT_JSON))
    img_id_map = {img['file_name']: img['id'] for img in gt_data['images']}
    anns_by_image = {}
    for ann in gt_data['annotations']:
        fname = next((fn for fn,iid in img_id_map.items() if iid == ann['image_id']), None)
        if fname:
            anns_by_image.setdefault(fname, []).append(ann)

    for fname, dets in det_results.items():
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        gt_anns = anns_by_image.get(fname, [])
        save_name = f"vis_{Path(fname).stem}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        visualize_predictions_and_gt(img_path, dets, gt_anns, save_path)
