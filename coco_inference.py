from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval 
from tqdm import tqdm
import os
import json
import cv2
import numpy as np

# ----------------------------
# 1. Set paths
# ----------------------------
config_file = 'detr_r50_8xb2-150e_coco.py'
checkpoint_file = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
data_root = '/home/woody/iwi5/iwi5237h/coco2017/'
ann_file = data_root + 'annotations/instances_val2017.json'
img_dir = data_root + 'val'
save_dir = './visualized_results'
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 2. Load model
# ----------------------------
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ----------------------------
# 3. Load COCO annotations
# ----------------------------
coco = COCO(ann_file)
img_ids = coco.getImgIds()
print(f"Total images in validation set: {len(img_ids)}\n")

# ----------------------------
# 4. Inference + draw boxes
# ----------------------------
results = []
total_preds = 0

for img_id in tqdm(img_ids):  # Change the slice to increase images
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    img = cv2.imread(img_path)

    result = inference_detector(model, img_path)
    pred_instances = result.pred_instances

    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    print(f"Image ID: {img_id} - {len(bboxes)} predictions")
    total_preds += len(bboxes)

    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x1 > x2 or y1 >y2:
            continue
        if score < 0.5 :
            continue
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]

        # category_id = coco.getCatIds()[label]  # maps class index to COCO category_id
        # Map model label to COCO category ID correctly
        coco_cat_ids = coco.getCatIds()
        class_name_to_cat_id = {cat['name']: cat['id'] for cat in coco.loadCats(coco_cat_ids)}

        category_name = model.CLASSES[label]
        category_id = class_name_to_cat_id[category_name]

        #if score > 0.85:
        results.append({
            "image_id": int(img_id),
            #"category_id": int(label),
            "category_id": int(category_id),

            "bbox": [round(float(x), 2) for x in coco_bbox],
            "score": float(score)
        })

        # Draw the box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label_text = f"{model.CLASSES[label]}: {score:.2f}"
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    # Save visualized image
    save_path = os.path.join(save_dir, f"{img_info['file_name']}")
    cv2.imwrite(save_path, img)

print(f"\n✅ Total predictions across all images: {total_preds}")

# ----------------------------
# 5. Save results and evaluate
# ----------------------------
with open('results.json', 'w') as f:
    json.dump(results, f)

coco_dt = coco.loadRes('results.json')
coco_eval = COCOeval(coco, coco_dt, iouType='bbox')





# Optional: customize evaluation settings
coco_eval.params.imgIds = img_ids  # or full list for full eval
coco_eval.params.catIds = coco.getCatIds()
coco_eval.params.useCats = 1  # Example: A predicted "cat" won’t match a GT "dog", even if boxes overlap perfectly.
# Use 10 IoU thresholds from 0.5 to 0.95 (step=0.05).   #The AP is averaged across these thresholds for robustness.
coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
# coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
# coco_eval.params.areaRngLbl = ['all']   # Evaluates predictions of all object sizes (no filtering by area).
coco_eval.params.maxDets = [1, 10, 100] # Compute metrics for top-1, top-10, and top-100 detections per image.

# Run evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
