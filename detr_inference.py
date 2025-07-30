import torch
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.visualization import DetLocalVisualizer

# Ensure that the GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you are running this script on a GPU-enabled node.")

# Path to config and checkpoint files
config_file = 'detr_r50_8xb2-150e_coco.py'
checkpoint_file = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

# Initialize the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Add CLASSES attribute to the model
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

# Path to the input image
image_path = '../test.jpg'

# Run inference
result = inference_detector(model, image_path)

# Read the image using mmcv (BGR format)
image = mmcv.imread(image_path)

# Convert the image from BGR to RGB
image_rgb = mmcv.bgr2rgb(image)

# Visualize the results
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = {'classes': model.CLASSES}

# Draw predictions on the image
visualizer.add_datasample(
    'result',
    image_rgb,  # Use the RGB image for visualization
    data_sample=result,
    draw_gt=False,
    show=False,
    out_file='output.jpg'
)

print("Inference completed. Results saved to output.jpg")