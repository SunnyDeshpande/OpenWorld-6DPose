from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import os

# Initialize YOLO for object detection
yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

# Initialize SAM for segmentation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fix: Use absolute path to SAM checkpoint
sam_checkpoint = os.path.expanduser("../Wonder3D/sam_pt/sam_vit_h_4b8939.pth")
# Or use: sam_checkpoint = "/home/ogawa3/Wonder3D/sam_pt/sam_vit_h_4b8939.pth"

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

def detect_and_segment_objects(image_path, output_folder):
    """Detect objects with YOLO and segment with SAM"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    results = yolo_model(image, conf=0.9)
    results[0].show()

    # Set image for SAM
    mask_predictor.set_image(image_rgb)
    
    object_count = 0
    detected_objects = []
    
    # Process each detection
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_label = yolo_model.names[class_id]
            
            # Get SAM mask using bounding box
            input_box = np.array([[x1, y1, x2, y2]])
            masks, _, _ = mask_predictor.predict(box=input_box, multimask_output=False)
            mask = masks[0]
            
            # Extract object with transparent background
            segmented_obj = image_rgb.copy()
            segmented_obj[~mask] = [255, 255, 255]  # White background
            
            # Crop to bounding box
            cropped = segmented_obj[y1:y2, x1:x2]
            
            # Save
            output_path = os.path.join(output_folder, f"{class_label}_{object_count:03d}.png")
            cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            
            detected_objects.append({
                'class': class_label,
                'bbox': [x1, y1, x2, y2],
                'path': output_path
            })
            object_count += 1
    
    return detected_objects

# Process all frames
frame_folder = "./frames"
output_objects = "./objects"

for frame_file in sorted(os.listdir(frame_folder)):
    frame_path = os.path.join(frame_folder, frame_file)
    objects = detect_and_segment_objects(frame_path, output_objects)
    print(f"Processed {frame_file}: Found {len(objects)} objects")
