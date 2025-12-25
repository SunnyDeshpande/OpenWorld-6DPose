import os
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

def segment_and_save(image_path, label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIX: Use the correct repository ID
    model_id = "facebook/sam3" 

    print(f"Loading {model_id}...")
    try:
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Check if you have accepted the license on HF and run 'huggingface-cli login'")
        return

    # 2. Load and Prepare Image
    if not os.path.exists(image_path):
        print(f"File {image_path} not found.")
        return
        
    image = Image.open(image_path).convert("RGB")
    
    # SAM 3 Text-to-Mask inference
    inputs = processor(images=image, text=label, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Post-process
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        target_sizes=[image.size[::-1]] # (H, W)
    )[0]

    if len(results["masks"]) == 0:
        print(f"No objects found for prompt: '{label}'")
        return

    # 4. Create Transparent Mask
    # Combine all instances of the prompted label
    combined_mask = torch.any(results["masks"], dim=0).cpu().numpy()
    
    img_np = np.array(image)
    rgba_image = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
    rgba_image[..., :3] = img_np
    rgba_image[..., 3] = (combined_mask * 255).astype(np.uint8)

    # 5. Save
    output_filename = f"{os.path.splitext(image_path)[0]}_{label}_mask.png"
    Image.fromarray(rgba_image).save(output_filename)
    print(f"Successfully saved mask to: {output_filename}")

if __name__ == "__main__":
    target_image = "/home/sunnynd2/mustard.png" # Make sure this exists in the same folder
    target_label = "mustard"
    segment_and_save(target_image, target_label)