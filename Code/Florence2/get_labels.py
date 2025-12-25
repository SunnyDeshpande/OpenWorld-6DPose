import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

def load_model(model_id: str, device: str):
    """
    Initializes Florence-2 model and processor with necessary configuration patches.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, 'forced_bos_token_id'):
        config.forced_bos_token_id = None

    dtype = torch.float16 if "cuda" in device else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        config=config, 
        torch_dtype=dtype, 
        trust_remote_code=True
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, dtype

@torch.no_grad()
def run_inference(model, processor, device, dtype, image, prompt="<OD>"):
    """
    Performs a forward pass and parses the vision-language output.
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(image.width, image.height)
    )

def visualize_detections(image, detections, task_prompt):
    """
    Overlays bounding boxes and labels onto the source image.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for task_key, data in detections.items():
        if 'bboxes' not in data:
            continue
            
        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1, label, color='white', fontsize=8, 
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 2})
    
    ax.set_axis_off()
    ax.set_title(f"Florence-2 Result: {task_prompt}")
    plt.tight_layout()
    plt.show()

def main(img_path: str):
    MODEL_ID = "microsoft/Florence-2-large"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model, processor, dtype = load_model(MODEL_ID, DEVICE)
    
    try:
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {img_path}")
        return

    print(f"Processing inference on {DEVICE}...")
    results = run_inference(model, processor, DEVICE, dtype, image)
    
    # Extract unique labels for downstream tasks (e.g., SAM 3 integration)
    for task, content in results.items():
        if 'labels' in content:
            unique_labels = list(set(content['labels']))
            print(f"Detected Objects: {unique_labels}")

    visualize_detections(image, results, "<OD>")

if __name__ == "__main__":
    # Update this path for your local environment
    target_image = "/path/to/your/image.png"
    main(target_image)
