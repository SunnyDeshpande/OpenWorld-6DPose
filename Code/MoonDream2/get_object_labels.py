import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

def load_vision_model(model_id, device):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(config, 'forced_bos_token_id'):
        config.forced_bos_token_id = None

    dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor, dtype

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, dtype = load_vision_model(args.model, device)
    
    image = Image.open(args.image).convert("RGB")
    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device, dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=5
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task="<OD>", image_size=image.size)
    
    labels = list(set(parsed['<OD>']['labels']))
    with open("labels.txt", "w") as f:
        f.write("\n".join(labels))
    
    print(f"Detected: {labels}")

if __name__ == "__main__":
    main()
