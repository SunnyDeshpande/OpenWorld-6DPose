import os
import argparse
import time
import torch
import trimesh
import objaverse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

class MeshRetrievalPipeline:
    def __init__(self, model_id="vikhyatk/moondream2", output_dir="./objects"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Vision-Language Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load Objaverse LVIS taxonomy
        print("Loading Objaverse LVIS annotations...")
        self.lvis_annotations = objaverse.load_lvis_annotations()

    def get_semantic_labels(self, image_path):
        """Extracts unique object labels from an image using VLM inference."""
        image = Image.open(image_path)
        prompt = "List all individual objects in this image as a comma separated list. Only include names."
        
        with torch.no_grad():
            encoded_image = self.model.encode_image(image)
            response = self.model.answer_question(encoded_image, prompt, self.tokenizer)
        
        labels = [obj.strip().lower() for obj in response.split(',')]
        # Add common workplace domain heuristics
        labels.extend(["computer", "monitor"])
        return list(set(labels))

    def export_meshes(self, labels, max_assets=5):
        """Matches labels to LVIS categories and exports simplified OBJ files."""
        for label in labels:
            # Find closest LVIS category match via substring matching
            matching_category = next(
                (cat for cat in self.lvis_annotations.keys() if label in cat or cat in label), 
                None
            )
            
            if not matching_category:
                continue

            uids = self.lvis_annotations[matching_category][:max_assets]
            objects = objaverse.load_objects(uids=uids)
            
            label_path = os.path.join(self.output_dir, label.replace(" ", "_"))
            os.makedirs(label_path, exist_ok=True)

            for i, (uid, glb_path) in enumerate(objects.items()):
                try:
                    # Load heterogeneous 3D formats and unify to mesh
                    mesh = trimesh.load(glb_path, force='mesh')
                    
                    export_name = f"{label.replace(' ', '_')}_{i}.obj"
                    dest_path = os.path.join(label_path, export_name)
                    
                    mesh.export(dest_path)
                    print(f"Exported: {dest_path}")
                except Exception as e:
                    print(f"Mesh conversion failed for {uid}: {e}")

def main():
    parser = argparse.ArgumentParser(description="VLM-based 3D Mesh Retrieval Pipeline")
    parser.add_argument("--image", type=str, default="../FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png")
    parser.add_argument("--out", type=str, default="./results", help="Target output directory")
    args = parser.parse_args()

    pipeline = MeshRetrievalPipeline(output_dir=args.out)
    
    start = time.perf_counter()
    labels = pipeline.get_semantic_labels(args.image)
    pipeline.export_meshes(labels)
    
    duration = time.perf_counter() - start
    print(f"\nPipeline completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
