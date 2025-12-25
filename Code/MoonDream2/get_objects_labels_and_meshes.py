import os
import argparse
import time
import torch
import trimesh
import objaverse.xl as oxl
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import fast_simplification as fs

class MeshExtractionPipeline:
    def __init__(self, model_id="vikhyatk/moondream2", output_dir="./results_xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Initializing {model_id} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load XL metadata (cached after first run)
        self.annotations = oxl.get_annotations()

    def get_labels(self, image_path):
        """Extract semantic labels from image using VLM."""
        image = Image.open(image_path)
        prompt = "List all individual objects in this image, comma separated. Only names."
        
        with torch.no_grad():
            enc_image = self.model.encode_image(image)
            response = self.model.answer_question(enc_image, prompt, self.tokenizer)
        
        labels = [obj.strip().lower() for obj in response.split(',')]
        # Heuristics for common lab environment clutter
        labels.extend(["computer", "monitor"])
        return list(set(labels))

    def decimate_mesh(self, mesh, target_v=100_000):
        """Simplifies mesh geometry to a manageable vertex budget."""
        if len(mesh.vertices) <= target_v:
            return mesh
        
        ratio = target_v / len(mesh.vertices)
        v, f = fs.simplify(mesh.vertices, mesh.faces, ratio)
        return trimesh.Trimesh(vertices=v, faces=f, process=True)

    def process_labels(self, labels, max_per_label=5):
        """Queries Objaverse-XL and exports optimized OBJ files."""
        for label in labels:
            print(f"\nQuerying: {label}")
            matches = self.annotations[
                self.annotations['metadata'].str.contains(label, case=False, na=False)
            ].head(max_per_label)

            if matches.empty:
                continue

            downloaded = oxl.download_objects(matches)
            label_dir = os.path.join(self.output_dir, label.replace(" ", "_"))
            os.makedirs(label_dir, exist_ok=True)

            for i, (uid, path) in enumerate(downloaded.items()):
                try:
                    # Unify heterogeneous formats (GLB/FBX/OBJ) via trimesh
                    mesh = trimesh.load(path, force='mesh')
                    mesh = self.decimate_mesh(mesh)
                    
                    out_path = os.path.join(label_dir, f"{label}_{i}.obj")
                    mesh.export(out_path)
                    print(f"  [+] Exported: {out_path} ({len(mesh.vertices)} verts)")
                except Exception as e:
                    print(f"  [-] Failed {uid}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png")
    parser.add_argument("--out", type=str, default="./objects_xl", help="Output directory")
    args = parser.parse_args()

    pipeline = MeshExtractionPipeline(output_dir=args.out)
    
    start = time.perf_counter()
    labels = pipeline.get_labels(args.image)
    pipeline.process_labels(labels)
    
    print(f"\nPipeline runtime: {time.perf_counter() - start:.2f}s")

if __name__ == "__main__":
    main()
