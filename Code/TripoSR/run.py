import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import torch.nn.functional as F
import xatlas
import trimesh
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model.",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Chunk size for surface extraction. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes resolution. Default: 256"
)
parser.add_argument(
    "--threshold",
    default=25.0,
    type=float,
    help="ISO-surface threshold. Higher = thinner/sharper. Default: 25.0"
)
parser.add_argument(
    "--smooth-iterations",
    default=15,
    type=int,
    help="Smoothing iterations. Increase to remove bumps. Default: 15"
)
parser.add_argument(
    "--target-faces",
    default=15728,
    type=int,
    help="Target face count to match benchmark. Default: 15728"
)
parser.add_argument(
    "--rescale-factor",
    default=0.4,
    type=float,
    help="Scale factor to match benchmark coordinate range. Default: 0.4"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="Skip background removal.",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    help="Bake texture atlas instead of vertex colors",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution. Default: 2048"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Save a NeRF-rendered video. Default: false",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []
rembg_session = None if args.no_remove_bg else rembg.new_session()

for i, image_path in enumerate(args.image):
    raw_image = Image.open(image_path)
    if args.no_remove_bg:
        image = np.array(raw_image.convert("RGB"))
    else:
        image = remove_background(raw_image, rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        save_path = os.path.join(output_dir, str(i))
        os.makedirs(save_path, exist_ok=True)
        image.save(os.path.join(save_path, "input.png"))
    images.append(image)
timer.end("Processing images")

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        save_video(render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30)
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(
        scene_codes, 
        not args.bake_texture, 
        resolution=args.mc_resolution,
        threshold=args.threshold
    )
    mesh = meshes[0]

    # 1. Laplacian Smoothing: Clean up bumpiness
    if args.smooth_iterations > 0:
        logging.info(f"Smoothing: {args.smooth_iterations} iterations...")
        trimesh.smoothing.filter_laplacian(mesh, iterations=args.smooth_iterations)

    # 2. Quadric Decimation: Match benchmark face count
    if len(mesh.faces) > args.target_faces:
        logging.info(f"Decimating: {len(mesh.faces)} -> {args.target_faces} faces...")
        mesh = mesh.simplify_quadric_decimation(args.target_faces)
    
    # 3. Scale Adjustment: Match benchmark coordinate range
    if args.rescale_factor != 1.0:
        logging.info(f"Rescaling by factor {args.rescale_factor}...")
        mesh.apply_scale(args.rescale_factor)

    meshes[0] = mesh
    timer.end("Extracting mesh")

    out_folder = os.path.join(output_dir, str(i))
    out_mesh_name = f"mesh.{args.model_save_format}"
    out_mesh_path = os.path.join(out_folder, out_mesh_name)
    
    if args.bake_texture:
        out_texture_name = "texture.png"
        out_texture_path = os.path.join(out_folder, out_texture_name)
        out_mtl_name = f"{out_mesh_name}.mtl"
        out_mtl_path = os.path.join(out_folder, out_mtl_name)

        timer.start("Baking texture")
        original_query = model.renderer.query_triplane
        def patched_query(decoder, sc, pos):
            # Scale positions back for the query
            query_pos = pos.to(device) / args.rescale_factor
            res = original_query(decoder, sc.to(device), query_pos)
            if isinstance(res, dict):
                return {k: v.cpu() if torch.is_tensor(v) else v for k, v in res.items()}
            return res.cpu()
        
        model.renderer.query_triplane = patched_query
        
        try:
            bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        finally:
            model.renderer.query_triplane = original_query
        timer.end("Baking texture")

        timer.start("Exporting formatted OBJ/MTL")
        # Write MTL file
        mtl_content = f"newmtl material_0\nKa 1.000000 1.000000 1.000000\nKd 1.000000 1.000000 1.000000\nKs 0.000000 0.000000 0.000000\nd 1.0\nillum 1\nmap_Kd {out_texture_name}\n"
        with open(out_mtl_path, "w") as f:
            f.write(mtl_content)

        # Export OBJ with Normals and UVs
        xatlas.export(
            out_mesh_path, 
            meshes[0].vertices[bake_output["vmapping"]], 
            bake_output["indices"], 
            bake_output["uvs"], 
            meshes[0].vertex_normals[bake_output["vmapping"]]
        )
        
        # Inject mtllib reference into the OBJ file
        with open(out_mesh_path, "r") as f:
            lines = f.readlines()
        with open(out_mesh_path, "w") as f:
            f.write(f"mtllib {out_mtl_name}\nusemtl material_0\n")
            f.writelines(lines)

        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting formatted OBJ/MTL")
    else:
        timer.start("Exporting raw mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting raw mesh")