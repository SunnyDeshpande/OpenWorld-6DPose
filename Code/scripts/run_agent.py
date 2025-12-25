#!/usr/bin/env python3
import trimesh
from estimater import FoundationPose
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from lang_sam import LangSAM
import subprocess
import shutil
import sys
import gc
import argparse
from skimage import morphology
from scipy.spatial import cKDTree
import warnings
import glob
warnings.filterwarnings('ignore')

# Try to import open3d for advanced reconstruction
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not installed. Some mesh refinement features disabled.\n")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTHONHASHSEED"] = "42"


# =============================================================================
# RGB-D SCENES DATASET V2 CONFIGURATION
# =============================================================================

# Kinect camera intrinsics (standard for RGB-D Scenes Dataset)
# These are the default Kinect v1 parameters used in most RGB-D datasets
KINECT_INTRINSICS = {
    'fx': 570.3,      # Focal length x
    'fy': 570.3,      # Focal length y  
    'cx': 320.0,      # Principal point x
    'cy': 240.0,      # Principal point y
    'width': 640,
    'height': 480,
    'depth_scale': 1000.0,  # Depth is in millimeters, convert to meters
}

# Alternative intrinsics (some datasets use these)
KINECT_INTRINSICS_ALT = {
    'fx': 585.0,
    'fy': 585.0,
    'cx': 320.0,
    'cy': 240.0,
    'width': 640,
    'height': 480,
    'depth_scale': 1000.0,
}

# Dataset paths
DATASET_BASE = "/home/anshb3/FoundationPose/rgbd-scenes-v2_imgs/rgbd-scenes-v2/imgs"
INSTANT_MESH_ROOT = "/home/anshb3/InstantMesh"


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description="Zero-Shot 6D Pose Estimation for RGB-D Scenes Dataset v2",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python run_agent_rgbd.py "coffee mug" --scene 1 --frame 100
    python run_agent_rgbd.py "cereal box" --scene 5 --frame 200 --ensemble 5
    python run_agent_rgbd.py "bowl" --rgb color.png --depth depth.png
    python run_agent_rgbd.py "soda can" --scene 3 --frame 50 --symmetry y
    """
)

# Object prompt
parser.add_argument("prompt", type=str, help="Object to detect (e.g., 'coffee mug', 'cereal box')")

# Data source options
parser.add_argument("--scene", type=int, default=None, help="Scene number (1-14)")
parser.add_argument("--frame", type=int, default=None, help="Frame number")
parser.add_argument("--rgb", type=str, default=None, help="Direct path to RGB image")
parser.add_argument("--depth", type=str, default=None, help="Direct path to depth image")

# Processing options
parser.add_argument("--ensemble", type=int, default=3, help="Number of mesh generations (default: 3)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--symmetry", type=str, default="none", choices=["x", "y", "z", "none"],
                    help="Symmetry axis for refinement (default: none)")
parser.add_argument("--enhance-level", type=int, default=2, choices=[1, 2, 3],
                    help="Image enhancement level (default: 2)")

# Advanced options
parser.add_argument("--depth-scale", type=float, default=1000.0,
                    help="Depth scale factor (default: 1000 for mm to m)")
parser.add_argument("--max-depth", type=float, default=10.0,
                    help="Maximum valid depth in meters (default: 10.0)")
parser.add_argument("--use-alt-intrinsics", action="store_true",
                    help="Use alternative Kinect intrinsics (fx=fy=585)")
parser.add_argument("--skip-ensemble", action="store_true", help="Single mesh generation")
parser.add_argument("--skip-poisson", action="store_true", help="Skip Poisson reconstruction")
parser.add_argument("--keep-intermediates", action="store_true", help="Keep intermediate files")
parser.add_argument("--output-dir", type=str, default="output_rgbd", help="Output directory")

args = parser.parse_args()


# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(WORK_DIR, args.output_dir)
MESH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "meshes")

# Select intrinsics
INTRINSICS = KINECT_INTRINSICS_ALT if args.use_alt_intrinsics else KINECT_INTRINSICS
INTRINSICS['depth_scale'] = args.depth_scale

N_ENSEMBLE = 1 if args.skip_ensemble else args.ensemble
SYMMETRY_AXIS = None if args.symmetry == "none" else args.symmetry


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_gpu_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_camera_matrix(intrinsics, img_width=None, img_height=None):
    """
    Create camera intrinsic matrix, scaling if image size differs from default.
    """
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Scale if image dimensions differ
    if img_width and img_height:
        scale_x = img_width / intrinsics['width']
        scale_y = img_height / intrinsics['height']
        
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            print(f"  📷 Scaling intrinsics: {scale_x:.2f}x, {scale_y:.2f}x")
            K[0, 0] *= scale_x  # fx
            K[1, 1] *= scale_y  # fy
            K[0, 2] *= scale_x  # cx
            K[1, 2] *= scale_y  # cy
    
    return K


# =============================================================================
# DATA LOADING
# =============================================================================

def find_scene_files(scene_num, frame_num):
    """Find RGB and depth files for a given scene and frame."""
    scene_dir = os.path.join(DATASET_BASE, f"scene_{scene_num:02d}")
    
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    rgb_pattern = os.path.join(scene_dir, f"{frame_num:05d}-color.png")
    depth_pattern = os.path.join(scene_dir, f"{frame_num:05d}-depth.png")
    
    if not os.path.exists(rgb_pattern):
        # Try to find closest frame
        available = glob.glob(os.path.join(scene_dir, "*-color.png"))
        available_nums = sorted([int(os.path.basename(f).split('-')[0]) for f in available])
        
        if not available_nums:
            raise FileNotFoundError(f"No frames found in {scene_dir}")
        
        closest = min(available_nums, key=lambda x: abs(x - frame_num))
        print(f"  ⚠️ Frame {frame_num} not found, using closest: {closest}")
        rgb_pattern = os.path.join(scene_dir, f"{closest:05d}-color.png")
        depth_pattern = os.path.join(scene_dir, f"{closest:05d}-depth.png")
    
    return rgb_pattern, depth_pattern


def load_depth_image(depth_path, depth_scale=1000.0, max_depth=10.0):
    """
    Load depth image and convert to meters.
    
    RGB-D Scenes Dataset uses 16-bit PNG depth images in millimeters.
    """
    # Load as 16-bit
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_raw is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    # Convert to float meters
    depth = depth_raw.astype(np.float32) / depth_scale
    
    # Handle invalid depth values
    depth[depth_raw == 0] = 0  # Missing depth
    depth[depth > max_depth] = 0  # Too far
    depth[np.isnan(depth)] = 0
    depth[np.isinf(depth)] = 0
    
    return depth


def load_rgb_image(rgb_path):
    """Load RGB image."""
    img = cv2.imread(rgb_path)
    if img is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def prepare_instantmesh_input(image_pil, mask, output_path, resolution=512):
    """Prepare image for InstantMesh with optimal preprocessing."""
    image_np = np.array(image_pil)
    mask_bool = mask > 0 if mask.dtype != bool else mask
    
    # Clean mask
    mask_cleaned = morphology.remove_small_objects(mask_bool, min_size=300)
    mask_cleaned = morphology.remove_small_holes(mask_cleaned, area_threshold=300)
    
    # Find bounding box
    rows = np.any(mask_cleaned, axis=1)
    cols = np.any(mask_cleaned, axis=0)
    
    if not rows.any() or not cols.any():
        print("  ⚠️ Empty mask, using full image")
        cropped = image_np
        mask_crop = mask_cleaned
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding
        pad = 10
        h, w = image_np.shape[:2]
        rmin, rmax = max(0, rmin - pad), min(h, rmax + pad + 1)
        cmin, cmax = max(0, cmin - pad), min(w, cmax + pad + 1)
        
        cropped = image_np[rmin:rmax, cmin:cmax]
        mask_crop = mask_cleaned[rmin:rmax, cmin:cmax]
    
    # Dilate mask for soft edges
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask_crop.astype(np.uint8), kernel, iterations=2)
    mask_soft = cv2.GaussianBlur(mask_dilated.astype(float), (7, 7), 0)
    mask_soft = np.clip(mask_soft, 0, 1)
    
    # Create square canvas
    h_crop, w_crop = cropped.shape[:2]
    canvas_size = int(max(h_crop, w_crop) * 1.35)
    
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    
    y_offset = (canvas_size - h_crop) // 2
    x_offset = (canvas_size - w_crop) // 2
    
    for c in range(3):
        canvas[y_offset:y_offset+h_crop, x_offset:x_offset+w_crop, c] = (
            cropped[:, :, c] * mask_soft + 255 * (1 - mask_soft)
        ).astype(np.uint8)
    
    # Resize and enhance
    result = Image.fromarray(canvas)
    result = result.resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    enhancer = ImageEnhance.Sharpness(result)
    result = enhancer.enhance(1.3)
    
    result.save(output_path, quality=100)
    return result


# =============================================================================
# MESH GENERATION AND REFINEMENT
# =============================================================================

def run_instantmesh_with_seed(input_image, output_mesh, seed=42):
    """Run InstantMesh with deterministic seeding."""
    
    seed_script = f"""
import sys
import random
random.seed({seed})
import numpy as np
np.random.seed({seed})
import torch
torch.manual_seed({seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed({seed})
    torch.cuda.manual_seed_all({seed})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
exec(open('run_headless.py').read())
"""
    
    seed_script_path = os.path.join(INSTANT_MESH_ROOT, f"_run_seeded_{seed}.py")
    
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = INSTANT_MESH_ROOT + os.pathsep + my_env.get("PYTHONPATH", "")
    my_env["PYTHONHASHSEED"] = str(seed)
    
    try:
        with open(seed_script_path, 'w') as f:
            f.write(seed_script)
        
        cmd = [sys.executable, seed_script_path, input_image, output_mesh]
        subprocess.check_call(cmd, cwd=INSTANT_MESH_ROOT, env=my_env)
        
    finally:
        if os.path.exists(seed_script_path):
            os.remove(seed_script_path)


def generate_mesh_ensemble(input_image, output_dir, n_runs=3, base_seed=42):
    """Generate multiple meshes and return paths."""
    mesh_paths = []
    
    for i in range(n_runs):
        seed = base_seed + i * 1000
        print(f"  Generation {i+1}/{n_runs} (seed={seed})...")
        
        mesh_path = os.path.join(output_dir, f"mesh_ensemble_{i}.obj")
        
        try:
            run_instantmesh_with_seed(input_image, mesh_path, seed=seed)
            if os.path.exists(mesh_path):
                mesh_paths.append(mesh_path)
                print(f"   Generated: {os.path.basename(mesh_path)}")
        except Exception as e:
            print(f"     Error: {e}")
        
        clear_gpu_memory()
    
    return mesh_paths


def select_best_mesh(mesh_paths):
    """Select most consistent mesh from ensemble."""
    print(f"Selecting best mesh from {len(mesh_paths)} candidates...")
    
    meshes = []
    volumes = []
    
    for path in mesh_paths:
        try:
            mesh = trimesh.load(path)
            meshes.append(mesh)
            
            if mesh.is_watertight:
                vol = abs(mesh.volume)
            else:
                try:
                    vol = abs(mesh.convex_hull.volume)
                except:
                    vol = len(mesh.vertices)
            
            volumes.append(vol)
            print(f"     - {os.path.basename(path)}: vol={vol:.4f}, verts={len(mesh.vertices)}")
            
        except Exception as e:
            print(f"     - {os.path.basename(path)}: ❌ {e}")
    
    if not meshes:
        raise RuntimeError("No valid meshes")
    
    median_vol = np.median(volumes)
    best_idx = np.argmin([abs(v - median_vol) for v in volumes])
    
    print(f"Selected mesh {best_idx}")
    return meshes[best_idx], mesh_paths[best_idx]


def clean_mesh(mesh_path, output_path):
    """Clean and optimize mesh."""
    mesh = trimesh.load(mesh_path)
    
    print(f"  - Input: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Remove disconnected components
    if len(mesh.split()) > 1:
        meshes = mesh.split(only_watertight=False)
        mesh = max(meshes, key=lambda m: len(m.vertices))
        print(f"  - Kept largest component: {len(mesh.vertices)} verts")
    
    # Clean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
    
    mesh.fix_normals()
    mesh.merge_vertices()
    
    print(f"  - Output: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}")
    
    mesh.export(output_path)
    return mesh


def apply_symmetry_refinement(mesh, axis='y'):
    """Apply symmetry refinement for symmetric objects."""
    print(f" Applying symmetry refinement (axis={axis})...")
    
    vertices = mesh.vertices.copy()
    centroid = mesh.centroid
    vertices -= centroid
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    mirrored = vertices.copy()
    mirrored[:, axis_idx] *= -1
    
    tree = cKDTree(mirrored)
    distances, indices = tree.query(vertices, k=1)
    
    threshold = np.percentile(distances, 80)
    close_mask = distances < threshold
    
    averaged = vertices.copy()
    averaged[close_mask] = (vertices[close_mask] + mirrored[indices[close_mask]]) / 2
    averaged += centroid
    
    new_mesh = mesh.copy()
    new_mesh.vertices = averaged
    
    print(f"  Modified {np.sum(close_mask)}/{len(vertices)} vertices")
    return new_mesh


# =============================================================================
# VISUALIZATION
# =============================================================================

def project_points(points_3d, pose, K):
    """Project 3D points to 2D."""
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones])
    points_cam = (pose @ points_hom.T).T
    xyz = points_cam[:, :3]
    
    # Filter points behind camera
    valid = xyz[:, 2] > 0.01
    
    projected = (K @ xyz.T).T
    z = projected[:, 2:3]
    z[z == 0] = 1e-5
    pixels = projected[:, :2] / z
    
    return pixels.astype(int), valid


def draw_pose_on_image(img_cv2, pose, mesh, K):
    """Draw 3D bounding box and axes."""
    bounds = mesh.bounds
    min_xyz, max_xyz = bounds[0], bounds[1]
    
    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ])
    
    center = mesh.centroid
    scale = np.max(mesh.extents) * 0.5
    axes_pts = np.array([
        center,
        center + [scale, 0, 0],
        center + [0, scale, 0],
        center + [0, 0, scale]
    ])

    box_2d, box_valid = project_points(corners, pose, K)
    axes_2d, axes_valid = project_points(axes_pts, pose, K)

    # Draw bounding box
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    
    h, w = img_cv2.shape[:2]
    for s, e in edges:
        if box_valid[s] and box_valid[e]:
            pt1 = tuple(np.clip(box_2d[s], 0, [w-1, h-1]))
            pt2 = tuple(np.clip(box_2d[e], 0, [w-1, h-1]))
            cv2.line(img_cv2, pt1, pt2, (0, 255, 255), 2)

    # Draw axes
    if axes_valid.all():
        origin = tuple(np.clip(axes_2d[0], 0, [w-1, h-1]))
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            end = tuple(np.clip(axes_2d[i+1], 0, [w-1, h-1]))
            cv2.arrowedLine(img_cv2, origin, end, color, 3, tipLength=0.2)
    
    return img_cv2


def visualize_depth(depth, output_path):
    """Create depth visualization."""
    depth_vis = depth.copy()
    depth_vis[depth_vis == 0] = np.nan
    
    valid_min = np.nanmin(depth_vis)
    valid_max = np.nanmax(depth_vis)
    
    depth_norm = (depth_vis - valid_min) / (valid_max - valid_min + 1e-6)
    depth_norm = np.nan_to_num(depth_norm, nan=0)
    
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colored[depth == 0] = [0, 0, 0]
    
    cv2.imwrite(output_path, depth_colored)
    return depth_colored


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    set_all_seeds(args.seed)
    
    # Setup output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MESH_OUTPUT_DIR, exist_ok=True)
    
    # Determine input files
    if args.rgb and args.depth:
        rgb_path = args.rgb
        depth_path = args.depth
    elif args.scene is not None and args.frame is not None:
        rgb_path, depth_path = find_scene_files(args.scene, args.frame)
    else:
        parser.error("Provide either --scene and --frame, or --rgb and --depth")
    
    # Header
    print(f"\n{'='*70}")
    print(f"Zero-Shot 6D Pose Estimation - RGB-D Mode")
    print(f"{'='*70}")
    print(f"Prompt:       '{args.prompt}'")
    print(f"RGB:          {os.path.basename(rgb_path)}")
    print(f"Depth:        {os.path.basename(depth_path)}")
    print(f"Ensemble:     {N_ENSEMBLE} runs")
    print(f"Symmetry:     {SYMMETRY_AXIS if SYMMETRY_AXIS else 'None'}")
    print(f"Intrinsics:   fx={INTRINSICS['fx']}, fy={INTRINSICS['fy']}")
    print(f"{'='*70}\n")

    # =========================================================================
    # STEP 1: LOAD RGB-D DATA
    # =========================================================================
    print(f"[1/5] Loading RGB-D data...")
    
    rgb = load_rgb_image(rgb_path)
    depth = load_depth_image(depth_path, INTRINSICS['depth_scale'], args.max_depth)
    
    h, w = rgb.shape[:2]
    print(f"  RGB loaded: {w}x{h}")
    print(f"  Depth loaded: {depth.shape[1]}x{depth.shape[0]}")
    print(f"     Depth range: {depth[depth > 0].min():.3f}m - {depth[depth > 0].max():.3f}m")
    print(f"     Valid pixels: {(depth > 0).sum()} / {depth.size}")
    
    # Save depth visualization
    depth_vis_path = os.path.join(OUTPUT_DIR, "depth_visualization.png")
    visualize_depth(depth, depth_vis_path)
    print(f"  Depth visualization: {depth_vis_path}")
    
    # Get camera matrix
    K = get_camera_matrix(INTRINSICS, w, h)

    # =========================================================================
    # STEP 2: OBJECT SEGMENTATION
    # =========================================================================
    print(f"\n[2/5] Segmenting object with LangSAM...")
    
    model = LangSAM()
    image_pil = Image.fromarray(rgb)
    masks, boxes, phrases, logits = model.predict(image_pil, args.prompt)

    if len(masks) == 0:
        print("❌ No object found!")
        sys.exit(1)

    mask = masks[0].numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = os.path.join(OUTPUT_DIR, "mask.png")
    mask_img.save(mask_path)
    print(f"  Segmented (confidence: {logits[0]:.2f})")
    print(f"  Mask saved: {mask_path}")
    
    # Get depth statistics in masked region
    mask_bool = mask > 0
    masked_depth = depth[mask_bool]
    valid_depth = masked_depth[masked_depth > 0]
    
    if len(valid_depth) > 0:
        print(f"  Object depth: {valid_depth.mean():.3f}m (±{valid_depth.std():.3f}m)")
    else:
        print("  No valid depth in masked region!")

    del model
    clear_gpu_memory()

    # =========================================================================
    # STEP 3: MESH GENERATION
    # =========================================================================
    print(f"\n[3/5] Generating 3D mesh...")
    
    # Prepare input
    input_image_path = os.path.join(OUTPUT_DIR, "instantmesh_input.png")
    prepare_instantmesh_input(image_pil, mask, input_image_path, resolution=512)
    print(f"  InstantMesh input: {input_image_path}")
    
    # Generate mesh(es)
    print(f"\n  Running InstantMesh...")
    
    if N_ENSEMBLE > 1:
        mesh_paths = generate_mesh_ensemble(input_image_path, MESH_OUTPUT_DIR, N_ENSEMBLE, args.seed)
        mesh, mesh_path = select_best_mesh(mesh_paths)
    else:
        mesh_path = os.path.join(MESH_OUTPUT_DIR, "mesh.obj")
        run_instantmesh_with_seed(input_image_path, mesh_path, args.seed)
        mesh = trimesh.load(mesh_path)
    
    clear_gpu_memory()
    
    # Clean mesh
    print("\n  Cleaning mesh...")
    cleaned_path = os.path.join(MESH_OUTPUT_DIR, "mesh_cleaned.obj")
    mesh = clean_mesh(mesh_path, cleaned_path)
    
    # Symmetry refinement
    if SYMMETRY_AXIS:
        mesh = apply_symmetry_refinement(mesh, SYMMETRY_AXIS)
        mesh.export(cleaned_path)
    
    final_mesh_path = cleaned_path

    # =========================================================================
    # STEP 4: POSE ESTIMATION
    # =========================================================================
    print(f"\n[4/5] Estimating 6D pose with FoundationPose...")
    
    mesh_pts = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    mesh_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device="cuda")

    print("  🔧 Initializing estimator...")
    est = FoundationPose(model_pts=mesh_pts, model_normals=mesh_normals, mesh=mesh)

    # Reduce batch size from default 64 to 16 to fit 16GB VRAM.
    # This strictly affects memory usage, NOT accuracy or final output.
    if hasattr(est, 'scorer'):
        print(f"  Reducing scorer batch size: {est.scorer.cfg.batch_size} -> 16")
        est.scorer.cfg.batch_size = 16
    if hasattr(est, 'refiner'):
        print(f"  Reducing refiner batch size: {est.refiner.cfg.batch_size} -> 16")
        est.refiner.cfg.batch_size = 16
    
    # aggressive memory cleanup before the heavy lifting
    gc.collect()
    torch.cuda.empty_cache()

    print("  Creating rotation grid...")

    # Scale mesh to reasonable size based on depth
    if len(valid_depth) > 0:
        # Estimate object size from mask and depth
        object_depth = valid_depth.mean()
        
        # Get mask bounding box size in pixels
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Convert to meters using depth and focal length
        pixel_width = cmax - cmin
        pixel_height = rmax - rmin
        
        real_width = pixel_width * object_depth / K[0, 0]
        real_height = pixel_height * object_depth / K[1, 1]
        estimated_size = max(real_width, real_height)
        
        print(f"  Estimated object size: {estimated_size*100:.1f} cm")
        
        # Scale mesh to match
        current_size = np.max(mesh.extents)
        scale_factor = estimated_size / current_size
        
        est.mesh.vertices = (est.mesh.vertices - est.mesh.centroid) * scale_factor + est.mesh.centroid
        if hasattr(est, 'pts'):
            est.pts = est.pts * scale_factor
        
        print(f"  Mesh scaled by {scale_factor:.3f}")
    else:
        # Fallback: scale to 10cm
        target_size = 0.10
        current_size = np.max(mesh.extents)
        scale_factor = target_size / current_size
        est.mesh.vertices = (est.mesh.vertices - est.mesh.centroid) * scale_factor + est.mesh.centroid
        print(f"  Mesh scaled to default 10cm")

    # Register
    print("  Registering object...")
    est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask_bool, ob_id=1, iteration=10)

    # Track
    print("  Refining pose...")
    pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=10)

    # =========================================================================
    # STEP 5: SAVE RESULTS
    # =========================================================================
    print(f"\n[5/5] Saving results...")

    if pose is not None:
        if hasattr(pose, 'cpu'):
            pose = pose.cpu().numpy()
        
        # Save pose
        pose_path = os.path.join(OUTPUT_DIR, "pose.txt")
        np.savetxt(pose_path, pose.reshape(4, 4), fmt='%.8f')
        print(f"  Pose saved: {pose_path}")
        
        # Visualization
        vis_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis_result = draw_pose_on_image(vis_img, pose, mesh, K)
        vis_path = os.path.join(OUTPUT_DIR, "result.jpg")
        cv2.imwrite(vis_path, vis_result)
        print(f"  Visualization saved: {vis_path}")
        
        # Also save RGB with mask overlay
        rgb_vis = rgb.copy()
        rgb_vis[mask_bool] = (rgb_vis[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        mask_overlay_path = os.path.join(OUTPUT_DIR, "mask_overlay.jpg")
        cv2.imwrite(mask_overlay_path, cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        
        # Summary
        print(f"\n{'='*70}")
        print(f"SUCCESS")
        print(f"{'='*70}")
        print(f"\nPose Matrix:")
        print(pose)
        
        # Extract pose info
        translation = pose[:3, 3]
        print(f"\nTranslation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m")
        print(f"Distance from camera: {np.linalg.norm(translation):.3f} m")
        
        print(f"\n Output files in: {OUTPUT_DIR}")
        print(f"   - result.jpg (visualization)")
        print(f"   - pose.txt (4x4 pose matrix)")
        print(f"   - mask.png (segmentation)")
        print(f"   - depth_visualization.png")
        print(f"   - mesh_cleaned.obj")
        print(f"\n{'='*70}\n")
        
    else:
        print("\nPose estimation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
