#!/usr/bin/env python
"""Simplified LINEMOD + SAM3 - bypasses problematic LinemodReader"""
from estimater import *
import argparse
import subprocess
import tempfile
from PIL import Image
import json
import glob
import sys

class SAM3MaskGenerator:
    """Bridge to SAM3"""
    def __init__(self, sam3_env_path="sam3", sam3_script_path="~/sam3/sam3_mask_service.py"):
        self.sam3_env = sam3_env_path
        self.sam3_script = os.path.expanduser(sam3_script_path)
        logging.info(f"SAM3 initialized: {self.sam3_script}")
    
    def generate_mask(self, rgb_image, text_prompt, score_threshold=0.5, min_pixels=500):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img_path = tmp_img.name
            Image.fromarray(rgb_image).save(tmp_img_path)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
            tmp_mask_path = tmp_mask.name
        
        try:
            cmd = [
                'conda', 'run', '-n', self.sam3_env, 'python',
                self.sam3_script,
                '--image', tmp_img_path,
                '--prompt', text_prompt,
                '--output', tmp_mask_path,
                '--threshold', str(score_threshold)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logging.error(f"SAM3 failed: {result.stderr}")
                return None
            
            score = float(result.stdout.strip().split('\n')[-1])
            logging.info(f"SAM3 score: {score:.3f}")
            
            if score < 0.01:
                return None
            
            if os.path.exists(tmp_mask_path):
                mask_img = np.array(Image.open(tmp_mask_path))
                mask = mask_img > 127
                
                if mask.sum() < min_pixels:
                    logging.warning(f"Mask too small: {mask.sum()} pixels")
                    return None
                
                return mask
            return None
                
        except Exception as e:
            logging.error(f"SAM3 error: {e}")
            return None
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if os.path.exists(tmp_mask_path):
                os.remove(tmp_mask_path)


LINEMOD_OBJECTS = {
    1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue", 
    12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
}

DEFAULT_PROMPTS = {
    1: "toy ape", 2: "bench vise", 4: "camera", 5: "red can", 6: "toy cat",
    8: "power drill", 9: "yellow duck", 10: "egg box", 11: "glue bottle",
    12: "hole puncher", 13: "toy iron", 14: "desk lamp", 15: "telephone"
}


def load_camera_intrinsics(scene_dir):
    """Load camera intrinsics from scene_camera.json"""
    with open(f'{scene_dir}/scene_camera.json', 'r') as f:
        cam_data = json.load(f)
    
    # Get K from first frame
    first_frame = list(cam_data.keys())[0]
    K = np.array(cam_data[first_frame]['cam_K']).reshape(3, 3)
    return K


def run_pose_estimation(opt):
    """Main function"""
    wp.force_load(device='cuda')
    
    sam3_generator = SAM3MaskGenerator(sam3_script_path=opt.sam3_script)
    
    debug_dir = opt.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/masks', exist_ok=True)
    os.makedirs(f'{debug_dir}/vis', exist_ok=True)
    
    # Get object ID
    ob_id = opt.object_id
    ob_name = LINEMOD_OBJECTS.get(ob_id, str(ob_id))
    
    print("\n" + "="*60)
    print(f"Processing LINEMOD Object {ob_id}: {ob_name}")
    print("="*60 + "\n")
    
    # Get prompt
    sam3_prompt = opt.sam3_prompt or DEFAULT_PROMPTS.get(ob_id, ob_name)
    print(f"SAM3 prompt: '{sam3_prompt}'")
    
    # Paths
    scene_dir = f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'
    mesh_path = f'{opt.linemod_dir}/models/obj_{ob_id:06d}.ply'
    
    if not os.path.exists(scene_dir):
        print(f"✗ Scene not found: {scene_dir}")
        sys.exit(1)
    
    if not os.path.exists(mesh_path):
        print(f"✗ Mesh not found: {mesh_path}")
        sys.exit(1)
    
    # Load mesh
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"✓ Mesh loaded: {mesh.vertices.shape[0]} vertices")
    
    # Load camera intrinsics
    K = load_camera_intrinsics(scene_dir)
    print(f"✓ Camera K loaded")
    
    # Get RGB files
    rgb_files = sorted(glob.glob(f'{scene_dir}/rgb/*.png'))
    depth_files = sorted(glob.glob(f'{scene_dir}/depth/*.png'))
    
    if opt.max_frames > 0:
        rgb_files = rgb_files[:opt.max_frames]
        depth_files = depth_files[:opt.max_frames]
    
    print(f"✓ Found {len(rgb_files)} frames\n")
    
    # Initialize FoundationPose
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=opt.debug,
        glctx=glctx
    )
    
    print("✓ FoundationPose initialized\n")
    
    # Process frames
    poses = []
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Frame {i+1}/{len(rgb_files)}: {os.path.basename(rgb_file)}")
        
        # Load images - IMPORTANT: make arrays contiguous
        color = cv2.imread(rgb_file)[...,::-1].copy()  # BGR to RGB, make contiguous
        depth = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0  # Convert to meters
        depth = np.ascontiguousarray(depth)  # Ensure contiguous
        
        # Generate mask with SAM3
        mask = sam3_generator.generate_mask(color, sam3_prompt, 
                                           opt.sam3_threshold, opt.sam3_min_pixels)
        
        if mask is None or mask.sum() == 0:
            print(f"  ✗ No mask found")
            poses.append(None)
            continue
        
        print(f"  ✓ SAM3 mask: {mask.sum()} pixels")
        
        # Make mask contiguous
        mask = np.ascontiguousarray(mask)
        
        # Save mask visualization
        if opt.debug >= 1:
            mask_vis = np.zeros_like(color)
            mask_vis[mask] = color[mask]
            cv2.imwrite(f'{debug_dir}/masks/frame_{i:04d}_mask.png', 
                       (mask * 255).astype(np.uint8))
            cv2.imwrite(f'{debug_dir}/masks/frame_{i:04d}_overlay.png',
                       cv2.addWeighted(color, 0.5, mask_vis, 0.5, 0)[...,::-1])
        
        # Estimate pose
        try:
            if i == 0:
                # Register on first frame
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, 
                                   iteration=opt.est_refine_iter)
            else:
                # Track on subsequent frames
                pose = est.track_one(rgb=color, depth=depth, K=K, 
                                    iteration=opt.track_refine_iter)
            
            if pose is None:
                print(f"  ✗ Pose estimation failed")
                poses.append(None)
                continue
            
            print(f"  ✓ Pose estimated")
            poses.append(pose)
            
            # Visualize
            if opt.debug >= 2:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, 
                                   K=K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/vis/frame_{i:04d}.png', vis[...,::-1])
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            poses.append(None)
    
    # Save results
    print(f"\n{'='*60}")
    print(f"Completed! Processed {len(poses)} frames")
    print(f"Successful: {sum(p is not None for p in poses)}")
    print(f"Failed: {sum(p is None for p in poses)}")
    print(f"Output: {debug_dir}")
    print(f"{'='*60}\n")
    
    # Save poses
    np.save(f'{debug_dir}/poses.npy', np.array(poses))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LINEMOD + SAM3 (Simplified)')
    
    parser.add_argument('--linemod_dir', type=str, required=True,
                       help='LINEMOD dataset root')
    parser.add_argument('--object_id', type=int, required=True,
                       help='Object ID (1-15, excluding 3,7)')
    parser.add_argument('--sam3_prompt', type=str, default=None,
                       help='SAM3 text prompt')
    parser.add_argument('--sam3_script', type=str, 
                       default='~/sam3/sam3_mask_service.py')
    parser.add_argument('--sam3_threshold', type=float, default=0.5)
    parser.add_argument('--sam3_min_pixels', type=int, default=500)
    parser.add_argument('--max_frames', type=int, default=0,
                       help='Max frames to process (0=all)')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default='./debug_linemod_sam3')
    
    opt = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    
    run_pose_estimation(opt)
