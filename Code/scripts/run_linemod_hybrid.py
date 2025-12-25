#!/usr/bin/env python
"""LINEMOD: GT mask for init, SAM3 for tracking"""
from estimater import *
import argparse
import subprocess
import tempfile
from PIL import Image
import json
import glob
import sys

class SAM3MaskGenerator:
    def __init__(self, sam3_env_path="sam3", sam3_script_path="~/sam3/sam3_mask_service.py"):
        self.sam3_env = sam3_env_path
        self.sam3_script = os.path.expanduser(sam3_script_path)
    
    def generate_mask(self, rgb_image, text_prompt, score_threshold=0.5, min_pixels=500):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img_path = tmp_img.name
            Image.fromarray(rgb_image).save(tmp_img_path)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
            tmp_mask_path = tmp_mask.name
        
        try:
            cmd = ['conda', 'run', '-n', self.sam3_env, 'python', self.sam3_script,
                   '--image', tmp_img_path, '--prompt', text_prompt,
                   '--output', tmp_mask_path, '--threshold', str(score_threshold)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                return None
            
            score = float(result.stdout.strip().split('\n')[-1])
            if score < 0.01 or not os.path.exists(tmp_mask_path):
                return None
            
            mask_img = np.array(Image.open(tmp_mask_path))
            mask = mask_img > 127
            
            if mask.sum() < min_pixels:
                return None
            
            return mask
        except:
            return None
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if os.path.exists(tmp_mask_path):
                os.remove(tmp_mask_path)


DEFAULT_PROMPTS = {
    1: "toy ape", 2: "bench vise", 4: "camera", 5: "red can", 6: "toy cat",
    8: "power drill", 9: "yellow duck", 10: "egg box", 11: "glue bottle",
    12: "hole puncher", 13: "toy iron", 14: "desk lamp", 15: "telephone"
}


def load_gt_mask(scene_dir, frame_idx, ob_id):
    """Load ground truth mask"""
    # Try mask_visib first (visible part only)
    mask_file = f'{scene_dir}/mask_visib/{frame_idx:06d}_{ob_id-1:06d}.png'
    if not os.path.exists(mask_file):
        # Try regular mask
        mask_file = f'{scene_dir}/mask/{frame_idx:06d}_{ob_id-1:06d}.png'
    
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file, -1)
        return mask > 0
    return None


def run_pose_estimation(opt):
    wp.force_load(device='cuda')
    
    sam3_generator = SAM3MaskGenerator(sam3_script_path=opt.sam3_script)
    
    debug_dir = opt.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f'{debug_dir}/masks', exist_ok=True)
    os.makedirs(f'{debug_dir}/vis', exist_ok=True)
    
    ob_id = opt.object_id
    sam3_prompt = opt.sam3_prompt or DEFAULT_PROMPTS.get(ob_id, f"object {ob_id}")
    
    print("\n" + "="*60)
    print(f"LINEMOD Hybrid: Object {ob_id}")
    print(f"First frame: Ground Truth mask")
    print(f"Tracking: SAM3 with prompt '{sam3_prompt}'")
    print("="*60 + "\n")
    
    scene_dir = f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'
    mesh_path = f'{opt.linemod_dir}/models/obj_{ob_id:06d}.ply'
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    print(f"✓ Mesh: {mesh.vertices.shape[0]} vertices")
    
    # Load camera
    with open(f'{scene_dir}/scene_camera.json', 'r') as f:
        cam_data = json.load(f)
    K = np.array(cam_data['0']['cam_K']).reshape(3, 3)
    print(f"✓ Camera intrinsics loaded")
    
    # Get files
    rgb_files = sorted(glob.glob(f'{scene_dir}/rgb/*.png'))
    depth_files = sorted(glob.glob(f'{scene_dir}/depth/*.png'))
    
    if opt.max_frames > 0:
        rgb_files = rgb_files[:opt.max_frames]
        depth_files = depth_files[:opt.max_frames]
    
    print(f"✓ Processing {len(rgb_files)} frames\n")
    
    # Initialize FoundationPose
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=opt.debug, glctx=glctx
    )
    
    poses = []
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Frame {i+1}/{len(rgb_files)}: {os.path.basename(rgb_file)}")
        
        color = cv2.imread(rgb_file)[...,::-1].copy()
        depth = cv2.imread(depth_file, -1).astype(np.float32) / 1000.0
        depth = np.ascontiguousarray(depth)
        
        # Get mask
        if i == 0:
            # Use GT mask for first frame
            mask = load_gt_mask(scene_dir, i, ob_id)
            if mask is None:
                print(f"  ✗ GT mask not found!")
                poses.append(None)
                continue
            mask = np.ascontiguousarray(mask)
            print(f"  ✓ GT mask: {mask.sum()} pixels")
        else:
            # Use SAM3 for tracking
            mask = sam3_generator.generate_mask(color, sam3_prompt, 
                                               opt.sam3_threshold, opt.sam3_min_pixels)
            if mask is None or mask.sum() == 0:
                print(f"  ✗ SAM3 failed")
                poses.append(None)
                continue
            mask = np.ascontiguousarray(mask)
            print(f"  ✓ SAM3 mask: {mask.sum()} pixels")
        
        # Save mask
        if opt.debug >= 1:
            mask_vis = np.zeros_like(color)
            mask_vis[mask] = color[mask]
            cv2.imwrite(f'{debug_dir}/masks/frame_{i:04d}_mask.png', (mask * 255).astype(np.uint8))
            cv2.imwrite(f'{debug_dir}/masks/frame_{i:04d}_overlay.png',
                       cv2.addWeighted(color, 0.5, mask_vis, 0.5, 0)[...,::-1])
        
        # Estimate pose
        try:
            if i == 0:
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, 
                                   iteration=opt.est_refine_iter)
            else:
                pose = est.track_one(rgb=color, depth=depth, K=K, 
                                    iteration=opt.track_refine_iter)
            
            if pose is None:
                print(f"  ✗ Pose failed")
                poses.append(None)
                continue
            
            print(f"  ✓ Pose estimated")
            poses.append(pose)
            
            if opt.debug >= 2:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, 
                                   K=K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/vis/frame_{i:04d}.png', vis[...,::-1])
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            poses.append(None)
    
    print(f"\n{'='*60}")
    print(f"Results: {sum(p is not None for p in poses)}/{len(poses)} successful")
    print(f"Output: {debug_dir}")
    print(f"{'='*60}\n")
    
    np.save(f'{debug_dir}/poses.npy', poses, allow_pickle=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--linemod_dir', type=str, required=True)
    parser.add_argument('--object_id', type=int, required=True)
    parser.add_argument('--sam3_prompt', type=str, default=None)
    parser.add_argument('--sam3_script', type=str, default='~/sam3/sam3_mask_service.py')
    parser.add_argument('--sam3_threshold', type=float, default=0.5)
    parser.add_argument('--sam3_min_pixels', type=int, default=500)
    parser.add_argument('--max_frames', type=int, default=0)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default='./debug_linemod_hybrid')
    
    opt = parser.parse_args()
    set_logging_format()
    set_seed(0)
    run_pose_estimation(opt)
