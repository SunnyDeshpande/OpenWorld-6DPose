# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
from estimater import *
from datareader import *
import argparse
import subprocess
import tempfile
from PIL import Image
import yaml
import sys

class SAM3MaskGenerator:
    """Bridge to SAM3 running in separate conda environment"""
    def __init__(self, sam3_env_path="sam3", sam3_script_path="~/sam3/sam3_mask_service.py"):
        self.sam3_env = sam3_env_path
        self.sam3_script = os.path.expanduser(sam3_script_path)
        logging.info(f"SAM3 bridge initialized (env: {self.sam3_env}, script: {self.sam3_script})")
    
    def generate_mask(self, rgb_image, text_prompt, score_threshold=0.5, min_pixels=500):
        """Generate segmentation mask using SAM3"""
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
            
            logging.info(f"Calling SAM3 with prompt: '{text_prompt}'")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logging.error(f"SAM3 failed: {result.stderr}")
                return None
            
            try:
                score = float(result.stdout.strip().split('\n')[-1])
                logging.info(f"SAM3 detection score: {score:.3f}")
                
                if score < 0.01:
                    logging.warning(f"No object found for '{text_prompt}'")
                    return None
            except:
                return None
            
            if os.path.exists(tmp_mask_path):
                mask_img = np.array(Image.open(tmp_mask_path))
                mask = mask_img > 127
                
                num_pixels = mask.sum()
                if num_pixels < min_pixels:
                    logging.warning(f"Mask too small ({num_pixels} < {min_pixels})")
                    return None
                
                return mask
            else:
                return None
                
        except Exception as e:
            logging.error(f"SAM3 error: {e}")
            return None
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if os.path.exists(tmp_mask_path):
                os.remove(tmp_mask_path)


# LINEMOD object ID to name mapping
LINEMOD_OBJECTS = {
    1: "ape",
    2: "benchvise", 
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone"
}

# Default prompts for LINEMOD objects
DEFAULT_PROMPTS = {
    1: "toy ape",
    2: "bench vise tool",
    4: "camera",
    5: "red can",
    6: "toy cat",
    8: "power drill",
    9: "yellow duck toy",
    10: "egg box",
    11: "glue bottle",
    12: "hole puncher",
    13: "toy iron",
    14: "desk lamp",
    15: "telephone"
}


def run_pose_estimation_worker(reader, i_frames, est, debug, ob_id, sam3_generator, 
                               sam3_prompt, device='cuda:0'):
    """Process frames with SAM3 mask generation"""
    torch.cuda.set_device(device)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)
    
    result = NestDict()
    
    for i, i_frame in enumerate(i_frames):
        logging.info(f"Frame {i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
        video_id = reader.get_video_id()
        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)
        id_str = reader.id_strs[i_frame]
        
        debug_dir = est.debug_dir
        
        # Generate mask with SAM3
        logging.info(f"Generating SAM3 mask with prompt: '{sam3_prompt}'")
        ob_mask = sam3_generator.generate_mask(color, sam3_prompt)
        
        if ob_mask is None or ob_mask.sum() == 0:
            logging.warning(f"No mask found for '{sam3_prompt}' in frame {i_frame}")
            result[video_id][id_str][ob_id] = np.eye(4)
            continue
        
        logging.info(f"SAM3 mask: {ob_mask.sum()} pixels")
        
        # Save mask visualization
        if debug >= 1:
            os.makedirs(f'{debug_dir}/masks', exist_ok=True)
            mask_vis = np.zeros_like(color)
            mask_vis[ob_mask] = color[ob_mask]
            cv2.imwrite(f'{debug_dir}/masks/frame_{i_frame:04d}_mask.png', 
                       (ob_mask * 255).astype(np.uint8))
            cv2.imwrite(f'{debug_dir}/masks/frame_{i_frame:04d}_overlay.png',
                       cv2.addWeighted(color, 0.5, mask_vis, 0.5, 0)[...,::-1])
        
        # Estimate pose
        try:
            pose = est.register(K=reader.K, rgb=color, depth=depth, 
                              ob_mask=ob_mask, ob_id=ob_id)
            
            if pose is None:
                logging.warning(f"Pose estimation failed for frame {i_frame}")
                result[video_id][id_str][ob_id] = np.eye(4)
                continue
                
            logging.info(f"Pose estimated successfully")
            
            if debug >= 2:
                # Visualize pose
                center_pose = pose @ np.linalg.inv(est.to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, 
                                       bbox=est.bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, 
                                   K=reader.K, thickness=3, transparency=0, 
                                   is_input_rgb=True)
                os.makedirs(f'{debug_dir}/vis', exist_ok=True)
                cv2.imwrite(f'{debug_dir}/vis/frame_{i_frame:04d}.png', vis[...,::-1])
            
            result[video_id][id_str][ob_id] = pose
            
        except Exception as e:
            logging.error(f"Pose estimation error: {e}")
            result[video_id][id_str][ob_id] = np.eye(4)
    
    return result


def run_pose_estimation(opt):
    """Main function to run pose estimation with SAM3"""
    wp.force_load(device='cuda')
    
    # Initialize SAM3
    sam3_generator = SAM3MaskGenerator(sam3_script_path=opt.sam3_script)
    
    debug = opt.debug
    debug_dir = opt.debug_dir
    os.makedirs(debug_dir, exist_ok=True)
    
    # Parse object IDs to process
    if opt.object_ids == 'all':
        object_ids = list(LINEMOD_OBJECTS.keys())
    else:
        object_ids = [int(x) for x in opt.object_ids.split(',')]
    
    logging.info(f"Processing objects: {[LINEMOD_OBJECTS.get(oid, oid) for oid in object_ids]}")
    
    # Load prompt mapping
    if opt.prompt_file:
        with open(opt.prompt_file, 'r') as f:
            prompt_mapping = yaml.safe_load(f)
    else:
        prompt_mapping = DEFAULT_PROMPTS
    
    res = NestDict()
    glctx = dr.RasterizeCudaContext()
    
    # Dummy mesh for initialization
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), 
                                     transform=np.eye(4)).to_mesh()
    est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), 
                        model_normals=mesh_tmp.vertex_normals.copy(), 
                        symmetry_tfs=None, mesh=mesh_tmp, 
                        scorer=None, refiner=None, glctx=glctx, 
                        debug_dir=debug_dir, debug=debug)
    
    for ob_id in object_ids:
        ob_name = LINEMOD_OBJECTS.get(ob_id, str(ob_id))
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing object {ob_id}: {ob_name}")
        logging.info(f"{'='*60}\n")
        
        # Get SAM3 prompt
        if opt.sam3_prompt:
            sam3_prompt = opt.sam3_prompt
        elif ob_id in prompt_mapping:
            sam3_prompt = prompt_mapping[ob_id]
        else:
            logging.error(f"No prompt found for object {ob_id}")
            continue
        
        logging.info(f"Using SAM3 prompt: '{sam3_prompt}'")
        
        # Load mesh
        video_dir = f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'
        if not os.path.exists(video_dir):
            logging.warning(f"Video directory not found: {video_dir}")
            continue
        
        reader = LinemodReader(video_dir, split=None)
        video_id = reader.get_video_id()
        
        try:
            if opt.use_reconstructed_mesh:
                mesh = reader.get_reconstructed_mesh(ob_id, 
                                                     ref_view_dir=opt.ref_view_dir)
            else:
                mesh = reader.get_gt_mesh(ob_id)
            
            symmetry_tfs = reader.symmetry_tfs[ob_id]
        except Exception as e:
            logging.error(f"Failed to load mesh for object {ob_id}: {e}")
            continue
        
        # Reset FoundationPose with new object
        est.reset_object(model_pts=mesh.vertices.copy(), 
                        model_normals=mesh.vertex_normals.copy(), 
                        symmetry_tfs=symmetry_tfs, mesh=mesh)
        
        # Determine frames to process
        if opt.max_frames > 0:
            num_frames = min(opt.max_frames, len(reader.color_files))
        else:
            num_frames = len(reader.color_files)
        
        frame_indices = list(range(0, num_frames, opt.frame_skip))
        logging.info(f"Processing {len(frame_indices)} frames")
        
        # Process frames
        try:
            result = run_pose_estimation_worker(
                reader, frame_indices, est, debug, ob_id, 
                sam3_generator, sam3_prompt, device='cuda:0'
            )
            
            # Merge results
            for vid in result:
                for id_str in result[vid]:
                    for oid in result[vid][id_str]:
                        res[vid][id_str][oid] = result[vid][id_str][oid]
            
            logging.info(f"✓ Completed object {ob_id}: {ob_name}")
            
        except Exception as e:
            logging.error(f"Failed to process object {ob_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = f'{debug_dir}/linemod_sam3_results.yml'
    with open(output_file, 'w') as ff:
        yaml.safe_dump(make_yaml_dumpable(res), ff)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"{'='*60}\n")
    
    return res


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run FoundationPose on LINEMOD dataset with SAM3 mask generation'
    )
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Dataset arguments
    parser.add_argument('--linemod_dir', type=str, required=True,
                       help='Path to LINEMOD dataset root directory')
    parser.add_argument('--object_ids', type=str, default='all',
                       help='Comma-separated object IDs (e.g., "1,5,8") or "all"')
    
    # Mesh arguments
    parser.add_argument('--use_reconstructed_mesh', type=int, default=0,
                       help='Use reconstructed mesh (1) or ground truth CAD (0)')
    parser.add_argument('--ref_view_dir', type=str, default='',
                       help='Reference views directory for mesh reconstruction')
    
    # SAM3 arguments
    parser.add_argument('--sam3_prompt', type=str, default=None,
                       help='Text prompt for SAM3 (overrides default prompts)')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='YAML file mapping object IDs to prompts')
    parser.add_argument('--sam3_script', type=str, 
                       default='~/sam3/sam3_mask_service.py',
                       help='Path to SAM3 service script')
    parser.add_argument('--sam3_threshold', type=float, default=0.5,
                       help='SAM3 confidence threshold')
    
    # Processing arguments
    parser.add_argument('--max_frames', type=int, default=0,
                       help='Maximum frames to process per object (0=all)')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='Process every Nth frame')
    
    # Debug arguments
    parser.add_argument('--debug', type=int, default=1,
                       help='Debug level (0=none, 1=masks, 2=visualization, 3=full)')
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug_linemod_sam3',
                       help='Debug output directory')
    
    opt = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    
    # Print configuration
    print("\n" + "="*60)
    print("LINEMOD + SAM3 Pose Estimation")
    print("="*60)
    print(f"Dataset: {opt.linemod_dir}")
    print(f"Objects: {opt.object_ids}")
    print(f"SAM3 prompt: {opt.sam3_prompt or 'Using defaults'}")
    print(f"Debug level: {opt.debug}")
    print(f"Output: {opt.debug_dir}")
    print("="*60 + "\n")
    
    run_pose_estimation(opt)
