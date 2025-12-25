# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
from estimater import *
from datareader import *
import argparse
import subprocess
import tempfile
from PIL import Image
import sys

class SAM3MaskGenerator:
    """Bridge to SAM3 running in separate conda environment"""
    def __init__(self, sam3_env_path="sam3", sam3_script_path="~/sam3/sam3_mask_service.py"):
        self.sam3_env = sam3_env_path
        self.sam3_script = os.path.expanduser(sam3_script_path)
        logging.info(f"SAM3 bridge initialized (env: {self.sam3_env}, script: {self.sam3_script})")
    
    def generate_mask(self, rgb_image, text_prompt, score_threshold=0.5):
        """
        Generate segmentation mask using SAM3 via subprocess
        Args:
            rgb_image: numpy array (H, W, 3) in RGB format
            text_prompt: string describing the object
            score_threshold: minimum confidence score
        Returns:
            mask: boolean numpy array (H, W) or None if no object found
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img_path = tmp_img.name
            Image.fromarray(rgb_image).save(tmp_img_path)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
            tmp_mask_path = tmp_mask.name
        
        try:
            # Build command to run SAM3 in its conda environment
            cmd = [
                'conda', 'run', '-n', self.sam3_env, 'python',
                self.sam3_script,
                '--image', tmp_img_path,
                '--prompt', text_prompt,
                '--output', tmp_mask_path,
                '--threshold', str(score_threshold)
            ]
            
            logging.info(f"Calling SAM3 with prompt: '{text_prompt}'")
            
            # Run SAM3 mask generation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logging.error(f"SAM3 failed with error: {result.stderr}")
                return None
            
            # Parse score from stdout
            try:
                score = float(result.stdout.strip().split('\n')[-1])
                logging.info(f"SAM3 detection score: {score:.3f}")
                
                # If score is 0, no object was found
                if score < 0.01:
                    logging.warning(f"No object found for prompt '{text_prompt}'")
                    return None
                    
            except Exception as e:
                logging.error(f"Failed to parse score: {e}")
                return None
            
            # Load generated mask
            if os.path.exists(tmp_mask_path):
                mask_img = np.array(Image.open(tmp_mask_path))
                mask = mask_img > 127  # Convert to boolean
                
                # Check if mask has any valid pixels
                num_pixels = mask.sum()
                if num_pixels == 0:
                    logging.warning(f"Mask is empty for prompt '{text_prompt}'")
                    return None
                
                return mask
            else:
                logging.error("SAM3 did not generate mask file")
                return None
                
        except subprocess.TimeoutExpired:
            logging.error("SAM3 timed out")
            return None
        except Exception as e:
            logging.error(f"SAM3 bridge error: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
        finally:
            # Cleanup temp files
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if os.path.exists(tmp_mask_path):
                os.remove(tmp_mask_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    
    # SAM3 arguments
    parser.add_argument('--use_sam3', action='store_true', help='Use SAM3 for semantic mask generation')
    parser.add_argument('--sam3_prompt', type=str, default='mustard bottle', help='Text prompt for SAM3')
    parser.add_argument('--sam3_threshold', type=float, default=0.5, help='Confidence threshold for SAM3')
    parser.add_argument('--sam3_script', type=str, default='~/sam3/sam3_mask_service.py', help='Path to SAM3 service script')
    
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, 
                        scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # Initialize SAM3 bridge if requested
    sam3_generator = None
    if args.use_sam3:
        sam3_generator = SAM3MaskGenerator(sam3_script_path=args.sam3_script)

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i==0:
            # Generate mask using SAM3 only (no fallback to precomputed)
            if args.use_sam3 and sam3_generator is not None:
                logging.info(f"Generating mask with SAM3 using prompt: '{args.sam3_prompt}'")
                mask = sam3_generator.generate_mask(color, args.sam3_prompt, args.sam3_threshold)
                
                # Check if mask was successfully generated
                if mask is None or mask.sum() == 0:
                    print("\n" + "="*60)
                    print(f"NO OBJECT FOUND FOR PROMPT: '{args.sam3_prompt}'")
                    print("="*60 + "\n")
                    logging.error(f"SAM3 could not find '{args.sam3_prompt}' in the scene. Exiting.")
                    sys.exit(1)
                else:
                    logging.info(f"SAM3 mask shape: {mask.shape}, True pixels: {mask.sum()}")
                    print(f"\n✓ Object '{args.sam3_prompt}' detected successfully with {mask.sum()} pixels\n")
                    
                    # Save SAM3 generated mask for debugging
                    if debug >= 1:
                        import cv2
                        mask_vis = np.zeros_like(color)
                        mask_vis[mask] = color[mask]
                        cv2.imwrite(f'{debug_dir}/sam3_mask_overlay.png', 
                                   cv2.addWeighted(color, 0.5, mask_vis, 0.5, 0)[...,::-1])
                        cv2.imwrite(f'{debug_dir}/sam3_mask.png', (mask * 255).astype(np.uint8))
            else:
                # If not using SAM3, use precomputed mask
                mask = reader.get_mask(0).astype(bool)

            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    print("\n✓ Processing completed successfully!\n")