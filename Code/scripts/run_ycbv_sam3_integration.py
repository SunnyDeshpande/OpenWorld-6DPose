#!/usr/bin/env python3
"""
FoundationPose + SAM3 Integration Script for YCB-Video Dataset

This script:
1. Loads YCB-V data (mesh, RGBD, ground truth)
2. Generates masks using SAM3 (via subprocess)
3. Estimates 6D pose with FoundationPose
4. Saves results in BOP format
5. Evaluates against ground truth

Usage:
    python run_ycbv_sam3_integration.py \
        --scene_id 48 \
        --object_id 2 \
        --num_frames 10 \
        --sam3_prompt "master chef can"
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import cv2
import trimesh
from pathlib import Path
import subprocess
import tempfile
from PIL import Image

# FoundationPose imports
from estimater import *
from datareader import YcbVideoReader
from Utils import *

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Object ID to Name mapping (YCB-Video)
OBJECT_NAMES = {
    1: "master chef can",
    2: "cracker box", 
    3: "sugar box",
    4: "tomato soup can",
    5: "mustard bottle",
    6: "tuna fish can",
    7: "pudding box",
    8: "gelatin box",
    9: "potted meat can",
    10: "banana",
    11: "pitcher base",
    12: "bleach cleanser",
    13: "bowl",
    14: "mug",
    15: "power drill",
    16: "wood block",
    17: "scissors",
    18: "large marker",
    19: "vertical large clamp",
    20: " horizontal extra large clamp",
    21: "foam brick"
}

# ═══════════════════════════════════════════════════════════
# SAM3 MASK GENERATOR (via Subprocess)
# ═══════════════════════════════════════════════════════════

class SAM3MaskGenerator:
    """
    Bridge to SAM3 running in separate conda environment
    
    Architecture:
    ┌──────────────┐         ┌──────────────┐
    │ FoundationPose│──────>│  SAM3 env    │
    │    (main)     │ subprocess│ (isolated)  │
    └──────────────┘         └──────────────┘
    """
    
    def __init__(self, sam3_env_path="sam3", 
                 sam3_script_path="~/sam3/sam3_mask_service.py"):
        self.sam3_env = sam3_env_path
        self.sam3_script = os.path.expanduser(sam3_script_path)
        
        # Verify SAM3 script exists
        if not os.path.exists(self.sam3_script):
            raise FileNotFoundError(
                f"SAM3 script not found at: {self.sam3_script}"
            )
        
        logging.info(f"✓ SAM3 bridge initialized")
        logging.info(f"  Environment: {self.sam3_env}")
        logging.info(f"  Script: {self.sam3_script}")
    
    def generate_mask(self, rgb_image, text_prompt, score_threshold=0.5):
        """
        Generate segmentation mask using SAM3
        
        Args:
            rgb_image: numpy array (H, W, 3) in RGB format
            text_prompt: string describing the object
            score_threshold: minimum confidence score
            
        Returns:
            mask: boolean numpy array (H, W) or None if failed
            score: confidence score
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
            
            logging.info(f"🔍 SAM3: Detecting '{text_prompt}'...")
            
            # Run SAM3 mask generation
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            sam3_time = (time.time() - start_time) * 1000  # ms
            
            if result.returncode != 0:
                logging.error(f"❌ SAM3 failed: {result.stderr}")
                return None, 0.0
            
            # Parse score from stdout
            try:
                score = float(result.stdout.strip().split('\n')[-1])
                logging.info(f"✓ SAM3 detection: score={score:.3f}, time={sam3_time:.1f}ms")
                
                if score < 0.01:
                    logging.warning(f"⚠ Low confidence for '{text_prompt}'")
                    return None, score
                    
            except Exception as e:
                logging.error(f"❌ Failed to parse SAM3 score: {e}")
                return None, 0.0
            
            # Load generated mask
            if os.path.exists(tmp_mask_path):
                mask_img = np.array(Image.open(tmp_mask_path))
                mask = mask_img > 127  # Convert to boolean
                
                # Check if mask has valid pixels
                num_pixels = mask.sum()
                if num_pixels == 0:
                    logging.warning(f"⚠ Empty mask for '{text_prompt}'")
                    return None, score
                
                logging.info(f"✓ Mask generated: {num_pixels} pixels")
                return mask, score
            else:
                logging.error("❌ SAM3 did not generate mask file")
                return None, 0.0
                
        except subprocess.TimeoutExpired:
            logging.error("❌ SAM3 timed out (>120s)")
            return None, 0.0
        except Exception as e:
            logging.error(f"❌ SAM3 error: {e}")
            return None, 0.0
        finally:
            # Cleanup temp files
            for path in [tmp_img_path, tmp_mask_path]:
                if os.path.exists(path):
                    os.remove(path)

# ═══════════════════════════════════════════════════════════
# BOP FORMAT OUTPUT
# ═══════════════════════════════════════════════════════════

class BOPResultWriter:
    """
    Write results in BOP challenge format
    
    Format:
    {
        "scene_id": int,
        "image_id": int,
        "obj_id": int,
        "score": float,
        "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
        "t": [tx, ty, tz],
        "time": float (milliseconds)
    }
    """
    
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.results = []
    
    def add_result(self, scene_id, image_id, obj_id, pose, 
                   score=1.0, time_ms=0.0):
        """
        Add a pose estimation result
        
        Args:
            scene_id: Scene/sequence ID
            image_id: Frame/image ID
            obj_id: Object ID
            pose: 4x4 transformation matrix
            score: Confidence score
            time_ms: Processing time in milliseconds
        """
        R = pose[:3, :3]
        t = pose[:3, 3] * 1000  # Convert to mm (BOP format)
        
        result = {
            "scene_id": int(scene_id),
            "image_id": int(image_id),
            "obj_id": int(obj_id),
            "score": float(score),
            "R": R.tolist(),
            "t": t.tolist(),
            "time": float(time_ms)
        }
        
        self.results.append(result)
    
    def save(self):
        """Save all results to JSON file"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"✓ Saved {len(self.results)} results to {self.output_path}")

# ═══════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════

class PoseEvaluator:
    """
    Evaluate 6D pose estimation against ground truth
    
    Metrics:
    - ADD: Average Distance of Model Points
    - ADD-S: ADD for Symmetric objects
    - Rotation Error (degrees)
    - Translation Error (cm)
    """
    
    @staticmethod
    def compute_ADD(pose_pred, pose_gt, model_points, diameter):
        """
        Average Distance of Model Points
        
        ADD = (1/m) * Σ ||R_pred*x + t_pred - (R_gt*x + t_gt)||
        
        Args:
            pose_pred: 4×4 predicted pose
            pose_gt: 4×4 ground truth pose
            model_points: Nx3 mesh vertices
            diameter: object diameter (for normalization)
            
        Returns:
            add_score: average distance (meters)
            add_auc: area under curve for threshold 0.1*diameter
        """
        R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
        R_gt, t_gt = pose_gt[:3, :3], pose_gt[:3, 3]
        
        # Transform model points
        pts_pred = (R_pred @ model_points.T).T + t_pred
        pts_gt = (R_gt @ model_points.T).T + t_gt
        
        # Compute distances
        distances = np.linalg.norm(pts_pred - pts_gt, axis=1)
        add_score = distances.mean()
        
        # Compute AUC (threshold = 0.1 * diameter)
        threshold = 0.1 * diameter
        add_auc = (distances < threshold).mean()
        
        return add_score, add_auc
    
    @staticmethod
    def compute_ADD_S(pose_pred, pose_gt, model_points, diameter):
        """
        ADD-S: ADD with nearest neighbor matching (for symmetric objects)
        
        ADD-S = (1/m) * Σ min_y ||R_pred*x + t_pred - y||
        where y ∈ {R_gt*x' + t_gt | x' ∈ model points}
        """
        R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
        R_gt, t_gt = pose_gt[:3, :3], pose_gt[:3, 3]
        
        # Transform model points
        pts_pred = (R_pred @ model_points.T).T + t_pred
        pts_gt = (R_gt @ model_points.T).T + t_gt
        
        # For each predicted point, find nearest ground truth point
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_gt)
        distances, _ = tree.query(pts_pred)
        
        add_s_score = distances.mean()
        
        # Compute AUC
        threshold = 0.1 * diameter
        add_s_auc = (distances < threshold).mean()
        
        return add_s_score, add_s_auc
    
    @staticmethod
    def compute_rotation_error(pose_pred, pose_gt):
        """
        Rotation error in degrees
        
        Error = arccos((trace(R_pred^T * R_gt) - 1) / 2)
        """
        R_pred = pose_pred[:3, :3]
        R_gt = pose_gt[:3, :3]
        
        R_diff = R_pred.T @ R_gt
        trace = np.trace(R_diff)
        
        # Clamp to valid range [-1, 3]
        trace = np.clip(trace, -1, 3)
        
        error_rad = np.arccos((trace - 1) / 2)
        error_deg = np.degrees(error_rad)
        
        return error_deg
    
    @staticmethod
    def compute_translation_error(pose_pred, pose_gt):
        """
        Translation error in centimeters
        
        Error = ||t_pred - t_gt||
        """
        t_pred = pose_pred[:3, 3]
        t_gt = pose_gt[:3, 3]
        
        error_m = np.linalg.norm(t_pred - t_gt)
        error_cm = error_m * 100
        
        return error_cm

# ═══════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pose_estimation_with_sam3(args):
    """
    Main pipeline: YCB-V → SAM3 → FoundationPose → Evaluation
    
    Pipeline Steps:
    1. Load YCB-V scene data
    2. Generate mask with SAM3
    3. Estimate pose with FoundationPose
    4. Save results in BOP format
    5. Evaluate vs ground truth
    """
    
    logging.info("="*60)
    logging.info("FoundationPose + SAM3 Integration")
    logging.info("="*60)
    
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Initialize Components
    # ═══════════════════════════════════════════════════════════
    
    logging.info("\n[STEP 1] Initializing components...")
    
    # Initialize SAM3
    sam3_generator = SAM3MaskGenerator(
        sam3_env_path=args.sam3_env,
        sam3_script_path=args.sam3_script
    )
    
    # Initialize FoundationPose
    set_logging_format()
    set_seed(0)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    # Create debug directory
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / 'masks').mkdir(exist_ok=True)
    (debug_dir / 'poses').mkdir(exist_ok=True)
    
    # Initialize result writer
    bop_writer = BOPResultWriter(
        debug_dir / f"results_scene{args.scene_id:06d}.json"
    )
    
    # ═══════════════════════════════════════════════════════════
    # STEP 2: Load YCB-V Dataset
    # ═══════════════════════════════════════════════════════════
    
    logging.info("\n[STEP 2] Loading YCB-Video dataset...")
    os.environ["YCB_VIDEO_DIR"] = args.ycbv_dir
    logging.info(f"✓ Set YCB_VIDEO_DIR: {args.ycbv_dir}")

    
    scene_path = Path(args.ycbv_dir) / "test" / f"{args.scene_id:06d}"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    
    reader = YcbVideoReader(str(scene_path), zfar=1.5)
    #reader = YcbVideoReader(str(scene_path), zfar=1.5)

    logging.info(f"First image file: {reader.color_files[0]}")
    logging.info(f"Total files: {len(reader.color_files)}")
    reader.keyframe_check = False  
    logging.info(f"✓ Loaded scene {args.scene_id:06d}")
    logging.info(f"  Total frames: {len(reader.color_files)}")
    
    # Load mesh for target object
    mesh = reader.get_gt_mesh(args.object_id)
    logging.info(f"✓ Loaded mesh for object {args.object_id}")
    logging.info(f"  Vertices: {len(mesh.vertices)}")
    logging.info(f"  Faces: {len(mesh.faces)}")
    
    # Get object diameter (for evaluation)
    extents = mesh.extents
    diameter = np.linalg.norm(extents)
    logging.info(f"  Diameter: {diameter:.4f}m")
    
    # Initialize FoundationPose with mesh
    est = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx
    )
    logging.info("✓ FoundationPose initialized")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Process Frames
    # ═══════════════════════════════════════════════════════════
    
    logging.info(f"\n[STEP 3] Processing {args.num_frames} frames...")

    # Get text prompt for object
    if args.sam3_prompt:
        text_prompt = args.sam3_prompt
    else:
        text_prompt = OBJECT_NAMES.get(args.object_id, "object")

    logging.info(f"SAM3 prompt: '{text_prompt}'")

    # Check for existing results if resume is enabled
    existing_results = {}
    if args.resume:
        existing_bop_file = debug_dir / f"results_scene{args.scene_id:06d}.json"
        if existing_bop_file.exists():
            try:
                with open(existing_bop_file, 'r') as f:
                    existing_bop = json.load(f)
                    for res in existing_bop:
                        existing_results[res['image_id']] = res
                logging.info(f"📂 Found {len(existing_results)} existing frame results to resume from")
            except Exception as e:
                logging.warning(f"⚠ Could not load existing results: {e}")

    results_summary = []
    frames_skipped = 0
    frames_processed = 0

    for frame_idx in range(min(args.num_frames, len(reader.color_files))):
        logging.info(f"\n{'─'*60}")
        logging.info(f"Frame {frame_idx}/{args.num_frames}")
        logging.info(f"{'─'*60}")

        # Check if frame already processed (check for pose file)
        pose_file = debug_dir / 'poses' / f'frame_{frame_idx:04d}.txt'
        if args.resume and pose_file.exists():
            logging.info(f"⊘ SKIPPING frame {frame_idx}: Already processed (found {pose_file.name})")

            # Try to load existing metrics from previous run
            if frame_idx in existing_results:
                # We have the BOP result, but need to reconstruct summary
                # For now, just mark it as skipped
                frames_skipped += 1
                continue
            else:
                # Pose file exists but no BOP result - reprocess to be safe
                logging.warning(f"⚠ Pose file exists but no BOP result for frame {frame_idx}, reprocessing...")

        # Load frame data
        color = reader.get_color(frame_idx)  # RGB
        depth = reader.get_depth(frame_idx)  # meters
        K = reader.K  # Camera intrinsics

        # Load ground truth
        gt_pose = reader.get_gt_pose(frame_idx, args.object_id)

        frames_processed += 1
        
        # ─────────────────────────────────────────────────────
        # Generate mask with SAM3
        # ─────────────────────────────────────────────────────
        
        mask_start = time.time()
        mask, sam3_score = sam3_generator.generate_mask(
            color, text_prompt, args.sam3_threshold
        )
        mask_time = (time.time() - mask_start) * 1000
        
        if mask is None:
            logging.warning(f"⚠ Skipping frame {frame_idx}: No mask generated")
            continue
        
        # Save mask visualization
        if args.debug >= 1:
            mask_vis = np.zeros_like(color)
            mask_vis[mask] = color[mask]
            overlay = cv2.addWeighted(color, 0.5, mask_vis, 0.5, 0)
            cv2.imwrite(
                str(debug_dir / 'masks' / f'frame_{frame_idx:04d}_overlay.png'),
                overlay[..., ::-1]  # RGB → BGR
            )
            cv2.imwrite(
                str(debug_dir / 'masks' / f'frame_{frame_idx:04d}_mask.png'),
                (mask * 255).astype(np.uint8)
            )
        
        # ─────────────────────────────────────────────────────
        # Estimate pose with FoundationPose
        # ─────────────────────────────────────────────────────
        
        pose_start = time.time()
        pose_pred = est.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.est_refine_iter
        )
        pose_time = (time.time() - pose_start) * 1000
        
        total_time = mask_time + pose_time
        
        logging.info(f"⏱ Timing: SAM3={mask_time:.1f}ms, FP={pose_time:.1f}ms, Total={total_time:.1f}ms")
        
        # ─────────────────────────────────────────────────────
        # Evaluate vs Ground Truth
        # ─────────────────────────────────────────────────────
        
        add_score, add_auc = PoseEvaluator.compute_ADD(
            pose_pred, gt_pose, mesh.vertices, diameter
        )
        
        add_s_score, add_s_auc = PoseEvaluator.compute_ADD_S(
            pose_pred, gt_pose, mesh.vertices, diameter
        )
        
        rot_error = PoseEvaluator.compute_rotation_error(
            pose_pred, gt_pose
        )
        
        trans_error = PoseEvaluator.compute_translation_error(
            pose_pred, gt_pose
        )
        
        logging.info(f"📊 Evaluation:")
        logging.info(f"  ADD:     {add_score*1000:.2f}mm (AUC: {add_auc*100:.1f}%)")
        logging.info(f"  ADD-S:   {add_s_score*1000:.2f}mm (AUC: {add_s_auc*100:.1f}%)")
        logging.info(f"  Rot Err: {rot_error:.2f}°")
        logging.info(f"  Trans Err: {trans_error:.2f}cm")
        
        # ─────────────────────────────────────────────────────
        # Save results
        # ─────────────────────────────────────────────────────
        
        # BOP format
        bop_writer.add_result(
            scene_id=args.scene_id,
            image_id=frame_idx,
            obj_id=args.object_id,
            pose=pose_pred,
            score=sam3_score,
            time_ms=total_time
        )
        
        # Save pose as text file
        np.savetxt(
            debug_dir / 'poses' / f'frame_{frame_idx:04d}.txt',
            pose_pred
        )
        
        # Summary for this frame
        results_summary.append({
            'frame': frame_idx,
            'add_auc': add_auc,
            'add_s_auc': add_s_auc,
            'rot_error': rot_error,
            'trans_error': trans_error,
            'time_ms': total_time
        })
        
        # Visualization
        if args.debug >= 1:
            # Get bounding box for visualization
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            center_pose = pose_pred @ np.linalg.inv(to_origin)

            # Draw 3D bounding box
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1,
                              K=K, thickness=3, is_input_rgb=True)

            # Save pose visualization
            cv2.imwrite(
                str(debug_dir / 'poses' / f'frame_{frame_idx:04d}_pose_vis.png'),
                vis[..., ::-1]  # RGB → BGR
            )

            cv2.imshow('Pose Estimation', vis[..., ::-1])
            cv2.waitKey(1)
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Save Final Results
    # ═══════════════════════════════════════════════════════════
    
    logging.info(f"\n{'='*60}")
    logging.info("FINAL RESULTS")
    logging.info(f"{'='*60}")

    if args.resume and frames_skipped > 0:
        logging.info(f"⊘ Frames skipped (already processed): {frames_skipped}")
        logging.info(f"✓ Frames newly processed: {frames_processed}")
        logging.info(f"📊 Total frames: {frames_skipped + frames_processed}")

    # Save BOP format results
    bop_writer.save()
    
    # Compute aggregate statistics
    if results_summary:
        add_aucs = [r['add_auc'] for r in results_summary]
        add_s_aucs = [r['add_s_auc'] for r in results_summary]
        rot_errors = [r['rot_error'] for r in results_summary]
        trans_errors = [r['trans_error'] for r in results_summary]
        times = [r['time_ms'] for r in results_summary]
        
        summary = {
            'scene_id': args.scene_id,
            'object_id': args.object_id,
            'num_frames': len(results_summary),
            'metrics': {
                'ADD_AUC_mean': float(np.mean(add_aucs)),
                'ADD_AUC_std': float(np.std(add_aucs)),
                'ADD-S_AUC_mean': float(np.mean(add_s_aucs)),
                'ADD-S_AUC_std': float(np.std(add_s_aucs)),
                'rotation_error_mean_deg': float(np.mean(rot_errors)),
                'rotation_error_std_deg': float(np.std(rot_errors)),
                'translation_error_mean_cm': float(np.mean(trans_errors)),
                'translation_error_std_cm': float(np.std(trans_errors)),
                'time_mean_ms': float(np.mean(times)),
                'time_std_ms': float(np.std(times)),
            },
            'per_frame': results_summary
        }
        
        # Save summary
        summary_path = debug_dir / f"summary_scene{args.scene_id:06d}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"\n📊 Summary Statistics:")
        logging.info(f"  ADD AUC:    {summary['metrics']['ADD_AUC_mean']*100:.1f}% ± {summary['metrics']['ADD_AUC_std']*100:.1f}%")
        logging.info(f"  ADD-S AUC:  {summary['metrics']['ADD-S_AUC_mean']*100:.1f}% ± {summary['metrics']['ADD-S_AUC_std']*100:.1f}%")
        logging.info(f"  Rot Error:  {summary['metrics']['rotation_error_mean_deg']:.2f}° ± {summary['metrics']['rotation_error_std_deg']:.2f}°")
        logging.info(f"  Trans Error: {summary['metrics']['translation_error_mean_cm']:.2f}cm ± {summary['metrics']['translation_error_std_cm']:.2f}cm")
        logging.info(f"  Time:       {summary['metrics']['time_mean_ms']:.1f}ms ± {summary['metrics']['time_std_ms']:.1f}ms")
        logging.info(f"\n✓ Summary saved to: {summary_path}")
    
    logging.info(f"\n✅ Processing complete!")
    logging.info(f"Results saved to: {debug_dir}")

# ═══════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FoundationPose + SAM3 Integration for YCB-Video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--ycbv_dir', type=str,
                       default='/home/hcp4/FoundationPose/datasets/ycbv',
                       help='Path to YCB-Video dataset')
    parser.add_argument('--scene_id', type=int, required=True,
                       help='Scene ID to process (e.g., 48)')
    parser.add_argument('--object_id', type=int, required=True,
                       help='Object ID to track (1-21)')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to process')
    
    # SAM3 arguments
    parser.add_argument('--sam3_env', type=str, default='sam3',
                       help='SAM3 conda environment name')
    parser.add_argument('--sam3_script', type=str,
                       default='~/sam3/sam3_mask_service.py',
                       help='Path to SAM3 mask service script')
    parser.add_argument('--sam3_prompt', type=str, default=None,
                       help='Text prompt for SAM3 (auto-detected if not provided)')
    parser.add_argument('--sam3_threshold', type=float, default=0.5,
                       help='SAM3 confidence threshold')
    
    # FoundationPose arguments
    parser.add_argument('--est_refine_iter', type=int, default=5,
                       help='Number of pose refinement iterations')
    
    # Output arguments
    parser.add_argument('--debug_dir', type=str,
                       default='./results_sam3',
                       help='Output directory for results')
    parser.add_argument('--debug', type=int, default=1,
                       help='Debug level (0=none, 1=save visualizations, 2=verbose)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results, skip already processed frames')

    args = parser.parse_args()
    os.environ["YCB_VIDEO_DIR"] = args.ycbv_dir
    
    # Validate arguments
    if args.object_id < 1 or args.object_id > 21:
        parser.error("object_id must be between 1 and 21")
    
    # Run pipeline
    try:
        run_pose_estimation_with_sam3(args)
    except KeyboardInterrupt:
        logging.info("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)