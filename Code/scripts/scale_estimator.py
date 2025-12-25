#!/usr/bin/env python3
"""Estimate correct mesh scale from depth measurements"""

import numpy as np
import trimesh
import cv2

class MeshScaleEstimator:
    """Match mesh scale to real-world depth observations"""
    
    @staticmethod
    def estimate_scale_from_depth(mesh, mask, depth, K):
        """Estimate mesh scale factor using depth data"""

        # Get object depth statistics
        object_depths = depth[mask > 0]
        object_depths = object_depths[object_depths > 0.01]

        if len(object_depths) == 0:
            return 1.0

        # Observed depth range (in meters)
        depth_median = np.median(object_depths)
        depth_span = np.percentile(object_depths, 95) - np.percentile(object_depths, 5)

        # Mesh extent (could be in any units)
        mesh_extents = mesh.extents
        mesh_max_extent = np.max(mesh_extents)

        # First, normalize mesh to reasonable scale (assume it should be ~0.1-0.3m)
        # Many Objaverse models are in arbitrary units
        if mesh_max_extent > 10:  # Likely in mm or cm
            base_scale = 0.001 if mesh_max_extent > 1000 else 0.01
        elif mesh_max_extent < 0.01:  # Too small
            base_scale = 10.0
        else:
            base_scale = 1.0

        # Then fine-tune based on depth
        mesh_max_extent_normalized = mesh_max_extent * base_scale
        scale_factor = depth_span / mesh_max_extent_normalized

        # Safety clamp
        scale_factor = np.clip(scale_factor * base_scale, 0.01, 100.0)

        return scale_factor
    
    @staticmethod
    def normalize_mesh_to_depth(mesh, mask, depth, K):
        """Return scaled mesh"""
        scale = MeshScaleEstimator.estimate_scale_from_depth(
            mesh, mask, depth, K
        )
        
        scaled_mesh = mesh.copy()
        scaled_mesh.vertices *= scale
        
        return scaled_mesh, scale
    
    @staticmethod
    def estimate_scale_multiple_meshes(meshes, mask, depth, K):
        """Estimate scale for multiple mesh candidates"""
        results = []

        for i, mesh in enumerate(meshes):
            scaled_mesh, scale = MeshScaleEstimator.normalize_mesh_to_depth(
                mesh, mask, depth, K
            )
            results.append({
                'original_mesh': mesh,
                'scaled_mesh': scaled_mesh,
                'scale_factor': scale,
                'mesh_index': i
            })

        return results

    @staticmethod
    def score_mesh_pose(pose, mesh, rgb, depth, mask, K, glctx=None):
        """
        Score a mesh-pose combination based on multiple factors.

        Args:
            pose: 4x4 transformation matrix
            mesh: trimesh object
            rgb: RGB image
            depth: depth image (meters)
            mask: binary mask of observed object
            K: camera intrinsics (3x3)
            glctx: OpenGL context for rendering (optional)

        Returns:
            dict: Scores including total, mask_iou, depth_agreement, silhouette_match
        """

        # Project mesh vertices to image
        vertices_cam = (pose[:3, :3] @ mesh.vertices.T).T + pose[:3, 3]
        vertices_img = (K @ vertices_cam.T).T
        vertices_img = vertices_img[:, :2] / vertices_img[:, 2:3]

        H, W = mask.shape[:2]

        # Create rendered mask (simple version using vertex projection)
        rendered_mask = np.zeros_like(mask, dtype=np.uint8)

        # Check which vertices project inside the image
        valid_verts = (
            (vertices_img[:, 0] >= 0) & (vertices_img[:, 0] < W) &
            (vertices_img[:, 1] >= 0) & (vertices_img[:, 1] < H) &
            (vertices_cam[:, 2] > 0)  # in front of camera
        )

        if valid_verts.sum() > 0:
            points = vertices_img[valid_verts].astype(np.int32)
            # Create convex hull of projected points as simple mask
            if len(points) >= 3:
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(rendered_mask, hull, 1)

        # Score 1: Mask IoU
        intersection = np.logical_and(mask, rendered_mask).sum()
        union = np.logical_or(mask, rendered_mask).sum()
        mask_iou = intersection / (union + 1e-8)

        # Score 2: Depth agreement (for pixels in observed mask)
        depth_score = 0.0
        if mask.sum() > 0:
            # Get rendered depth at mask locations
            rendered_depth = np.full_like(depth, np.inf)

            for vert_idx in np.where(valid_verts)[0]:
                x, y = int(vertices_img[vert_idx, 0]), int(vertices_img[vert_idx, 1])
                if 0 <= x < W and 0 <= y < H:
                    z = vertices_cam[vert_idx, 2]
                    rendered_depth[y, x] = min(rendered_depth[y, x], z)

            # Compare depths in masked region
            mask_pixels = mask > 0
            valid_depth = (depth[mask_pixels] > 0.01) & (rendered_depth[mask_pixels] < 10)

            if valid_depth.sum() > 10:
                depth_diff = np.abs(depth[mask_pixels][valid_depth] -
                                   rendered_depth[mask_pixels][valid_depth])
                depth_score = np.exp(-depth_diff.mean() * 5.0)  # Exponential decay

        # Score 3: Silhouette matching (edge alignment)
        observed_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        rendered_edges = cv2.Canny((rendered_mask * 255).astype(np.uint8), 50, 150)

        # Distance transform for silhouette matching
        if observed_edges.sum() > 0 and rendered_edges.sum() > 0:
            dist_obs = cv2.distanceTransform(255 - observed_edges, cv2.DIST_L2, 3)
            dist_render = cv2.distanceTransform(255 - rendered_edges, cv2.DIST_L2, 3)

            # Average distance from rendered edges to observed edges
            edge_dist_obs = dist_obs[rendered_edges > 0].mean() if rendered_edges.sum() > 0 else 100
            edge_dist_render = dist_render[observed_edges > 0].mean() if observed_edges.sum() > 0 else 100

            silhouette_score = np.exp(-(edge_dist_obs + edge_dist_render) / 20.0)
        else:
            silhouette_score = 0.0

        # Score 4: Coverage (how much of the mask is covered)
        coverage = intersection / (mask.sum() + 1e-8)

        # Score 5: Compactness (penalize too-large rendered masks)
        compactness = mask.sum() / (rendered_mask.sum() + 1e-8)
        compactness = min(compactness, 1.0)  # Don't reward being smaller than target

        # Weighted combination
        total_score = (
            0.35 * mask_iou +           # Main metric: IoU
            0.25 * depth_score +        # Depth alignment
            0.20 * silhouette_score +   # Edge alignment
            0.10 * coverage +           # Coverage completeness
            0.10 * compactness          # Size appropriateness
        )

        return {
            'total': total_score,
            'mask_iou': mask_iou,
            'depth_agreement': depth_score,
            'silhouette_match': silhouette_score,
            'coverage': coverage,
            'compactness': compactness
        }