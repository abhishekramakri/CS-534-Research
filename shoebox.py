"""
shoebox.py
Estimates shoebox room dimensions from ScanNet-format depth frames.

Takes a scene directory containing:
  calibration.txt                       — camera intrinsics + depth scale
  images/frame-XXXXXX.depth.pgm         — uint16 depth images
  images/frame-XXXXXX.pose.txt          — 4x4 camera-to-world SE3 matrices

Samples n_frames evenly across the sequence, back-projects each depth
image into 3-D world space using the pinhole camera model and the pose,
accumulates all points into one cloud, removes outliers, then fits an
axis-aligned bounding box to get width × length × height.

Legacy PLY-based path kept as estimate_shoebox_from_ply() for synthetic scenes.
"""

import os
import glob
import numpy as np

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False

from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Calibration parsing
# ---------------------------------------------------------------------------

def parse_calibration(calib_path: str) -> dict:
    """
    Parse an InfiniTAM/ScanNet calibration.txt file.

    File format (blank lines separate sections):
      <color_w> <color_h>
      <color_fx> <color_fy>
      <color_cx> <color_cy>

      <depth_w> <depth_h>
      <depth_fx> <depth_fy>
      <depth_cx> <depth_cy>

      <extrinsic row 0>   (3x4 color-to-depth, skipped here)
      <extrinsic row 1>
      <extrinsic row 2>

      affine <scale> <offset>

    Returns dict with depth intrinsics and depth scale factor.
    """
    with open(calib_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Sections after stripping blanks:
    #   0: color_w color_h
    #   1: color_fx color_fy
    #   2: color_cx color_cy
    #   3: depth_w depth_h
    #   4: depth_fx depth_fy
    #   5: depth_cx depth_cy
    #   6-8: extrinsic rows
    #   9: affine <scale> <offset>
    depth_w, depth_h = map(int,   lines[3].split())
    fx, fy            = map(float, lines[4].split())
    cx, cy            = map(float, lines[5].split())
    scale             = float(lines[9].split()[1])   # raw_pixel * scale = metres

    return {"depth_w": depth_w, "depth_h": depth_h,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "scale": scale}


# ---------------------------------------------------------------------------
# Depth image → world-space point cloud (one frame)
# ---------------------------------------------------------------------------

def _depth_frame_to_world_points(depth_path: str, pose_path: str,
                                  calib: dict,
                                  max_depth: float = 6.0,
                                  subsample: int = 4) -> np.ndarray:
    """
    Back-project a single depth image into world-space 3-D points.

    Pinhole back-projection for pixel (u, v) with metric depth z:
        X_cam = (u - cx) * z / fx
        Y_cam = (v - cy) * z / fy
        Z_cam = z

    Then transform camera → world using the 4x4 pose:
        P_world = R @ P_cam + t

    Args:
        depth_path:  Path to .depth.pgm (uint16, raw value * scale = metres)
        pose_path:   Path to .pose.txt (4x4 camera-to-world SE3)
        calib:       Output of parse_calibration()
        max_depth:   Discard points beyond this depth in metres (sensor noise)
        subsample:   Take every Nth pixel in each axis to reduce point count

    Returns:
        (N, 3) float32 world-space XYZ array, may be empty if pose is invalid.
    """
    from PIL import Image

    # Load pose first — skip frame if pose is degenerate (common in ScanNet)
    pose = np.loadtxt(pose_path)
    if np.isinf(pose).any() or np.isnan(pose).any():
        return np.empty((0, 3), dtype=np.float32)

    # Load depth image (PIL mode 'I' = int32; values are raw uint16 depth)
    depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
    depth_m   = depth_raw * calib["scale"]          # convert raw → metres

    H, W = depth_m.shape

    # Build pixel grid at the subsampled resolution
    rows = np.arange(0, H, subsample)
    cols = np.arange(0, W, subsample)
    uu, vv = np.meshgrid(cols, rows)                # uu=col(u), vv=row(v)
    uu = uu.ravel()
    vv = vv.ravel()

    z = depth_m[vv, uu]

    # Valid pixels: depth present and within sensor range
    valid = (z > 0) & (z < max_depth)
    uu, vv, z = uu[valid], vv[valid], z[valid]

    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Pinhole back-projection to camera space
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)  # (N, 3)

    # Rigid-body transform: camera → world
    R = pose[:3, :3]                                # (3, 3)
    t = pose[:3, 3]                                 # (3,)
    pts_world = (R @ pts_cam.T).T + t              # (N, 3)

    return pts_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Statistical outlier removal
# ---------------------------------------------------------------------------

def _remove_outliers(verts: np.ndarray,
                     nb_neighbors: int = 20,
                     std_ratio: float = 2.0) -> np.ndarray:
    """Remove points that are more than std_ratio std-devs from their neighbours."""
    if _OPEN3D_AVAILABLE and len(verts) > nb_neighbors:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        cleaned, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(cleaned.points)

    # Numpy fallback: remove points whose distance to centroid exceeds threshold
    centroid  = verts.mean(axis=0)
    dists     = np.linalg.norm(verts - centroid, axis=1)
    threshold = dists.mean() + std_ratio * dists.std()
    return verts[dists <= threshold]


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def estimate_shoebox(
    scene_dir: str,
    n_frames: int = 30,
    max_depth: float = 6.0,
    subsample: int = 4,
    floor_percentile: float = 2.0,
    source_pos: list | None = None,
    listener_pos: list | None = None,
) -> dict:
    """
    Estimate shoebox room dimensions from a ScanNet scene directory.

    Args:
        scene_dir:        Path to scene (contains calibration.txt and images/).
        n_frames:         Number of frames to sample across the sequence.
        max_depth:        Max valid depth in metres (filters sensor noise).
        subsample:        Pixel subsampling factor per axis (4 → 16x fewer points).
        floor_percentile: Bottom Z-percentile used to isolate floor cluster.
        source_pos:       Override [x,y,z] for audio source (shoebox-relative).
        listener_pos:     Override [x,y,z] for listener (shoebox-relative).

    Returns:
        {
            "room_dims":      [width, length, height],  # metres
            "source_pos":     [x, y, z],
            "listener_pos":   [x, y, z],
            "floor_polygon":  np.ndarray (N, 2),
            "wall_normals":   np.ndarray (4, 3),
            "n_points":       int,
        }
    """
    calib_path = os.path.join(scene_dir, "calibration.txt")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"calibration.txt not found in {scene_dir}")
    calib = parse_calibration(calib_path)

    images_dir  = os.path.join(scene_dir, "images")
    depth_files = sorted(glob.glob(os.path.join(images_dir, "*.depth.pgm")))
    if not depth_files:
        raise FileNotFoundError(f"No *.depth.pgm files found in {images_dir}")

    # Evenly sample n_frames across the full sequence
    if n_frames >= len(depth_files):
        selected = depth_files
    else:
        idxs     = np.linspace(0, len(depth_files) - 1, n_frames, dtype=int)
        selected = [depth_files[i] for i in idxs]

    # Accumulate world-space point cloud across all selected frames
    all_points = []
    for depth_path in selected:
        base      = depth_path[: -len(".depth.pgm")]
        pose_path = base + ".pose.txt"
        if not os.path.exists(pose_path):
            continue
        pts = _depth_frame_to_world_points(
            depth_path, pose_path, calib,
            max_depth=max_depth, subsample=subsample
        )
        if len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        raise RuntimeError("No valid depth points found across sampled frames.")

    verts = np.concatenate(all_points, axis=0)
    print(f"  [shoebox] Raw cloud: {len(verts):,} pts from {len(selected)} frames")

    verts = _remove_outliers(verts)
    print(f"  [shoebox] After outlier removal: {len(verts):,} pts")

    # Separate floor and ceiling clusters using Z percentiles.
    #
    # Depth cameras scanning a room rarely point at the ceiling, so ceiling
    # coverage is sparse.  We use a tight floor percentile (p2) to find the
    # floor plane, and a very high ceiling percentile (p98) to capture the
    # sparse upward-looking points.  Even so, height will be a lower bound;
    # if a ground-truth PLY is available, prefer estimate_shoebox_from_ply()
    # for the Z dimension.
    z            = verts[:, 2]
    floor_thresh = np.percentile(z, floor_percentile)          # e.g. p2
    ceil_thresh  = np.percentile(z, 100 - floor_percentile)    # e.g. p98
    floor_verts  = verts[z <= floor_thresh]
    ceil_verts   = verts[z >= ceil_thresh]

    # Axis-aligned bounding box extent
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    width  = float(x_max - x_min)
    length = float(y_max - y_min)
    height = float(np.median(ceil_verts[:, 2]) - np.median(floor_verts[:, 2]))

    if width < 1.0 or length < 1.0 or height < 1.0:
        raise ValueError(
            f"Estimated dims [{width:.3f}, {length:.3f}, {height:.3f}] too small. "
            "Try more frames or a larger max_depth."
        )

    room_dims = [width, length, height]
    w, l, h   = room_dims

    # Positions are relative to pyroomacoustics ShoeBox origin (0,0,0)→(w,l,h)
    if source_pos is None:
        source_pos   = [w * 0.25, l * 0.5, h * 0.5]
    if listener_pos is None:
        listener_pos = [w * 0.75, l * 0.5, h * 0.5]

    # Convex hull of floor cluster XY → floor polygon
    floor_xy      = floor_verts[:, :2]
    hull          = ConvexHull(floor_xy)
    floor_polygon = floor_xy[hull.vertices]

    wall_normals = np.array([
        [ 1,  0, 0],
        [-1,  0, 0],
        [ 0,  1, 0],
        [ 0, -1, 0],
    ], dtype=float)

    return {
        "room_dims":     room_dims,
        "source_pos":    source_pos,
        "listener_pos":  listener_pos,
        "floor_polygon": floor_polygon,
        "wall_normals":  wall_normals,
        "n_points":      len(verts),
    }


# ---------------------------------------------------------------------------
# Legacy PLY path (for synthetic / Blender test scenes)
# ---------------------------------------------------------------------------

def estimate_shoebox_from_ply(
    ply_path: str,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    floor_percentile: float = 10.0,
    source_pos: list | None = None,
    listener_pos: list | None = None,
) -> dict:
    """Estimate shoebox dimensions from a PLY mesh (synthetic scenes only)."""
    try:
        import open3d as o3d
        mesh  = o3d.io.read_triangle_mesh(ply_path)
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            raise ValueError("empty mesh")
    except Exception:
        import trimesh
        mesh  = trimesh.load(ply_path, force="mesh")
        verts = np.asarray(mesh.vertices)

    verts = _remove_outliers(verts, outlier_nb_neighbors, outlier_std_ratio)

    z            = verts[:, 2]
    floor_thresh = np.percentile(z, floor_percentile)
    ceil_thresh  = np.percentile(z, 100 - floor_percentile)
    floor_verts  = verts[z <= floor_thresh]
    ceil_verts   = verts[z >= ceil_thresh]

    x_min, x_max = floor_verts[:, 0].min(), floor_verts[:, 0].max()
    y_min, y_max = floor_verts[:, 1].min(), floor_verts[:, 1].max()
    width  = float(x_max - x_min)
    length = float(y_max - y_min)
    height = float(np.median(ceil_verts[:, 2]) - np.median(floor_verts[:, 2]))

    if width < 1.0 or length < 1.0 or height < 1.0:
        raise ValueError(
            f"Estimated dims [{width:.3f}, {length:.3f}, {height:.3f}] too small."
        )

    room_dims    = [width, length, height]
    w, l, h      = room_dims
    source_pos   = source_pos   or [w * 0.25, l * 0.5, h * 0.5]
    listener_pos = listener_pos or [w * 0.75, l * 0.5, h * 0.5]

    floor_xy      = floor_verts[:, :2]
    hull          = ConvexHull(floor_xy)
    floor_polygon = floor_xy[hull.vertices]

    wall_normals = np.array([
        [ 1,  0, 0], [-1,  0, 0],
        [ 0,  1, 0], [ 0, -1, 0],
    ], dtype=float)

    return {
        "room_dims":     room_dims,
        "source_pos":    source_pos,
        "listener_pos":  listener_pos,
        "floor_polygon": floor_polygon,
        "wall_normals":  wall_normals,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    scene = sys.argv[1] if len(sys.argv) > 1 else "scannet/scene0005"
    result = estimate_shoebox(scene)
    w, l, h = result["room_dims"]
    print(f"Room dims:    {w:.2f}m x {l:.2f}m x {h:.2f}m")
    print(f"Source pos:   {[round(v, 2) for v in result['source_pos']]}")
    print(f"Listener pos: {[round(v, 2) for v in result['listener_pos']]}")
    print(f"Points used:  {result['n_points']:,}")
