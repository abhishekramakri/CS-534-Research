"""
shoebox.py
Estimates shoebox room dimensions from a PLY mesh using open3d.
Outputs room_dims, source/listener positions, and floor geometry.
"""

import numpy as np

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False

from scipy.spatial import ConvexHull


def _load_vertices(ply_path: str) -> np.ndarray:
    """Load vertex array from PLY, trying open3d first then trimesh."""
    if _OPEN3D_AVAILABLE:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        verts = np.asarray(mesh.vertices)
        if len(verts) > 0:
            return verts
    if _TRIMESH_AVAILABLE:
        mesh = trimesh.load(ply_path, force="mesh")
        return np.asarray(mesh.vertices)
    raise ImportError("Neither open3d nor trimesh is available. Install one of them.")


def _remove_outliers(verts: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Statistical outlier removal via open3d, or a numpy fallback."""
    if _OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        cleaned, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(cleaned.points)
    # Numpy fallback: remove points more than std_ratio std devs from centroid
    centroid = verts.mean(axis=0)
    dists = np.linalg.norm(verts - centroid, axis=1)
    threshold = dists.mean() + std_ratio * dists.std()
    return verts[dists <= threshold]


def _pca_align(verts: np.ndarray):
    """
    Check if the XY footprint is axis-aligned.
    If the dominant PCA axis deviates >5 degrees from X, rotate to align.
    Returns (rotated_verts, rotation_angle_rad).
    """
    xy = verts[:, :2]
    cov = np.cov(xy.T)
    _, eigvecs = np.linalg.eigh(cov)
    dominant = eigvecs[:, -1]  # eigenvector for largest eigenvalue
    angle = np.arctan2(dominant[1], dominant[0])
    if abs(angle) < np.deg2rad(5):
        return verts, 0.0
    # Rotate XY plane so dominant axis aligns with X
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ verts.T).T, angle


def estimate_shoebox(
    ply_path: str,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    floor_percentile: float = 10.0,
    source_pos: list | None = None,
    listener_pos: list | None = None,
) -> dict:
    """
    Estimate shoebox room dimensions from a PLY mesh.

    Args:
        ply_path:              Path to .ply file.
        outlier_nb_neighbors:  Neighbors for statistical outlier removal.
        outlier_std_ratio:     Std deviation multiplier for outlier removal.
        floor_percentile:      Bottom Z-percentile used to isolate floor vertices.
        source_pos:            Override [x, y, z] for audio source. Derived if None.
        listener_pos:          Override [x, y, z] for listener. Derived if None.

    Returns:
        {
            "room_dims":      [width, length, height],   # meters
            "source_pos":     [x, y, z],
            "listener_pos":   [x, y, z],
            "floor_polygon":  np.ndarray shape (N, 2),   # XY convex hull of floor cluster
            "wall_normals":   np.ndarray shape (4, 3),   # outward normals of 4 walls
        }
    """
    verts = _load_vertices(ply_path)
    verts = _remove_outliers(verts, outlier_nb_neighbors, outlier_std_ratio)
    verts, rotation_angle = _pca_align(verts)

    z = verts[:, 2]
    floor_thresh = np.percentile(z, floor_percentile)
    ceil_thresh = np.percentile(z, 100 - floor_percentile)

    floor_verts = verts[z <= floor_thresh]
    ceil_verts = verts[z >= ceil_thresh]

    # AABB on floor cluster for XY footprint
    x_min, x_max = floor_verts[:, 0].min(), floor_verts[:, 0].max()
    y_min, y_max = floor_verts[:, 1].min(), floor_verts[:, 1].max()

    width = x_max - x_min
    length = y_max - y_min
    height = np.median(ceil_verts[:, 2]) - np.median(floor_verts[:, 2])

    if width < 1.0 or length < 1.0 or height < 1.0:
        raise ValueError(
            f"Estimated room dims [{width:.3f}, {length:.3f}, {height:.3f}] are too small. "
            "Check that the PLY is exported in meters (Blender may default to cm)."
        )

    room_dims = [float(width), float(length), float(height)]
    w, l, h = room_dims

    # Positions must be relative to the ShoeBox origin (0,0,0) → (w,l,h).
    # pra.ShoeBox always places its corner at the origin regardless of world coords.
    floor_z_median = float(np.median(floor_verts[:, 2]))
    if source_pos is None:
        source_pos = [w * 0.25, l * 0.5, h * 0.5]
    if listener_pos is None:
        listener_pos = [w * 0.75, l * 0.5, h * 0.5]

    # Floor polygon: convex hull of floor cluster XY
    floor_xy = floor_verts[:, :2]
    hull = ConvexHull(floor_xy)
    floor_polygon = floor_xy[hull.vertices]

    # 4 axis-aligned wall outward normals
    wall_normals = np.array([
        [1, 0, 0],   # east  (+X)
        [-1, 0, 0],  # west  (-X)
        [0, 1, 0],   # north (+Y)
        [0, -1, 0],  # south (-Y)
    ], dtype=float)

    return {
        "room_dims":     room_dims,
        "source_pos":    source_pos,
        "listener_pos":  listener_pos,
        "floor_polygon": floor_polygon,
        "wall_normals":  wall_normals,
    }


if __name__ == "__main__":
    import os
    ply = os.path.join(os.path.dirname(__file__), "testing", "modernroom.ply")
    result = estimate_shoebox(ply)
    w, l, h = result["room_dims"]
    print(f"Room dims:    {w:.2f}m x {l:.2f}m x {h:.2f}m")
    print(f"Source pos:   {[round(v,2) for v in result['source_pos']]}")
    print(f"Listener pos: {[round(v,2) for v in result['listener_pos']]}")
    print(f"Floor polygon vertices: {len(result['floor_polygon'])}")
