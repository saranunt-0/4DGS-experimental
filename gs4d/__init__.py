"""gs4d - animate static 3D Gaussian Splatting trees with wind and export 4D splats.

Quick start::

    from gs4d import generate_tree, WindAnimation, WindParams, export_all
    model, skeleton = generate_tree()
    anim = WindAnimation(model, skeleton, WindParams(speed=6, duration=4))
    export_all(anim, "out")          # PLY sequence + USD + Blender importer

For a captured tree::

    from gs4d import load_gaussians, isolate_object, extract_skeleton
    scene = load_gaussians("my_tree.ply")
    keep, _ = isolate_object(scene, up="-y")
    tree = scene.subset(keep)
    skeleton = extract_skeleton(tree, up="-y")
"""

from .gaussians import GaussianModel, SH_C0
from .io_ply import load_gaussians, save_ply, save_splat
from .skeleton import TreeSkeleton, extract_skeleton, guess_up_axis, estimate_up, color_leafness, parse_up, subset_skeleton
from .procedural_tree import TreeParams, generate_tree
from .wind import WindParams, WindAnimation, animate_tree
from .cleanup import isolate_object, mask_vote
from .render import Camera, look_at, orbit_cameras, render, render_cpu
from .export import export_all, export_ply_sequence, export_usd, export_npz, ReorientedFrames, up_rotation

__version__ = "0.1.0"

__all__ = [
    "GaussianModel", "SH_C0", "load_gaussians", "save_ply", "save_splat",
    "TreeSkeleton", "extract_skeleton", "guess_up_axis", "estimate_up", "color_leafness", "parse_up", "subset_skeleton",
    "TreeParams", "generate_tree", "WindParams", "WindAnimation", "animate_tree",
    "isolate_object", "mask_vote", "Camera", "look_at", "orbit_cameras", "render", "render_cpu",
    "export_all", "export_ply_sequence", "export_usd", "export_npz", "ReorientedFrames", "up_rotation",
]
