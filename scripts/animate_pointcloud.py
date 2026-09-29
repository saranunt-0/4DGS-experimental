"""Animate a scanned tree (point cloud or 3DGS file) with wind, end to end.

    python scripts/animate_pointcloud.py oak.zip --out outputs/oak --tree-height-m 20

Input: a ``.zip`` (the largest point cloud / splat file inside is used), a
point cloud (``.ply`` without splat attributes, ``.xyz/.txt/.pts/.csv/.npy/.las/.laz``)
or a 3DGS ``.ply`` / ``.splat``.

Steps: load -> fit Gaussians (point clouds) -> up axis -> isolate the tree
(ground / clutter / floaters removed) -> skeleton + leafness -> wind simulation
-> preview (MP4 + GIF + stills) -> export (PLY sequence, USD, Blender importer).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs4d import WindAnimation, WindParams, extract_skeleton, look_at, parse_up, save_ply  # noqa: E402
from gs4d.pointcloud import load_scan  # noqa: E402
from gs4d.export import ReorientedFrames, export_all, up_rotation  # noqa: E402
from gs4d.render import framing, render, to_uint8  # noqa: E402
from gs4d.skeleton import horizontal_basis  # noqa: E402

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", default="outputs/scan_wind")
    ap.add_argument("--max-gaussians", type=int, default=600_000, help="point clouds are fitted to ~this many Gaussians")
    ap.add_argument("--up", default="auto", help="auto | +z | -y | ... | 'x,y,z'")
    ap.add_argument("--no-isolate", action="store_true", help="skip ground/background removal")
    ap.add_argument("--tree-height-m", type=float, default=0.0, help="real tree height (0 = model units are metres)")
    ap.add_argument("--slices", type=int, default=48, help="skeleton resolution")
    ap.add_argument("--leaf-hue", type=float, default=95.0, help="foliage hue in degrees (35 for autumn)")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--direction", type=float, default=0.0)
    ap.add_argument("--gustiness", type=float, default=0.6)
    ap.add_argument("--flexibility", type=float, default=1.0)
    ap.add_argument("--trunk-stiffness", type=float, default=1.5)
    ap.add_argument("--flutter-deg", type=float, default=25.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--preview-size", type=int, default=480)
    ap.add_argument("--preview-every", type=int, default=2, help="render every Nth frame for the preview")
    ap.add_argument("--formats", default="ply_sequence,usd,blender")
    ap.add_argument("--sh-degree", type=int, default=0)
    ap.add_argument("--reorient", default="Z", choices=["Z", "Y", "keep"])
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    up_arg = "auto" if args.up == "auto" else (
        parse_up(args.up) if args.up[0] in "+-" else np.array([float(v) for v in args.up.split(",")]))
    tree, up, src = load_scan(args.input, max_gaussians=args.max_gaussians, isolate=not args.no_isolate,
                              up=up_arg, work_dir=out / "_input")
    print(f"up = {np.round(up, 3).tolist()}; tree: {tree.summary()}")
    save_ply(tree, out / "tree_gaussians_static.ply")

    skel = extract_skeleton(tree, up=up, n_slices=args.slices)
    if args.leaf_hue != 95.0:
        from gs4d.skeleton import auto_leafness
        skel.leafness = auto_leafness(tree, skel.parents, skel.bind, hue_center_deg=args.leaf_hue)
    print(skel.summary())
    height = skel.height
    tree_h = args.tree_height_m or None
    if tree_h is None and not (1.0 < height < 80.0):
        print(f"warning: tree height is {height:.3g} model units - pass --tree-height-m for physical wind scaling")

    params = WindParams(speed=args.speed, direction_deg=args.direction, gustiness=args.gustiness,
                        flexibility=args.flexibility, trunk_stiffness=args.trunk_stiffness,
                        leaf_flutter_deg=args.flutter_deg, tree_height_m=tree_h,
                        duration=args.duration, fps=args.fps, loop=True)
    anim = WindAnimation(tree, skel, params)
    print(anim.summary())

    # ---- preview: across the wind, full tree + canopy close-up
    import imageio.v2 as iio

    from gs4d.viz import write_mp4

    cf, df = framing(tree)
    e1, _ = horizontal_basis(up)
    a = np.deg2rad(args.direction)
    e2 = np.cross(up, e1)
    wind_dir = np.cos(a) * e1 + np.sin(a) * e2
    side = np.cross(up, wind_dir)
    S = args.preview_size
    cam_full = look_at(cf - side * df * 0.8 + up * 0.06 * df, cf, up, S, S)
    leafy = skel.leafness > 0.5
    crown = np.asarray(tree.means[leafy].mean(0) if leafy.any() else cf, np.float64)
    cam_close = look_at(crown - side * df * 0.4, crown, up, S, S)
    frames = []
    idx = list(range(0, len(anim), max(1, args.preview_every)))
    tr = time.time()
    for k, i in enumerate(idx):
        f = anim[i]
        frames.append(to_uint8(np.concatenate([render(f, cam_full), render(f, cam_close, background=(0.86, 0.91, 0.97))], 1)))
        if k % max(1, len(idx) // 6) == 0:
            print(f"  preview {k + 1}/{len(idx)} ({time.time() - tr:.0f}s)")
    fps_prev = args.fps / max(1, args.preview_every)
    write_mp4(frames, out / "preview.mp4", fps=fps_prev, loops=2)
    small = [f[::2, ::2] for f in frames]
    iio.mimsave(out / "preview.gif", small, duration=1000 / fps_prev, loop=0)
    iio.imwrite(out / "preview_rest_vs_wind.png", np.concatenate([
        to_uint8(render(tree, cam_full)), frames[len(frames) // 3][:, :S], frames[2 * len(frames) // 3][:, :S]], 1))

    # ---- export
    frames_out = anim if args.reorient == "keep" else ReorientedFrames(anim, up_rotation(up, args.reorient))
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    res = export_all(frames_out, out / "export", "tree_wind", formats=formats, sh_degree=args.sh_degree,
                     up_axis="Z" if args.reorient != "Y" else "Y", fps=args.fps)
    summary = {
        "input": str(src), "gaussians": len(tree), "joints": skel.num_joints, "height_model_units": height,
        "frames": len(anim), "fps": args.fps, "seconds": round(time.time() - t0, 1),
        "outputs": {k: str(v) if not isinstance(v, list) else f"{len(v)} files" for k, v in res.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
