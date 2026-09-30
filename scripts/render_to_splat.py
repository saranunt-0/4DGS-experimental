"""Render-to-splat vs direct fit: two ways to turn a scanned point cloud into an object-only 3DGS.

    python scripts/render_to_splat.py test_asset/200-year-old-oak-tree.zip --out outputs/oak_distill

A: *direct fit*   - fit one Gaussian per voxel of points (``gs4d.pointcloud``, no training);
B: *render-to-splat* - photograph the point cloud from a camera rig (RGBA, known poses) and train a
   3DGS on those images only, like a photo capture with perfect masks and cameras.

Stages (each result is cached in ``--out``; delete a file to redo that stage):

  1 scan    zip / ply -> sky-bleed repair -> isolate the tree            -> pointcloud.npz
  2 render  camera rig -> RGBA point renders + COLMAP text model         -> dataset/train/, dataset/test/
  3 train   3DGS from dataset/train only (visual-hull init, masked,      -> render_to_splat.ply, train_log.json
            random background, capped densification)
  4 direct  Gaussians fitted to the points at the same budget, and 500k  -> direct_equal.ply, direct_500k.ply
  5 eval    held-out test views: PSNR / SSIM / silhouette + image grids  -> metrics.json, compare_full.png,
                                                                             compare_zoom.png

Runs on the CPU (``gs4d.torch_raster``) or, when CUDA + gsplat are available, on the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gs4d import load_gaussians, save_ply  # noqa: E402
from gs4d.distill import (distill_rig, evaluate, load_rendered_dataset, render_points,  # noqa: E402
                          save_json, write_dataset)
from gs4d.pointcloud import (PointCloud, extract_from_zip, fix_sky_bleed, isolate_point_cloud,  # noqa: E402
                             pointcloud_to_gaussians, read_point_cloud)
from gs4d.skeleton import parse_up  # noqa: E402


def stage_scan(args, out: Path) -> tuple[PointCloud, np.ndarray]:
    f = out / "pointcloud.npz"
    if f.exists():
        d = np.load(f)
        print(f"[1 scan] cached {f.name}: {len(d['points']):,} points")
        return PointCloud(d["points"], d["colors"], None, d["offset"]), d["up"]
    t = time.time()
    path = Path(args.input)
    if path.suffix.lower() == ".zip":
        path = extract_from_zip(path, out / "_input")
    pc = fix_sky_bleed(read_point_cloud(path))
    pc, up, _ = isolate_point_cloud(pc, up=args.up if args.up == "auto" else parse_up(args.up))
    np.savez(f, points=pc.points, colors=pc.colors, offset=pc.offset, up=np.asarray(up, np.float64))
    print(f"[1 scan] {len(pc):,} object points, up {np.round(up, 3).tolist()} ({time.time() - t:.0f}s)")
    return pc, np.asarray(up, np.float64)


def stage_render(args, out: Path, pc: PointCloud, up: np.ndarray):
    rig = distill_rig(pc.points, up=up, width=args.res, height=args.res, fov_deg=args.fov, n_test=args.test_views)
    ds = out / "dataset"
    for split, cams in (("train", rig.train), ("test", rig.test)):
        d = ds / split
        if (d / "sparse" / "0" / "images.txt").exists():
            print(f"[2 render] cached {split}: {len(cams)} views")
            continue
        t = time.time()
        imgs, alphas = [], []
        for i, cam in enumerate(cams):
            rgb, a = render_points(pc.points, pc.colors, cam, point_size=args.point_size)
            imgs.append(rgb)
            alphas.append(a)
            if i % 25 == 0:
                print(f"  {split} view {i + 1}/{len(cams)} ({time.time() - t:.0f}s)", flush=True)
        write_dataset(d, cams, imgs, alphas, prefix=split)
        print(f"[2 render] {split}: {len(cams)} views at {args.res}px ({time.time() - t:.0f}s)")
    return rig


def stage_train(args, out: Path) -> tuple:
    import torch

    from gs4d.train import TrainConfig, train_object_splat

    f = out / "render_to_splat.ply"
    if f.exists():
        print(f"[3 train] cached {f.name}")
        return load_gaussians(f), json.loads((out / "train_log.json").read_text())
    gpu = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if gpu:
        rasterize_fn, device = None, "cuda"
    else:
        from gs4d.torch_raster import rasterize_cpu
        rasterize_fn, device = rasterize_cpu, "cpu"
    train = load_rendered_dataset(out / "dataset" / "train")
    test = load_rendered_dataset(out / "dataset" / "test")
    probe = [0, 3, len(test.images) - 2, len(test.images) - 1]
    gt = [test.images[i] / 255.0 for i in probe]
    ga = [test.masks[i] for i in probe]
    log = {"device": device, "curve": []}
    t0 = time.time()

    def on_eval(step, model):
        m = evaluate({"m": model}, [test.cameras[i] for i in probe], gt, ga)["m"]
        log["curve"].append({"step": step, "gaussians": len(model), "psnr": round(m["psnr"], 3),
                             "ssim": round(m["ssim"], 4), "seconds": round(time.time() - t0, 1)})
        print(f"  [eval] step {step}: {len(model):,} splats, test PSNR {m['psnr']:.2f} dB, SSIM {m['ssim']:.3f}",
              flush=True)
        save_json(out / "train_log.json", log)

    cfg = TrainConfig(iters=args.iters, sh_degree=args.sh_degree, device=device, max_gaussians=args.max_gaussians,
                      init_points=args.init_points, eval_every=args.eval_every, log_every=args.eval_every // 2,
                      reset_every=args.reset_every, alpha_lambda=0.3, random_background=True,
                      init_scale=args.init_scale,
                      res_schedule=tuple(tuple(int(v) for v in p.split(":")) for p in args.res_schedule.split(",")))
    log["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    print(f"[3 train] {len(train.images)} views, {args.iters} steps on {device}")
    model = train_object_splat(train, cfg, rasterize_fn=rasterize_fn, callback=on_eval)
    keep = model.opacities > 1.0 / 255.0
    model = model.subset(np.flatnonzero(keep))
    log["seconds"] = round(time.time() - t0, 1)
    log["gaussians"] = len(model)
    save_json(out / "train_log.json", log)
    save_ply(model, f)
    print(f"[3 train] {len(model):,} splats ({log['seconds']:.0f}s)")
    return model, log


def stage_direct(args, out: Path, pc: PointCloud, n_equal: int) -> dict:
    res = {}
    for name, n in (("direct_equal", n_equal), ("direct_500k", args.direct_full)):
        f = out / f"{name}.ply"
        if f.exists():
            res[name] = (load_gaussians(f), None)
            continue
        t = time.time()
        m = pointcloud_to_gaussians(pc, target_count=n)
        dt = time.time() - t
        save_ply(m, f)
        res[name] = (m, round(dt, 1))
        print(f"[4 direct] {name}: {len(m):,} splats ({dt:.0f}s)")
    return res


def label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(255, 255, 255))
    d.text((4, 2), text, fill=(0, 0, 0))
    return np.asarray(im)


def stage_eval(args, out: Path, models: dict, rig) -> dict:
    import imageio.v2 as iio

    from gs4d.render import to_uint8
    from gs4d.torch_raster import render_model

    test = load_rendered_dataset(out / "dataset" / "test")
    gt = [im / 255.0 for im in test.images]
    t = time.time()
    metrics = evaluate(models, test.cameras, gt, test.masks)
    print(f"[5 eval] {len(test.cameras)} held-out views ({time.time() - t:.0f}s)")
    for k, v in metrics.items():
        print(f"  {k:16s} {v['gaussians']:>8,} splats  PSNR {v['psnr']:.2f} dB  SSIM {v['ssim']:.4f}  "
              f"silhouette IoU {v['silhouette_iou']:.3f}  alpha MAE {v['alpha_mae']:.4f}")

    views = [0, 5, len(test.cameras) - 2, len(test.cameras) - 1]
    rows, zooms = [], []
    for i in views:
        a = test.masks[i][..., None]
        g = to_uint8(gt[i] * a + (1 - a))
        cells = [label(g, "ground truth (point render)")]
        for name, m in models.items():
            cells.append(label(to_uint8(render_model(m, test.cameras[i])), f"{name} ({len(m) // 1000}k)"))
        rows.append(np.concatenate(cells, 1))
        if i >= len(test.cameras) - 2:        # close-up views: centre crop, 2x nearest
            h, w = g.shape[:2]
            ys, xs = slice(h // 3, h // 3 + h // 3), slice(w // 3, w // 3 + w // 3)
            zooms.append(np.concatenate([np.kron(c[ys, xs], np.ones((2, 2, 1), np.uint8)) for c in
                                         [g] + [to_uint8(render_model(m, test.cameras[i])) for m in models.values()]], 1))
    iio.imwrite(out / "compare_full.png", np.concatenate(rows, 0))
    iio.imwrite(out / "compare_zoom.png", np.concatenate(zooms, 0))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="point cloud (.ply/.las/.xyz/...) or a .zip containing one")
    ap.add_argument("--out", default="outputs/render_to_splat")
    ap.add_argument("--up", default="+z", help="auto | +z | -y | ...")
    ap.add_argument("--res", type=int, default=512, help="render / training resolution (square)")
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--point-size", type=float, default=0.025, help="point disc size in scan units")
    ap.add_argument("--test-views", type=int, default=16)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--max-gaussians", type=int, default=200_000, help="densification budget")
    ap.add_argument("--init-points", type=int, default=50_000, help="visual-hull initial splats")
    ap.add_argument("--sh-degree", type=int, default=0, help="0: the renders are view-independent")
    ap.add_argument("--init-scale", type=float, default=0.5, help="initial splat size x neighbour spacing")
    ap.add_argument("--res-schedule", default="0:4,1500:2,3500:1",
                    help="coarse-to-fine training, step:downsample pairs (CPU speed)")
    ap.add_argument("--reset-every", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--direct-full", type=int, default=500_000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--align-to", default="", help="3DGS whose frame the exports should match, e.g. "
                    "assets/oak/oak_3dgs_500k.ply (must be the direct fit of the same scan)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pc, up = stage_scan(args, out)
    rig = stage_render(args, out, pc, up)
    trained, log = stage_train(args, out)
    direct = stage_direct(args, out, pc, len(trained))
    models = {"render_to_splat": trained, "direct_equal": direct["direct_equal"][0],
              "direct_500k": direct["direct_500k"][0]}
    metrics = stage_eval(args, out, models, rig)
    summary = {"input": args.input, "train_views": len(rig.train), "test_views": len(rig.test),
               "resolution": args.res, "train_seconds": log.get("seconds"), "device": log.get("device"),
               "direct_seconds": {k: v[1] for k, v in direct.items()}, "metrics": metrics,
               "file_mb": {k: round((out / f"{k}.ply").stat().st_size / 1e6, 1) for k in models},
               "total_seconds": round(time.time() - t0, 1)}
    if args.align_to:
        ref = load_gaussians(args.align_to)
        d500 = models["direct_500k"]
        if len(ref) == len(d500):
            off = np.median(ref.means - d500.means, axis=0)
            if np.abs(ref.means - d500.means - off).max() < 1e-3:
                for k, m in models.items():
                    save_ply(m.transformed(translation=off), out / f"{k}_aligned.ply")
                summary["aligned_offset"] = off.tolist()
                print(f"exports aligned to {args.align_to} (offset {np.round(off, 3).tolist()})")
        if "aligned_offset" not in summary:
            print(f"warning: {args.align_to} is not the direct fit of this scan; exports not aligned")
    save_json(out / "metrics.json", summary)
    print(f"done in {time.time() - t0:.0f}s -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
