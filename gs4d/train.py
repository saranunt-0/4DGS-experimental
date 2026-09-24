"""Object-only 3D Gaussian Splatting from a phone video (GPU + gsplat).

Pipeline::

    frames  = extract_frames("tree.mp4", "work/frames", num_frames=150)
    undist  = run_colmap("work/frames", "work/colmap")           # pycolmap SfM + undistortion
    compute_masks(undist / "images", "work/masks")               # rembg / BiRefNet object mattes
    data    = load_dataset(undist, "work/masks", max_width=1000)
    model   = train_object_splat(data, TrainConfig(iters=7000))  # masked, random-background training

Why this gives an *object-only* model: every step composites the ground
truth over a **random background colour** using the object mask and renders
with the same colour, and an **alpha loss** pushes accumulated opacity to the
mask.  Gaussians that try to explain the background are penalised (their
colour can never match a background that changes every step), so the
optimiser deletes or never creates them.  Remaining specks are removed by
:func:`gs4d.cleanup.mask_vote` / :func:`gs4d.cleanup.isolate_object`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .colmap import read_colmap
from .gaussians import GaussianModel, logit, rgb_to_sh0
from .render import Camera

IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# ------------------------------------------------------------------ frames
def _sharpness(gray: np.ndarray) -> float:
    lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(lap.var())


def extract_frames(video_path: str | Path, out_dir: str | Path, num_frames: int = 150, max_size: int = 1600,
                   quality: int = 95) -> list[Path]:
    """Pick the sharpest frame in each of ``num_frames`` equal time windows."""
    import imageio.v2 as iio
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = iio.get_reader(str(video_path), "ffmpeg")
    try:
        total = reader.count_frames()
    except Exception:
        meta = reader.get_meta_data()
        total = int(meta.get("duration", 0) * meta.get("fps", 30))
    total = max(total, 1)
    num_frames = min(num_frames, total)
    window = total / num_frames
    best: dict[int, tuple[float, np.ndarray]] = {}
    for idx, frame in enumerate(reader):
        w = min(int(idx / window), num_frames - 1)
        small = frame[::4, ::4].astype(np.float32).mean(-1)
        s = _sharpness(small)
        if w not in best or s > best[w][0]:
            best[w] = (s, frame)
    reader.close()
    paths = []
    for w in sorted(best):
        img = Image.fromarray(best[w][1])
        scale = max_size / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        p = out_dir / f"frame_{w:04d}.jpg"
        img.save(p, quality=quality)
        paths.append(p)
    print(f"extracted {len(paths)} frames -> {out_dir}")
    return paths


# ------------------------------------------------------------------ COLMAP
def run_colmap(image_dir: str | Path, work_dir: str | Path, matcher: str = "sequential") -> Path:
    """Structure-from-motion with pycolmap, then undistort to PINHOLE images.

    Returns the undistorted folder containing ``images/`` and ``sparse/``.
    ``matcher='sequential'`` suits videos; use ``'exhaustive'`` for photo sets.
    """
    import pycolmap

    image_dir = Path(image_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    t = time.time()
    try:
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE)
    except TypeError:  # very old pycolmap
        pycolmap.extract_features(db, image_dir)
    print(f"  features: {time.time() - t:.0f}s")
    t = time.time()
    if matcher == "sequential":
        pycolmap.match_sequential(db)
    else:
        pycolmap.match_exhaustive(db)
    print(f"  matching: {time.time() - t:.0f}s")
    t = time.time()
    recs = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not recs:
        raise RuntimeError("COLMAP could not register the images (too little overlap / texture?)")
    best = max(recs, key=lambda k: recs[k].num_reg_images())
    n_img = len(list(p for p in image_dir.iterdir() if p.suffix in IMG_EXT))
    print(f"  mapping: {time.time() - t:.0f}s, registered {recs[best].num_reg_images()}/{n_img} images")
    out = work / "undistorted"
    pycolmap.undistort_images(out, sparse / str(best), image_dir)
    return out


# ------------------------------------------------------------------ masks
def compute_masks(image_dir: str | Path, mask_dir: str | Path, model: str = "birefnet-general",
                  keep_center: bool = True) -> Path:
    """Soft object mattes with rembg (BiRefNet handles thin branches / leaves well).

    ``keep_center`` drops blobs that do not touch the central third of the
    image (other trees, people...).  Masks are saved as 8-bit PNGs named like
    the images.  Replace with SAM 2 / SAM 3 masks for difficult scenes.
    """
    from PIL import Image
    from rembg import new_session, remove
    from scipy import ndimage

    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(model)
    files = sorted(p for p in image_dir.iterdir() if p.suffix in IMG_EXT)
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB")
        m = np.asarray(remove(img, session=session, only_mask=True), dtype=np.float32) / 255.0
        if keep_center:
            lab, n = ndimage.label(ndimage.binary_dilation(m > 0.3, iterations=15))
            if n > 1:
                h, w = m.shape
                center = lab[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
                keep_ids = np.unique(center[center > 0])
                if len(keep_ids):
                    m = m * np.isin(lab, keep_ids)
        Image.fromarray((m * 255).astype(np.uint8)).save(mask_dir / (p.stem + ".png"))
        if i % 20 == 0:
            print(f"  mask {i + 1}/{len(files)}")
    return mask_dir


# ------------------------------------------------------------------ dataset
@dataclass
class TrainData:
    images: list            # uint8 HxWx3
    masks: list             # float32 HxW in [0, 1] (all ones if no masks)
    cameras: list           # list[Camera] at image resolution
    points: np.ndarray      # (P, 3) SfM points
    colors: np.ndarray      # (P, 3)
    names: list = field(default_factory=list)


def load_dataset(scene_dir: str | Path, mask_dir: str | Path | None = None, max_width: int = 1000) -> TrainData:
    """Load an undistorted COLMAP scene (``images/`` + ``sparse/``) and optional masks."""
    from PIL import Image

    scene_dir = Path(scene_dir)
    sparse = scene_dir / "sparse"
    if (sparse / "0").exists() and not (sparse / "cameras.bin").exists():
        sparse = sparse / "0"
    sc = read_colmap(sparse)
    images, masks, cams = [], [], []
    for cam, name in zip(sc.cameras, sc.image_names):
        img = Image.open(scene_dir / "images" / name).convert("RGB")
        s = min(1.0, max_width / img.width)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        sx, sy = img.width / cam.width, img.height / cam.height
        cams.append(Camera(cam.viewmat, cam.fx * sx, cam.fy * sy, cam.cx * sx, cam.cy * sy, img.width, img.height))
        images.append(np.asarray(img))
        m = np.ones((img.height, img.width), np.float32)
        if mask_dir is not None:
            mp = Path(mask_dir) / (Path(name).stem + ".png")
            if mp.exists():
                m = np.asarray(Image.open(mp).convert("L").resize(img.size, Image.BILINEAR), np.float32) / 255.0
        masks.append(m)
    print(f"loaded {len(images)} views at {images[0].shape[1]}x{images[0].shape[0]}, {len(sc.points)} SfM points")
    return TrainData(images, masks, cams, sc.points, sc.colors, sc.image_names)


# ------------------------------------------------------------------ training
@dataclass
class TrainConfig:
    iters: int = 7000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    alpha_lambda: float = 0.3        # mask (alpha) loss weight; 0 disables
    random_background: bool = True   # the key trick for object-only splats
    init_opacity: float = 0.1
    init_scale: float = 1.0
    object_init: bool = True         # keep only SfM points that project inside the masks
    refine_stop_frac: float = 0.7
    reset_every: int = 3000
    log_every: int = 500
    seed: int = 0
    device: str = "cuda"


def _ssim(img1, img2, window: int = 11, sigma: float = 1.5):
    import torch
    import torch.nn.functional as F

    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(window, device=x.device) - window // 2) ** 2) / (2 * sigma**2))
    g = (g / g.sum()).float()
    k = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)
    mu1 = F.conv2d(x, k, padding=window // 2, groups=3)
    mu2 = F.conv2d(y, k, padding=window // 2, groups=3)
    s11 = F.conv2d(x * x, k, padding=window // 2, groups=3) - mu1**2
    s22 = F.conv2d(y * y, k, padding=window // 2, groups=3) - mu2**2
    s12 = F.conv2d(x * y, k, padding=window // 2, groups=3) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return m.mean()


def _probe(pts: np.ndarray, cols: np.ndarray | None = None) -> GaussianModel:
    n = len(pts)
    cols = np.full((n, 3), 0.5) if cols is None else np.clip(cols, 0, 1)
    return GaussianModel.from_activated(pts, np.tile([1.0, 0, 0, 0], (n, 1)), np.full((n, 3), 1e-3), np.full(n, 0.5), cols)


def visual_hull_points(data: TrainData, n_samples: int = 300_000, threshold: float = 0.9, max_points: int = 40_000,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Carve random samples with the masks (points inside the object in ~all views).

    Robust object initialisation even when SfM put almost no points on the
    object (thin foliage often gives few stable features).  Colours are the
    mean pixel colour over the views that see the point.
    """
    from .cleanup import cameras_focus_point, mask_vote

    rng = np.random.default_rng(seed)
    c = cameras_focus_point(data.cameras)
    dist = np.median([np.linalg.norm(cam.position - c) for cam in data.cameras])
    half = 0.6 * dist
    pts = c + rng.uniform(-half, half, (n_samples, 3))
    keep = mask_vote(_probe(pts), data.cameras, data.masks, threshold=threshold, min_views=max(3, len(data.cameras) // 4))
    pts = pts[keep]
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    col = np.zeros((len(pts), 3))
    cnt = np.zeros(len(pts))
    for cam, img in zip(data.cameras, data.images):
        R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
        p = pts @ R.T + t
        z = np.maximum(p[:, 2], 1e-6)
        u = np.floor(cam.fx * p[:, 0] / z + cam.cx).astype(int)
        v = np.floor(cam.fy * p[:, 1] / z + cam.cy).astype(int)
        ok = (p[:, 2] > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        col[ok] += img[v[ok], u[ok]] / 255.0
        cnt[ok] += 1
    return pts, col / np.maximum(cnt, 1)[:, None]


def _init_points(data: TrainData, cfg: TrainConfig):
    from scipy.spatial import cKDTree

    pts, cols = data.points.astype(np.float64), data.colors.astype(np.float64)
    if cfg.object_init and any(m.min() < 0.99 for m in data.masks):
        from .cleanup import mask_vote

        keep = mask_vote(_probe(pts, cols), data.cameras, data.masks, threshold=0.5, min_views=2) if len(pts) else np.zeros(0, bool)
        print(f"object init: {keep.sum()}/{len(pts)} SfM points lie inside the masks")
        pts, cols = pts[keep], cols[keep]
        if len(pts) < 5000:
            hp, hc = visual_hull_points(data, seed=cfg.seed)
            print(f"object init: + {len(hp)} visual-hull samples")
            pts = np.concatenate([pts, hp])
            cols = np.concatenate([cols, hc])
        if len(pts) < 10:
            raise RuntimeError("no object points: check that the masks match the images")
    k = min(4, len(pts) - 1)
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    dist = np.sqrt(np.maximum((d[:, 1:] ** 2).mean(1), 1e-10))
    return pts, cols, dist


def train_object_splat(data: TrainData, cfg: TrainConfig | None = None, rasterize_fn=None) -> GaussianModel:
    """Masked 3DGS training with gsplat's DefaultStrategy densification."""
    import torch
    from gsplat.strategy import DefaultStrategy

    if rasterize_fn is None:
        from gsplat import rasterization as rasterize_fn

    cfg = cfg or TrainConfig()
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pts, cols, dist = _init_points(data, cfg)
    n = len(pts)
    centers = np.stack([c.position for c in data.cameras])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    k_rest = (cfg.sh_degree + 1) ** 2 - 1
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(pts, dtype=torch.float32)),
        "scales": torch.nn.Parameter(torch.tensor(np.log(dist * cfg.init_scale), dtype=torch.float32)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.rand(n, 4)),
        "opacities": torch.nn.Parameter(torch.full((n,), float(logit(cfg.init_opacity)))),
        "sh0": torch.nn.Parameter(torch.tensor(rgb_to_sh0(np.clip(cols, 0, 1)), dtype=torch.float32)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, k_rest, 3)),
    }).to(dev)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {
        k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()
    }
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / cfg.iters))
    strategy = DefaultStrategy(
        refine_stop_iter=int(cfg.iters * cfg.refine_stop_frac), reset_every=cfg.reset_every, verbose=False
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    imgs = [torch.from_numpy(im).to(dev) for im in data.images]  # uint8 on device
    masks = [torch.from_numpy(m).to(dev)[..., None] for m in data.masks]
    views = [torch.from_numpy(c.viewmat.astype(np.float32)).to(dev) for c in data.cameras]
    Ks = [torch.from_numpy(c.K.astype(np.float32)).to(dev) for c in data.cameras]
    order = rng.permutation(len(imgs))
    t0 = time.time()
    for step in range(cfg.iters):
        i = int(order[step % len(order)])
        if step % len(order) == len(order) - 1:
            order = rng.permutation(len(imgs))
        cam = data.cameras[i]
        gt = imgs[i].float() / 255.0
        m = masks[i]
        bg = torch.rand(3, device=dev) if cfg.random_background else torch.zeros(3, device=dev)
        gt_c = gt * m + bg * (1.0 - m)
        sh_deg = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        renders, alphas, info = rasterize_fn(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]), colors=colors,
            viewmats=views[i][None], Ks=Ks[i][None], width=cam.width, height=cam.height,
            sh_degree=sh_deg, packed=False, backgrounds=bg[None],
        )
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        img = renders[0]
        l1 = (img - gt_c).abs().mean()
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - _ssim(img, gt_c))
        if cfg.alpha_lambda > 0:
            loss = loss + cfg.alpha_lambda * (alphas[0] - m).abs().mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        # same order as gsplat's simple_trainer: densify / prune after the optimiser step
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.iters - 1):
            print(f"  step {step:5d}/{cfg.iters}  loss {loss.item():.4f}  gaussians {len(splats['means']):,}  "
                  f"{time.time() - t0:.0f}s")

    with torch.no_grad():
        return GaussianModel(
            splats["means"].cpu().numpy(),
            splats["quats"].cpu().numpy(),
            splats["scales"].cpu().numpy(),
            splats["opacities"].cpu().numpy(),
            splats["sh0"][:, 0].cpu().numpy(),
            splats["shN"].cpu().numpy(),
        )
