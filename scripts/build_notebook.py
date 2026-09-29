"""Generate notebooks/tree_wind_4dgs_demo.ipynb (Google Colab).

    python scripts/build_notebook.py

Colab "forms" (``#@param``) are plain assignments, so the notebook also runs
in Jupyter; ``GS4D_NOTEBOOK_TEST=1`` shrinks the workload for CI.
"""

from pathlib import Path

import nbformat as nbf

REPO = "saranunt-0/4DGS-experimental"
NB_PATH = "notebooks/tree_wind_4dgs_demo.ipynb"
COLAB_URL = f"https://colab.research.google.com/github/{REPO}/blob/main/{NB_PATH}"

cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str, hidden: bool = False):
    c = nbf.v4.new_code_cell(text.strip("\n"))
    if hidden:
        c.metadata["cellView"] = "form"
    cells.append(c)


# ---------------------------------------------------------------------------
md(f"""
# 🌳 Wind-blown tree in 3D Gaussian Splatting (4DGS demo)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook turns a **static, object-only 3D Gaussian Splatting (3DGS) tree** into a **4D animation with
swaying branches and fluttering leaves**, then exports it for **Blender, Houdini, USD tools, SuperSplat, …**

| step | what happens | runtime |
|---|---|---|
| 1 | install `gs4d` (this repo) | ~30 s |
| 2 | get a tree: procedural splat tree · your `.ply/.splat` · a URL · one you trained in the appendix | ~1 s |
| 3 | **object-only**: remove ground, background, sky splats & floaters | ~1 s |
| 4 | tree skeleton (joint hierarchy) + leaf mask | ~1 s |
| 5 | wind simulation: gusts travelling downwind, damped oscillator per branch joint, leaf flutter | ~5 s |
| 6 | preview video (CPU splatter, or gsplat on GPU) | 1–4 min on CPU |
| 7 | export: PLY sequence · animated USD (`ParticleField3DGaussianSplat`) · Blender importer | ~30 s |
| A | *(appendix, GPU)* train your **own** object-only tree from a phone video | 30–60 min |

**How the animation works** — a training-free, physics-inspired model (same family as game-engine tree wind,
Houdini 22's splat vegetation demos and the per-part oscillator prior studied in *Wind on Trees*, 2026):
each skeleton joint is a 2-DOF damped oscillator driven by the aerodynamic drag torque of everything it
carries; thick parts are stiff and slow, twigs are flexible and fast; forward kinematics poses every
Gaussian (position **and** rotation); a coherent noise field adds leaf flutter on top. Clips can loop
seamlessly.

> Everything except the appendix runs on a **CPU runtime**. A T4 GPU only speeds up the preview renders.
""")

code('''
#@title 1 · Setup  { display-mode: "form" }
#@markdown Clones the repo and installs the few extra dependencies (`usd-core` for USD export).
#@markdown Tick **USE_GPU_RENDERER** on a GPU runtime for fast previews — gsplat compiles its CUDA
#@markdown kernels on first use (~5–10 min once), afterwards previews render in real time.
USE_GPU_RENDERER = False  #@param {type:"boolean"}
REPO_URL = "https://github.com/saranunt-0/4DGS-experimental"  #@param {type:"string"}

import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
TEST_MODE = os.environ.get("GS4D_NOTEBOOK_TEST") == "1"   # tiny workload for automated tests

def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)

if IN_COLAB:
    ROOT = "/content/4DGS-experimental"
    if not os.path.exists(ROOT):
        sh(f"git clone -q --depth 1 {REPO_URL} {ROOT}")
    sh('pip -q install "usd-core>=26.3" imageio-ffmpeg')
    if USE_GPU_RENDERER:
        sh("pip -q install gsplat")
else:  # local Jupyter inside the repo checkout
    ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import gs4d
from gs4d import *
from gs4d.render import framing, gsplat_available
from gs4d.skeleton import auto_leafness
from gs4d.viz import show_grid, render_frames, write_mp4, html_video, plot_skeleton, plot_wind
from IPython.display import display

BACKEND = "gsplat" if (USE_GPU_RENDERER and gsplat_available()) else "cpu"
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
print(f"gs4d {gs4d.__version__} | preview renderer: {BACKEND}")
''', hidden=True)

md("""
## 2 · Get a tree

* **procedural** — a broad-leaf tree grown directly out of Gaussians (bark splats + flat leaf splats), object-only
  by construction and with its exact branch skeleton. Tick `SIMULATE_CAPTURE_ARTIFACTS` to wrap it in the junk
  a real capture contains (ground, a bush, a shell of sky splats, floaters) so step 3 has something to clean.
* **upload / url / google drive** — any 3DGS `.ply` (Postshot, Polycam, Luma, KIRI, Scaniverse, gsplat,
  nerfstudio, Brush, SuperSplat *uncompressed*) or `.splat`, **or a scanned point cloud** (`.ply` without splat
  attributes, `.xyz/.txt/.pts/.las/.laz`), or a `.zip` containing one. Point clouds have no photos to optimise
  against, so each local cluster of points is *fitted* with a Gaussian (flat surfels on bark and leaves, elongated
  along twigs; dense scans are merged per voxel). The tree is isolated from the ground *before* fitting, so the
  whole `MAX_GAUSSIANS` budget goes to the tree. For **google drive**, set `DRIVE_PATH` to the file inside
  `/content/drive/MyDrive/…` (Colab asks for permission to mount your Drive).
* **trained_from_video** — the model produced by the appendix (`work/trained_object.ply`).

`UP_AXIS = auto` uses the ground plane when there is one, otherwise the tree's shape (thin trunk below a wide
crown) and colours (bark below, leaves above). If the previews show the tree sideways, set it by hand.
""")

code('''
#@title 2 · Choose the tree  { display-mode: "form" }
SOURCE = "procedural"  #@param ["procedural", "upload (splat / point cloud / zip)", "url", "google drive", "trained_from_video"]
URL = ""  #@param {type:"string"}
DRIVE_PATH = "/content/drive/MyDrive/200-year-old-oak-tree.zip"  #@param {type:"string"}
#@markdown **Point clouds** are fitted with at most this many Gaussians
MAX_GAUSSIANS = 600000  #@param {type:"integer"}
#@markdown **Procedural tree**
TREE_SEED = 21  #@param {type:"integer"}
LEAF_DENSITY = 1.0  #@param {type:"slider", min:0.3, max:2.0, step:0.1}
AUTUMN = 0.0  #@param {type:"slider", min:0.0, max:1.0, step:0.1}
SIMULATE_CAPTURE_ARTIFACTS = True  #@param {type:"boolean"}
#@markdown **Orientation of the file**
UP_AXIS = "auto"  #@param ["auto", "+z", "-z", "+y", "-y", "+x", "-x"]

if TEST_MODE and os.environ.get("GS4D_TEST_SCAN"):   # automated test of the scan path
    SOURCE, DRIVE_PATH, MAX_GAUSSIANS = "google drive", os.environ["GS4D_TEST_SCAN"], 20000
skeleton, truth, pre_isolated, up_scan = None, None, False, None
if SOURCE == "procedural":
    base = TreeParams()
    lpb = tuple(max(2, int(round(n * LEAF_DENSITY))) for n in base.leaves_per_branch)
    tree, skeleton = generate_tree(TreeParams(seed=TREE_SEED, leaves_per_branch=lpb, autumn=AUTUMN))
    if SIMULATE_CAPTURE_ARTIFACTS:
        from gs4d.synthetic import make_captured_scene
        scene, truth = make_captured_scene(tree, seed=TREE_SEED)
    else:
        scene = tree
else:
    from gs4d.pointcloud import extract_from_zip, is_point_cloud_file, load_scan
    if SOURCE.startswith("upload"):
        from google.colab import files
        path = list(files.upload())[0]
    elif SOURCE == "url":
        import urllib.request
        path = os.path.join(OUT, os.path.basename(URL.split("?")[0]) or "download.ply")
        urllib.request.urlretrieve(URL, path)
    elif SOURCE == "google drive":
        if IN_COLAB and not os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        path = DRIVE_PATH
    else:
        path = os.path.join(ROOT, "work", "trained_object.ply")
    if str(path).lower().endswith(".zip"):
        path = extract_from_zip(path, os.path.join(OUT, "scan"))
    if is_point_cloud_file(path):
        # isolate the tree on a coarse fit, then fit only the tree's points with the full budget
        scene, up_scan, _ = load_scan(path, max_gaussians=MAX_GAUSSIANS, isolate=True,
                                      up="auto" if UP_AXIS == "auto" else UP_AXIS)
        pre_isolated = True
    else:
        scene = load_gaussians(path)
print(scene.summary())

if UP_AXIS != "auto":
    up = parse_up(UP_AXIS); how = "set by hand"
elif SOURCE == "procedural":
    up = parse_up("+z"); how = "procedural trees are +Z up"
elif up_scan is not None:
    up, how = up_scan, "ground plane of the scan"
else:
    up, how = estimate_up(scene)
print("up vector:", np.round(up, 3), "-", how)

center, dist = framing(tree if SOURCE == "procedural" else scene)
cams = orbit_cameras(center, dist * (1.6 if truth is not None else 1.0), up=up, n=3, elevation_deg=12,
                     width=320, height=320)
show_grid([render(scene, c, backend=BACKEND) for c in cams], ["view 1", "view 2", "view 3"])
''', hidden=True)

md("""
## 3 · Object-only: remove ground, background & floaters

Real captures reconstruct *everything* the cameras saw. Heuristic pipeline (each step is a boolean mask, see
`gs4d/cleanup.py`): opacity prune → giant/needle splat prune → **RANSAC ground plane** (a plane with almost
nothing below it) → **vertical crop cylinder** around the tallest dense column → statistical outlier removal →
**largest connected component**. If you trained with masks (appendix) most of this is already gone —
masked training with a random background is the most effective fix; `mask_vote()` lifts 2D masks to 3D.
""")

code('''
#@title 3 · Isolate the tree  { display-mode: "form" }
RUN_ISOLATION = True  #@param {type:"boolean"}
MIN_OPACITY = 0.05  #@param {type:"slider", min:0.0, max:0.5, step:0.01}
REMOVE_GROUND = True  #@param {type:"boolean"}
#@markdown Crop-cylinder radius in model units (0 = automatic)
CROP_RADIUS = 0.0  #@param {type:"number"}

if pre_isolated:
    print("point cloud: the tree was already isolated before fitting (step 2)")
    keep, tree_only = np.ones(len(scene), bool), scene
elif RUN_ISOLATION:
    keep, report = isolate_object(scene, up=up, min_opacity=MIN_OPACITY, remove_ground=REMOVE_GROUND,
                                  radius=CROP_RADIUS or None)
    tree_only = scene.subset(keep)
    if truth is not None:
        print(f"vs. ground truth: kept {np.mean(keep[truth]):.1%} of the tree, "
              f"{np.mean(keep[~truth]):.2%} of the background splats")
else:
    keep, tree_only = np.ones(len(scene), bool), scene

if skeleton is not None:
    skeleton_kept = subset_skeleton(skeleton, keep)   # binding follows the kept splats
far = orbit_cameras(center, dist * 1.6, up=up, n=1, elevation_deg=12, width=360, height=360)[0]
show_grid([render(scene, far, backend=BACKEND), render(tree_only, far, backend=BACKEND)],
          ["captured scene", "object-only tree"])
save_ply(tree_only, os.path.join(OUT, "tree_object_only.ply"))
print("saved", os.path.join(OUT, "tree_object_only.ply"))
''', hidden=True)

md("""
## 4 · Skeleton + leaf mask

The wind model needs a **joint hierarchy** and a per-splat **leafness** (0 = wood, 1 = leaf).

* Procedural trees carry their exact skeleton.
* For captured trees (or with `FORCE_EXTRACT_SKELETON`) it is recovered from the splats by
  **level-set skeletonisation**: k-NN graph on voxelised centres → geodesic distance from the trunk base →
  slices split into connected components = joints → parents follow the shortest paths.
* Leafness combines colour (green hue vs. bark brown/grey) with structure (thin, peripheral joints).
  For autumn foliage move `LEAF_HUE_DEG` to ~35.
""")

code('''
#@title 4 · Build the skeleton  { display-mode: "form" }
FORCE_EXTRACT_SKELETON = False  #@param {type:"boolean"}
SLICES = 36  #@param {type:"slider", min:12, max:80, step:2}
LEAF_HUE_DEG = 95  #@param {type:"slider", min:0, max:180, step:5}

if skeleton is None or FORCE_EXTRACT_SKELETON:
    skel = extract_skeleton(tree_only, up=up, n_slices=SLICES)
    if LEAF_HUE_DEG != 95:
        skel.leafness = auto_leafness(tree_only, skel.parents, skel.bind, hue_center_deg=LEAF_HUE_DEG)
else:
    skel = skeleton_kept
print(skel.summary())

cf, df = framing(tree_only)
cam_side = orbit_cameras(cf, df, up=up, n=1, elevation_deg=8, start_deg=-90, width=400, height=400)[0]
leaf_rgb = skel.leafness[:, None] * [0.25, 0.8, 0.2] + (1 - skel.leafness[:, None]) * [0.15, 0.3, 0.95]
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_skeleton(render(tree_only, cam_side, backend=BACKEND), skel, cam_side, f"{skel.num_joints} joints", ax=axes[0])
axes[1].imshow(render(tree_only, cam_side, colors=leaf_rgb, backend=BACKEND)); axes[1].axis("off")
axes[1].set_title("leafness (green = leaf, blue = wood)")
plt.tight_layout(); plt.show()
''', hidden=True)

md("""
## 5 · Wind

`WIND_SPEED` is in m/s (2 breeze · 5 moderate · 10+ strong). If your model is not in metres set
`TREE_HEIGHT_M` so gust sizes and speeds are physically scaled. `FLEXIBILITY` scales all bending,
`TRUNK_STIFFNESS` only the trunk. `LEAF_FLUTTER_*` controls the fast shimmer of the leaves.
`LOOP` makes the clip seamless (all wind frequencies become harmonics of the clip length).
""")

code('''
#@title 5 · Simulate the wind  { display-mode: "form" }
WIND_SPEED = 6.0  #@param {type:"slider", min:0.0, max:20.0, step:0.5}
WIND_DIRECTION_DEG = 0  #@param {type:"slider", min:0, max:360, step:15}
GUSTINESS = 0.6  #@param {type:"slider", min:0.0, max:1.5, step:0.05}
TURBULENCE = 0.35  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
FLEXIBILITY = 1.0  #@param {type:"slider", min:0.0, max:3.0, step:0.1}
TRUNK_STIFFNESS = 1.0  #@param {type:"slider", min:0.2, max:5.0, step:0.1}
LEAF_FLUTTER_DEG = 25  #@param {type:"slider", min:0, max:60, step:1}
LEAF_FLUTTER_HZ = 4.5  #@param {type:"slider", min:1.0, max:10.0, step:0.5}
TREE_HEIGHT_M = 0.0  #@param {type:"number"}
DURATION_S = 4.0  #@param {type:"number"}
FPS = 24  #@param {type:"integer"}
LOOP = True  #@param {type:"boolean"}
if TEST_MODE:
    DURATION_S, FPS = 1.0, 6

params = WindParams(
    speed=WIND_SPEED, direction_deg=WIND_DIRECTION_DEG, gustiness=GUSTINESS, turbulence=TURBULENCE,
    flexibility=FLEXIBILITY, trunk_stiffness=TRUNK_STIFFNESS, leaf_flutter_deg=LEAF_FLUTTER_DEG,
    leaf_flutter_hz=LEAF_FLUTTER_HZ, tree_height_m=TREE_HEIGHT_M or None,
    duration=DURATION_S, fps=FPS, loop=LOOP)
anim = WindAnimation(tree_only, skel, params)
print(anim.summary())
plot_wind(anim)
''', hidden=True)

md("""
## 6 · Preview

The camera looks across the wind (wind blows left → right for direction 0°). The CPU splatter does exact
front-to-back alpha compositing; enable the GPU renderer in step 1 for speed.
""")

code('''
#@title 6 · Render a preview video  { display-mode: "form" }
CAMERA = "front"  #@param ["front", "slow orbit", "canopy close-up"]
PREVIEW_SIZE = 400  #@param {type:"slider", min:200, max:1080, step:40}
BACKGROUND = "white"  #@param ["white", "sky", "black"]
#@markdown Render every Nth frame (2-3 keeps big scans quick on a CPU runtime)
PREVIEW_EVERY = 1  #@param {type:"slider", min:1, max:4, step:1}
bg = {"white": (1, 1, 1), "sky": (0.78, 0.86, 0.96), "black": (0, 0, 0)}[BACKGROUND]
if TEST_MODE:
    PREVIEW_SIZE = 128

cf, df = framing(tree_only)
from gs4d.skeleton import horizontal_basis
e1, e2 = horizontal_basis(up)
wd = np.deg2rad(WIND_DIRECTION_DEG)
wind_dir = np.cos(wd) * e1 + np.sin(wd) * e2
side = np.cross(up, wind_dir)              # look across the wind
if CAMERA == "front":
    cams = [look_at(cf - side * df + up * 0.08 * df, cf, up, PREVIEW_SIZE, PREVIEW_SIZE)] * len(anim)
elif CAMERA == "slow orbit":
    start = np.rad2deg(np.arctan2(-side @ e2, -side @ e1))
    cams = orbit_cameras(cf, df, up=up, n=len(anim), elevation_deg=8, start_deg=start, sweep_deg=60,
                         width=PREVIEW_SIZE, height=PREVIEW_SIZE)
else:
    leafy = skel.leafness > 0.5
    crown = tree_only.means[leafy].mean(0) if leafy.any() else cf
    cams = [look_at(crown - side * df * 0.45, crown, up, PREVIEW_SIZE, PREVIEW_SIZE)] * len(anim)

import time
t0, frames = time.time(), []
for i in range(0, len(anim), PREVIEW_EVERY):
    frames.append(render_frames([anim[i]], cams[i], background=bg, backend=BACKEND, progress=False)[0])
    if i % max(1, len(anim) // 6) == 0:
        print(f"  frame {i + 1}/{len(anim)}  ({time.time() - t0:.0f}s)")
mp4 = write_mp4(frames, os.path.join(OUT, "preview.mp4"), fps=FPS / PREVIEW_EVERY, loops=2)
print("saved", mp4)
display(html_video(mp4, width=min(PREVIEW_SIZE, 640)))
''', hidden=True)

md("""
## 7 · Export for Blender / Houdini / USD / viewers

* **PLY sequence** `ply_sequence/<name>_0001.ply …` — the de-facto 4DGS interchange (one reference-layout 3DGS
  PLY per frame). Houdini (native splats / GSOPs, `$F4`), Blender (script below or KIRI 3DGS Render), SuperSplat,
  Postshot, Brush, Unity.
* **USD** `<name>.usdc` (off by default, tick `USD`) — OpenUSD `ParticleField3DGaussianSplat` (OpenUSD ≥ 26.03) with time-sampled positions and
  orientations; appearance stored once (2.5× smaller than the PLY sequence at SH 0, ~9× at SH 3). Houdini Solaris / Karma, Omniverse,
  Nuke 17, usdview.
* **Blender importer** `gs4d_import_sequence.py` — native Geometry Nodes (no add-on, render-farm safe).
* `REORIENT` rotates the export so the tree stands up in the target tool (Blender = Z-up, Houdini/Maya = Y-up).
* `SH_DEGREE = 0` gives the smallest files (17 floats per splat per frame).
""")

code('''
#@title 7 · Export  { display-mode: "form" }
EXPORT_NAME = "tree_wind"  #@param {type:"string"}
PLY_SEQUENCE = True  #@param {type:"boolean"}
USD = False  #@param {type:"boolean"}
NPZ = False  #@param {type:"boolean"}
BLENDER_SCRIPT = True  #@param {type:"boolean"}
REORIENT = "Z-up (Blender, Unreal)"  #@param ["keep file axes", "Z-up (Blender, Unreal)", "Y-up (Houdini, Maya, USD default)"]
SH_DEGREE = "keep"  #@param ["keep", "0", "1", "2", "3"]
ZIP_AND_DOWNLOAD = True  #@param {type:"boolean"}
COPY_TO_GOOGLE_DRIVE = False  #@param {type:"boolean"}

import shutil
from gs4d.export import zip_dir
formats = [f for f, on in [("ply_sequence", PLY_SEQUENCE), ("usd", USD), ("npz", NPZ), ("blender", BLENDER_SCRIPT)] if on]
frames_out, usd_up = anim, "Z"
if REORIENT.startswith("Z"):
    frames_out = ReorientedFrames(anim, up_rotation(up, "Z"))
elif REORIENT.startswith("Y"):
    frames_out, usd_up = ReorientedFrames(anim, up_rotation(up, "Y")), "Y"
else:
    usd_up = "Y" if abs(up[1]) > abs(up[2]) else "Z"
export_dir = os.path.join(OUT, EXPORT_NAME)
shutil.rmtree(export_dir, ignore_errors=True)
result = export_all(frames_out, export_dir, EXPORT_NAME, formats=formats,
                    sh_degree=None if SH_DEGREE == "keep" else int(SH_DEGREE), up_axis=usd_up, fps=FPS)
if os.path.exists(os.path.join(OUT, "preview.mp4")):
    shutil.copy(os.path.join(OUT, "preview.mp4"), export_dir)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(export_dir) for f in fs)
print(f"\\nexported to {export_dir} ({total / 1e6:.1f} MB)")
for k, v in result.items():
    print(f"  {k:15s} {v if not isinstance(v, list) else f'{len(v)} files, e.g. {os.path.basename(v[0])}'}")

if ZIP_AND_DOWNLOAD:
    zpath = zip_dir(export_dir, os.path.join(OUT, EXPORT_NAME + ".zip"))
    print("zip:", zpath, f"({os.path.getsize(zpath) / 1e6:.1f} MB)")
    if os.path.getsize(zpath) > 200e6:
        print("tip: large export - COPY_TO_GOOGLE_DRIVE is faster than a browser download, "
              "or use SH_DEGREE=0 / a shorter clip / USD only")
    if IN_COLAB:
        from google.colab import files
        files.download(str(zpath))
if COPY_TO_GOOGLE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    dst = f"/content/drive/MyDrive/gs4d/{EXPORT_NAME}"
    shutil.copytree(export_dir, dst, dirs_exist_ok=True)
    print("copied to", dst)
''', hidden=True)

md("""
## 8 · Open the result in your 3D software

**Blender 4.5 LTS / 5.x** (tested headless with Blender 5.0)
1. Unzip. Open Blender → *Scripting* workspace → *Open* `gs4d_import_sequence.py` → **Run Script**
   (it finds `ply_sequence/` next to itself and sets the timeline).
2. Press play. Render with Cycles or EEVEE. Each splat is an oriented ellipsoid with a Gaussian-falloff
   emissive material driven by native Geometry Nodes (*Import PLY* per frame), so it also works in
   background renders: `blender -b -P gs4d_import_sequence.py -- --dir ply_sequence --render //frames/f_#### --camera`
3. For hero-quality splat rendering and editing use the free **KIRI Engine 3DGS Render** add-on (v5 adds rigging
   and a 4DGS PLY-sequence mode) on the same PLY files.

**Houdini** — H22 has native Gaussian splats (import, deform, simulate, Karma). Load a frame per timestep with a
File SOP path `…/ply_sequence/tree_wind_$F4.ply` (or the **GSOPs** *Gaussian Splats Import* node on H20.5/21), or
reference `tree_wind.usdc` in **Solaris**. The rest pose (`tree_wind_rest.ply`) is handy for re-simulating with
Vellum / KineFX in Houdini itself.

**USD tools** (Omniverse, Nuke 17, usdview ≥ 26.03) — open `tree_wind.usdc`.

**Viewers** — drag the PLY sequence into **SuperSplat**, **Postshot** or **Brush**; Unity via GSOPs' Unity
sequence player.
""")

# ---------------------------------------------------------------------------
md("""
---
# Appendix · Train your own object-only tree from a phone video (GPU)

**Capture tips (they matter more than any algorithm):** walk 1–2 full circles around the tree at two heights,
keep the whole tree in frame, overcast light, no wind (leaves must hold still!), 1–2 min of 4K/1080p video,
lock exposure. Avoid strong backlight through the crown.

Pipeline: sharpest frames → **COLMAP** (pycolmap) → **object mattes** (rembg / BiRefNet, good at leaves & thin
branches; SAM 2/3 for crowded scenes) → **masked 3DGS training with a random background + alpha loss** (gsplat) →
**mask voting + isolation** → `work/trained_object.ply`. Then go back to step 2 with
`SOURCE = "trained_from_video"`.

Runtime on a T4: frames ~1 min · COLMAP 5–15 min (CPU) · masks ~2 min · gsplat compile ~5–10 min (once) ·
training 7k steps ~10–20 min.
""")

code('''
#@title A1 · Install training extras + upload a video  { display-mode: "form" }
RUN_TRAINING = False  #@param {type:"boolean"}
NUM_FRAMES = 150  #@param {type:"slider", min:40, max:300, step:10}
VIDEO_PATH = ""  #@param {type:"string"}
#@markdown Leave `VIDEO_PATH` empty to upload a file.
WORK = os.path.join(ROOT, "work")
import shutil
if RUN_TRAINING:
    sh('pip -q install pycolmap "rembg[gpu]" gsplat')
    os.makedirs(WORK, exist_ok=True)
    if not VIDEO_PATH:
        from google.colab import files
        VIDEO_PATH = os.path.join(WORK, list(files.upload())[0])
        # files.upload() saves into the cwd
        if not os.path.exists(VIDEO_PATH):
            shutil.move(os.path.basename(VIDEO_PATH), VIDEO_PATH)
    from gs4d.train import extract_frames
    extract_frames(VIDEO_PATH, os.path.join(WORK, "frames"), num_frames=NUM_FRAMES, max_size=1600)
else:
    print("RUN_TRAINING is off - skipping the appendix")
''', hidden=True)

code('''
#@title A2 · Camera poses with COLMAP  { display-mode: "form" }
MATCHER = "sequential"  #@param ["sequential", "exhaustive"]
if RUN_TRAINING:
    from gs4d.train import run_colmap
    UNDIST = run_colmap(os.path.join(WORK, "frames"), os.path.join(WORK, "colmap"), matcher=MATCHER)
    print("undistorted scene:", UNDIST)
''', hidden=True)

code('''
#@title A3 · Object masks  { display-mode: "form" }
MASK_MODEL = "birefnet-general"  #@param ["birefnet-general", "isnet-general-use", "u2net"]
if RUN_TRAINING:
    from gs4d.train import compute_masks
    from PIL import Image
    MASKS = compute_masks(os.path.join(UNDIST, "images"), os.path.join(WORK, "masks"), model=MASK_MODEL)
    names = sorted(os.listdir(os.path.join(UNDIST, "images")))[:: max(1, NUM_FRAMES // 4)][:4]
    ims = [np.asarray(Image.open(os.path.join(UNDIST, "images", n)).convert("RGB")) / 255.0 for n in names]
    ms = [np.asarray(Image.open(os.path.join(MASKS, os.path.splitext(n)[0] + ".png")).resize(
        (im.shape[1], im.shape[0]))) / 255.0 for n, im in zip(names, ims)]
    show_grid([im * (0.25 + 0.75 * m[..., None]) for im, m in zip(ims, ms)], names, size=3)
''', hidden=True)

code('''
#@title A4 · Masked 3DGS training + clean-up  { display-mode: "form" }
ITERS = 7000  #@param {type:"slider", min:2000, max:30000, step:1000}
MAX_WIDTH = 1000  #@param {type:"slider", min:480, max:1600, step:40}
ALPHA_LOSS = 0.3  #@param {type:"slider", min:0.0, max:1.0, step:0.05}
if RUN_TRAINING:
    from gs4d.train import load_dataset, train_object_splat, TrainConfig
    data = load_dataset(UNDIST, MASKS, max_width=MAX_WIDTH)
    trained = train_object_splat(data, TrainConfig(iters=ITERS, alpha_lambda=ALPHA_LOSS))
    vote = mask_vote(trained, data.cameras, data.masks, threshold=0.6)
    up_trained, how = estimate_up(trained.subset(vote))
    keep_t, _ = isolate_object(trained, up=up_trained, initial_keep=vote, remove_ground=True)
    obj = trained.subset(keep_t)
    save_ply(obj, os.path.join(WORK, "trained_object.ply"))
    print(obj.summary())
    c_t, d_t = framing(obj)
    show_grid([render(obj, c, backend="gsplat" if gsplat_available() else "cpu")
               for c in orbit_cameras(c_t, d_t, up=up_trained, n=3, width=360, height=360)])
    print("Now go back to step 2 and choose SOURCE = trained_from_video")
''', hidden=True)

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
Path(NB_PATH).parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH, f"({len(cells)} cells)")
