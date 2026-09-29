# Object-only Gaussian splats (no background artifacts)

**Goal:** a 3DGS model that contains *only the tree*, with no ground, sky shell, neighbouring objects or floaters,
so it can be animated, composited and exported cleanly.

## 1. Why background artifacts appear

3DGS explains **every pixel** of every training image. So:

* everything the camera saw gets reconstructed: ground, fences, other trees;
* the **sky** and anything far away become a shell of huge, soft Gaussians;
* regions seen from only a few views produce **floaters**: semi-transparent splats hovering in front of the
  cameras that are "correct" for one view and garbage from all others;
* thin structures (leaves, twigs) against a bright sky produce **halo** splats that mix foliage and sky colour;
* if the tree moved during capture (wind!), leaves get smeared into semi-transparent haze.

Merely *ignoring* background pixels in the loss (a common "mask" option) is **not enough**. Background Gaussians are
then unconstrained rather than penalised, so floaters survive. You need a loss that actively pushes the background
to be empty (see 3.3).

## 2. The strategy ladder (cheapest and most effective first)

| # | Stage | What to do | Effect |
|---|---|---|---|
| 1 | Capture | 1–2 full orbits at two heights, whole tree in frame, overcast light, **no wind**, locked exposure, 150–300 sharp frames | Fewer floaters and cleaner geometry. Nothing later compensates for a bad capture |
| 2 | 2D masks | Object mattes for every frame: **BiRefNet** via rembg (automatic, good on leaves and twigs), **SAM 2** (click once, propagate through the video), **SAM 3** (text prompt "tree", tracks through video) | Defines "object" in each view |
| 3 | **Masked training** | Composite the ground truth over a **random background colour** each step, render with the same colour, add an **alpha loss** `|α − mask|` | Background can't be explained by any Gaussian, so it's never created. **Largest single win** |
| 4 | 3D mask lifting | **Mask voting** (keep splats that project inside the mask in most views), FlashSplat (optimal closed-form 2D→3D label assignment, ECCV 2024), SAGA, Gaussian Grouping, GaussianCut, Trace3D | Removes what survived training |
| 5 | Geometric clean-up | Opacity prune → giant/needle prune → RANSAC **ground plane** → crop → statistical outlier removal → **connected component** | Works even with no cameras or masks (downloaded PLYs) |
| 6 | Interactive | SuperSplat selection and crop, GSOPs isolate and floater tools (Houdini), KIRI crop-box modifier (Blender) | Final polish |
| — | Generative | **SAM 3D Objects** (Meta, Nov 2025; single image, object splats + mesh), **TRELLIS.2** (Microsoft, 4B; image → 3DGS or mesh, ≥ 24 GB GPU), Hunyuan3D | Object-only by construction, but the unseen sides are hallucinated and trees lose fine leaf structure. Good for quick background assets |

## 3. What this repository provides

### 3.1 Procedural object-only tree (`gs4d.procedural_tree`)
A tree grown directly as Gaussians: bark splats tiled around branch segments, flat leaf splats, and baked AO / sun
gradient. It is clean by construction and comes with the exact skeleton that grew it.

### 3.2 Geometric isolation of any splat (`gs4d.cleanup.isolate_object`)
```python
from gs4d import load_gaussians, isolate_object, estimate_up
scene = load_gaussians("my_capture.ply")
up, how = estimate_up(scene)            # dominant ground plane normal (or tree shape/colour)
keep, report = isolate_object(scene, up=up)
tree = scene.subset(keep)
```
The steps, each of which returns a boolean *keep* mask you can inspect:

1. **Opacity** ≥ 0.05, and **giant / needle** splats removed (typical of sky and background).
2. **RANSAC ground plane.** It must be near horizontal, and a valid ground has *almost the whole scene above it*.
   This rule stops the fit from locking onto the bottom of the crown. The fitted normal is also a good gravity
   "up" estimate for tilted captures.
3. **Object column.** Histogram the above-ground mass (weighted by height) on the ground plane and take the peak
   column. This works even when background splats outnumber the tree.
4. **Seeded connectivity.** The column is only a seed: the object is the voxel-connected blob(s) touching it, inside
   a 2× safety cylinder. Sparse, leaning or lopsided crowns aren't cut off.
5. **Statistical outlier removal** (loose, `std_ratio=4`, because isolated leaves at the crown edge are real).

### 3.3 Masked, object-only training (`gs4d.train`, GPU)
```python
from gs4d.train import extract_frames, run_colmap, compute_masks, load_dataset, train_object_splat, TrainConfig
extract_frames("tree.mp4", "work/frames", num_frames=150)          # sharpest frame per time window
undist = run_colmap("work/frames", "work/colmap")                    # pycolmap SfM + undistortion
compute_masks(undist / "images", "work/masks")                       # BiRefNet mattes (rembg)
data = load_dataset(undist, "work/masks", max_width=1000)
model = train_object_splat(data, TrainConfig(iters=7000, alpha_lambda=0.3))
```
* **Random background + alpha loss** on every step (see 1).
* **Object-aware initialisation.** Keep the SfM points that fall inside the masks. When foliage produced few stable
  features (common: SfM points land on the textured ground), add **visual-hull samples**: random points that
  project inside the mask in ~all views. In our synthetic test, SfM put only 14 of 6,665 points on the tree, and
  the visual hull supplied a correctly shaped, correctly coloured initial tree.
* gsplat `DefaultStrategy` densification; SH degree ramps up to 3.
* Afterwards: `mask_vote` + `isolate_object(initial_keep=vote)` remove any remaining specks.

### 3.4 Measured on synthetic captures
`gs4d.synthetic.make_captured_scene` wraps a procedural tree in the junk a real capture contains: a 12k-splat
ground disc, a bush, a 3k-splat sky shell (partly below the horizon) and 2.5k floaters. The tree is 30–80% of
all splats. We evaluated 4 trees (one strongly leaning, one small and sparse) × 3 random scenes
(reproduce with `python scripts/benchmark_isolation.py`):

| Method | Tree splats kept | Background splats kept |
|---|---|---|
| `isolate_object` (no cameras / masks) | **99.9 %** (min 99.8 %) | 0.21 % (max 0.35 %) |
| `mask_vote`, 16 views with object masks | **100 %** | 0.46 % (max 0.82 %) |
| `mask_vote` + `isolate_object` | 99.9 % (min 99.8 %) | **0.12 %** (max 0.23 %) |
| clean tree only (must not be damaged) | 99.7–99.8 %, "no ground" correctly reported | – |
| capture tilted 15° / 30° off the declared up axis | 99.95 % | 0.25 %, ground normal recovered to < 0.1° |
| **point-cloud scan** (200k points, tree + ground), `isolate_point_cloud` | 99.4–99.6 % of tree *points* | 0.02–0.08 % of ground points |

Lessons from building this (each was a real failure along the way):
* A plain RANSAC "lowest plane" locks onto the **underside of the crown**. Requiring the plane to have almost all
  *opaque* mass above it fixes this.
* Thresholds derived from the full scene extent are ruined by the **sky shell** (and by junk *below* the ground).
  Use robust quantiles and measure the ground's actual thickness after a least-squares refit.
* Real ground (grass, litter, fitted scan splats) is a **layer, not a plane**. Also remove its fringe outside the
  trunk footprint.
* A hard crop radius cuts **leaning or lopsided crowns**. Using the crop only as a seed and growing by
  connectivity fixes this.
* Connectivity needs care in both directions. Bridge small gaps so sparse crowns stay one piece, but let only
  **voxels with ≥ 2 opaque splats** form bridges, or chains of floaters glue a nearby bush onto the tree. Size
  the voxels from the splat spacing.
* Loose **outlier removal** matters: at `std_ratio=2.5` it deleted 2% of a clean tree (isolated leaves).

The COLMAP stage was verified on 36 synthetic renders: all 36 registered, and camera centres matched ground truth to
0.2% of the orbit radius after similarity alignment.

### 3.5 Scanned point clouds (LiDAR / photogrammetry `.ply`, `.las`, `.xyz`)
There are no photos to optimise against, so `gs4d.pointcloud` *fits* Gaussians to the points:
* **surfels** (up to 3 points per Gaussian): one Gaussian per point, oriented by a PCA of its 12 neighbours. The
  result is flat on bark and leaves, elongated along twigs, and sized from the local spacing;
* **voxel fitting** (dense scans): mean colour and point *covariance* per voxel, with the voxel size chosen to hit
  the Gaussian budget;
* survey coordinates (UTM etc.) are recentred in float64 before the float32 cast.

`load_scan()` (and `scripts/animate_pointcloud.py`) isolates the tree on a coarse fit first, then spends the
whole budget on the tree's points. Limitation: fitted splats are view-independent (SH degree 0). For true 3DGS
quality you still need the original photos (appendix of the notebook).

## 4. Tree-specific tips

* **Sky through the crown.** Make sure the masks are *mattes* (soft), not blobs; BiRefNet handles the holes.
  Blob-like masks (hand-painted or SAM at low resolution) glue sky splats inside the crown.
* **Other trees in the background.** rembg keeps the salient object; `compute_masks(keep_center=True)` drops blobs
  that don't touch the image centre. For forests, use SAM 2/3 with a click or box on your tree.
* **Roots / ground contact.** Use `ground_mask(..., keep_radius=0.05)` to protect a disc around the trunk base, and
  `extract_skeleton(pin_below=0.03)` to keep the lowest splats static during animation.
* **Autumn foliage.** Set `LEAF_HUE_DEG≈35` (notebook step 4) so leafness picks orange and yellow.
* **Wind during capture** smears leaves. Capture on a calm day, or see *Iterative Motion Compensation for Canonical
  3D Reconstruction from UAV Plant Images Captured in Windy Conditions* (arXiv 2510.15491).
