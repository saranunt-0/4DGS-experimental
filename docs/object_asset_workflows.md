# How people make object-only 3DGS assets (and what we did with the oak scan)

**Question:** how do studios and tools produce a Gaussian-splat *asset*, meaning one object with no environment,
ready to drop into Blender, Houdini, Unreal or a web viewer? And which route fits a scanned tree that exists only
as a coloured point cloud?

Short answer: there are five routes. Which one you can use depends on what you have: photos, a mesh, a point cloud
or a single image. For a point cloud *without* its photos, the practical options are **direct conversion** (fit
Gaussians to the points, which is what this repo does) and **render-to-splat distillation** (render the points from
many virtual cameras, then train a real 3DGS; this needs a GPU).

## 1. The five routes

| # | Route | Input | How the environment is excluded | Tools (2025-26) | Quality / cost |
|---|---|---|---|---|---|
| 1 | **Masked capture + training** | photos / video of the real object | Per-frame object masks; train on a random or transparent background with an alpha loss; crop box before export | Polycam *Object capture*, Postshot, Luma, KIRI Engine; gsplat/nerfstudio with masks; SuperSplat for final crop | Best, with view-dependent shine and real detail. Needs the photos and a GPU (or a cloud app) |
| 2 | **Render-to-splat distillation** | any renderable asset: mesh, point cloud, volume, CG scene | The render has an **alpha channel** and nothing else in it, so the trained splat is object-only by construction | Houdini 22 native splats and GSOPs `generate_training_data` (PNG with alpha → masked training); Blender *Camera Array Tool* (rig of cameras + COLMAP export, also for 4DGS); KIRI 3DGS Render | Very clean. Quality is limited by the renderer. Needs a GPU for training (~10-30 min) |
| 3 | **Direct conversion** (no training) | mesh or point cloud | Only the object's geometry is converted | EA SEED **Mesh2Splat** (mesh → splats in milliseconds, one splat per texel-sized triangle area); point-cloud surfel/voxel fitting (PCA of neighbours; the standard 3DGS init from SfM/LiDAR points); `gs4d.pointcloud` | Instant and CPU-only. No view-dependent effects (SH degree 0). Detail equals the input's density |
| 4 | **LiDAR + photo hybrids** | LiDAR point cloud + photos | LiDAR geometry initialises and regularises the splats; semantic masks separate the trees | LiDAR-3DGS, DensifyBeforehand, GTLR-GS, forest digital-twin pipelines (LiDAR-guided semantic 3DGS), TreeDGS | Research-grade and best for large vegetation. Needs both sensors and a GPU |
| 5 | **Generative** | one or a few images | The model outputs only the object | Microsoft TRELLIS.2, Meta SAM 3D Objects | Seconds to minutes. The unseen side is invented, and fine foliage becomes blobby. Good for background props |

Common clean-up at the end of *every* route: opacity prune → delete giant/needle splats → crop box → remove
floaters (SuperSplat, GSOPs, KIRI crop modifier). For captured scenes, see
[`object_only_gaussians.md`](object_only_gaussians.md). It covers masks, alpha loss, mask voting and geometric
isolation, with measured numbers.

## 2. Tree-specific problem: sky bleed

Photogrammetry and LiDAR colourisation project **sky pixels onto leaf and twig points**. This happens at the
crown's silhouette and through gaps between leaves. The geometry is real, but the colour is blue or white. A
fitted or trained splat then shows a blue-grey haze on the crown. Survey tools address this with
**select-by-colour** clean-up (3Dsurvey) or by **annotating sky** in the images before densification (Pix4D).

The oak scan has exactly this problem. 11.1% of its 4.6M points are sky-coloured:
* clearly **blue**;
* a **neutral grey**, the colour of an overcast sky. The median is RGB (0.66, 0.68, 0.67), three times brighter
  than the median leaf colour (0.20, 0.22, 0.03).

Deleting these points would thin out the crown's edge. Instead, `gs4d.pointcloud.fix_sky_bleed` **keeps the points
and repaints them** with the median colour of their 16 nearest non-sky neighbours. It deletes only sky points with
no object neighbour nearby (real floaters).

Grey is only treated as sky when it sits **inside foliage** (at least 20% of its neighbours are leaf-green).
Otherwise white birch bark or a white object would be repainted. The first version flagged every near-white point,
and the test showed it deleted a white birch trunk. If more than 35% of a cloud looks like sky, the object itself is
probably white or blue, and the fix does nothing.

## 3. What we did with the 200-year-old oak (`test_asset/200-year-old-oak-tree.zip`)

The asset is a Sketchfab-style download: `model.zip → source/oak_RGB_2cm.zip → oak_RGB_2cm.ply`. It holds
**4,635,119 coloured points** at 2 cm spacing and no photos. That rules out route 1. Route 2 needs a GPU, which the
build machine doesn't have. So we used **route 3 (direct conversion)**, plus the sky fix and the geometric
isolation from this repo:

```bash
python scripts/animate_pointcloud.py test_asset/200-year-old-oak-tree.zip --out outputs/oak \
    --max-gaussians 600000 --static-only          # static object-only 3DGS only (~2-3 min on 4 CPUs)
```

| Step | Result on the oak |
|---|---|
| Nested zip → point cloud | `source/oak_RGB_2cm.zip` opened automatically |
| Sky bleed repair | 513,491 points recoloured (blue, and grey inside foliage), 592 sky floaters removed |
| Up axis | **+Z**. The first heuristic said +Y: this oak's crown is wider than it is tall and hangs down beside a short trunk. The detector now also uses a *mass* cue (the trunk end is the lightest end); regression test `test_guess_up_broad_crown` |
| Isolation (ground, clutter, floaters) | No ground plane in this scan. 0.6% statistical outliers removed; 4,559,656 points kept |
| Gaussian fitting | Voxel mode (~7 points per splat): each splat gets the mean colour and **covariance** of its voxel's points, so it is flat on leaves and bark and elongated along twigs |
| Output | `tree_gaussians_static.ply`, a standard 3DGS PLY (SH degree 0) that opens in SuperSplat, Postshot, KIRI 3DGS Render, Houdini 22 and this repo |

Then the tree is animated like any other splat tree: skeleton by geodesic slicing, wind oscillators, forward
kinematics and leaf flutter (see the README).

### Honest limitations of route 3
* **No view-dependent colour.** Leaves don't sparkle as you orbit; the colour is the scan's baked colour.
* **Detail is capped by the scan.** 2 cm points → ~13 cm splats at 600k. Individual leaves blur into foliage clumps
  up close. Raise `--max-gaussians` (the file grows ~68 bytes per splat) or crop to a region.
* **Scan lighting is baked in.** Relighting needs a normal-aware renderer (Houdini 22, KIRI 5).

### Upgrade path (GPU, e.g. Colab): route 2 on the same scan
1. Render the fitted splat (or the raw points) from 150-300 cameras on two orbits and a dome, **with alpha**
   (`gs4d.render.orbit_cameras` + gsplat, or Houdini `generate_training_data`, or Blender Camera Array Tool).
2. Write the cameras as a COLMAP model (`gs4d.colmap`) and train with `gs4d.train.train_object_splat`
   (random background + alpha loss, so no background can form).
3. Result: a *trained* object-only 3DGS with optimised opacity and anisotropy, typically sharper foliage for the
   same splat count. View-dependent effects stay absent unless the renders include them.

## Sources
- [Polycam: 3D object capture with photogrammetry & Gaussian splatting](https://poly.cam/object-capture)
- [Swyvl: Creating Gaussian splats with Postshot, Polycam, Luma](https://swyvl.io/blog/how-to-create-gaussian-splats/)
- [PlayCanvas: recommended tools for creating splats](https://developer.playcanvas.com/user-manual/gaussian-splatting/creating/recommended-tools/)
- [SuperSplat editor](https://playcanvas.com/supersplat/editor/) · [Jawset Postshot](https://www.jawset.com/)
- [SideFX: Houdini 22 Gaussian splats](https://www.sidefx.com/products/whats-new-in-h22/gaussian-splats/)
- [GSOPs: Gaussian Splatting Operators for Houdini (`generate_training_data`, alpha-masked training)](https://github.com/cgnomads/GSOPs)
- [Camera Array Tool for Blender (synthetic 3DGS/4DGS training data)](https://toppinappi.gumroad.com/l/Camarray)
- [KIRI Engine 3DGS Render for Blender](https://github.com/Kiri-Innovation/3dgs-render-blender-addon)
- [EA SEED Mesh2Splat](https://github.com/electronicarts/mesh2splat) · [Radiance Fields: Mesh2Splat, instant mesh → 3DGS](https://radiancefields.com/mesh2splat-instant-3d-mesh-conversion-to-3d-gaussian-splatting)
- [graphdeco-inria/gaussian-splatting #811: using a point cloud to seed Gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting/issues/811)
- [3DGS-to-PC (the reverse direction, splats → points)](https://arxiv.org/abs/2501.07478)
- [LiDAR-3DGS: LiDAR reinforcement for multimodal initialization](https://www.sciencedirect.com/science/article/abs/pii/S0097849325001347)
- [DensifyBeforehand: LiDAR-assisted content-aware densification](https://arxiv.org/pdf/2511.19294)
- [GTLR-GS: LiDAR-regularised 3DGS](https://arxiv.org/pdf/2603.23192)
- [LiDAR-Guided Semantic 3D Gaussian Splatting for Forest Digital Twins](https://doi.org/10.3390/rs18111696)
- [TreeDGS: aerial Gaussian splatting for DBH measurement](https://arxiv.org/pdf/2601.12823)
- [Comparative analysis of novel view synthesis and photogrammetry for forest stands](https://arxiv.org/pdf/2410.05772)
- [Microsoft TRELLIS / TRELLIS.2](https://github.com/microsoft/TRELLIS) · [Meta SAM 3D (Roboflow overview)](https://blog.roboflow.com/sam-3d/)
- [3Dsurvey: select-by-colour point cloud cleaning](https://3dsurvey.si/select-by-color-feature-cleaning-point-cloud/)
- [Pix4D: removing the sky from the point cloud with the annotation tool](https://support.pix4d.com/hc/en-us/articles/212262943-How-to-remove-the-Sky-from-the-Point-Cloud-using-the-Annotation-Tool-)
