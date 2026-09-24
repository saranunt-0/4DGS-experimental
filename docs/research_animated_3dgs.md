# Animated 3D Gaussian Splatting: state of the art (September 2026)

This report covers the two ways to get a *moving* Gaussian-splat asset:

* **A. Animate a static 3DGS model**: add motion to a scan or generated splat.
* **B. Reconstruct 4DGS from video**: the motion comes from footage.

It then covers **C.** tree- and vegetation-specific work, **D.** the tools and file formats that matter for
getting results into Blender, Houdini and other DCCs, and **E.** what this repository implements and why.

> Scope note: sources were gathered in September 2026. Papers are cited by title and arXiv id or venue so they can be
> looked up. Items marked *(code n/a)* had no public code at the time of writing.

---

## TL;DR: what to use for "a tree with flowing, shaky leaves"

| Goal | Best practical option today | Research SOTA to watch |
|---|---|---|
| Hero shot in a VFX pipeline | **Houdini 22** native splats: drive splat points with a vegetation / wind simulation, render in Karma | DynamicTree (CVPR 2026) *(code n/a)* |
| Blender-only pipeline | **KIRI 3DGS Render 5.x** (proxy-mesh rigging, bake to a 4DGS PLY sequence), or this repo's wind + native GN importer | — |
| Quick, free, scriptable, runs in Colab | **This repo**: skeleton + damped-oscillator wind + leaf flutter, exporting PLY sequence / USD | "Wind on Trees" (2026) oscillator prior |
| Real tree moving in real wind (capture) | Multi-view synchronized video to 4DGS; monocular capture of wind-blown foliage is still largely unsolved | Wind on Trees (2026), MoSca / Shape of Motion |
| Physically interactive (poke / drag) | PhysGaussian-family MPM with learned materials (PhysDreamer, OmniPhysGS) | Gaussian Swaying (WACV 2026, aerodynamics) |

Why a procedural/physics prior rather than a learned 4D model for trees? Dense canopies are the hardest case for
video-based 4D reconstruction. The canopy is thin and self-occluding, almost everything moves at once, and motion
along the viewing direction is unobservable from a single camera. The 2026 *Wind on Trees* study found that a
per-part **damped-oscillator prior extrapolates better** (to new time windows and new wind speeds) than freely learned
deformation fields. This is the modelling family used here and in production tools (Houdini, SpeedTree-style game
shaders).

---

## A. Animating a static 3DGS model

### A1. Physics simulation on the Gaussians
Gaussians are treated as simulation particles, or embedded in a simulated proxy, and re-posed every frame.
Positions, rotations and covariance are updated from the deformation gradient.

| Method | Idea | Notes |
|---|---|---|
| **PhysGaussian** (CVPR 2024) | Material Point Method (MPM) directly on the Gaussians; covariance follows the deformation gradient | Foundation of the family; materials are set by hand |
| **PhysDreamer** (ECCV 2024, arXiv 2404.13026) | Learns a Young's-modulus field and initial velocities by matching a *video-diffusion* rollout; differentiable MPM | Demos: flowers, plants, an alocasia. Elastic dynamics, closest prior work to "plants that move" |
| **DreamPhysics** (AAAI 2025) | Physical parameters optimised with text-to-video / SVD priors | |
| **OmniPhysGS** (ICLR 2025, arXiv 2501.18982) | Per-Gaussian learnable constitutive models (elastic, plastic, fluids…) | |
| **PhysSplat** (ICCV 2025, arXiv 2411.12789) | An MLLM predicts materials, with efficient driving-particle sampling | Zero-shot material guessing |
| **Gaussians2Life** | Text-driven: video diffusion output lifted to 3D motion of scene parts | Scene-level animation |
| Spring-Gaus, GIC, NeuMA, PhysTwin | Spring-mass / continuum models identified from real video | "Digital twins" of deformable objects |
| **Gaussian Swaying** (WACV 2026, arXiv 2512.01306) | Gaussians as surface patches with normals and area, so *aerodynamic* forces act on them | Wind on thin surfaces such as leaves and cloth |
| i-PhysGaussian (arXiv 2602.17117), NewtonGS (2608.07598), PhysMAS (2609.07174), Real-Time Physics with Dynamic Mesh-Gaussian Reconstructions (2606.00444) | 2026 follow-ups: implicit integration, learned Newtonian dynamics, multi-agent composition, real-time mesh-Gaussian hybrids | |

**Takeaway:** MPM-style methods give the most physical results but are heavy (a GPU plus minutes per clip) and need
material fields. For wind on a tree, a **reduced model** is far cheaper and easier to art-direct: rigid parts or
joints plus oscillators, or modal analysis.

### A2. Rigging and skinning (skeleton- or cage-driven)
The static splat is bound to a skeleton, cage or proxy mesh (linear-blend or dual-quaternion skinning), which is then
animated like any rig.

* **SC-GS** (CVPR 2024): sparse control points drive dense Gaussians, which makes a scene editable and animatable.
* Auto-rigging, mostly mesh-first but directly usable for splats via a proxy: **UniRig** (SIGGRAPH 2025),
  **Puppeteer** (arXiv 2508.10898), MagicArticulate, RigAnything, Anymate; **G-Skin** (2608.01726, binds 3D Gaussians
  with generative visual priors), **GaussiAnimate** (2604.08547), **Rigel3D** (2605.13129),
  **RigPAPR** (2606.06685, rig from a fixed-viewpoint video).
* Tooling: Houdini (KineFX, Bone Deform, Surface Deform on splat points), **KIRI 3DGS Render 5** for Blender (proxy
  rigging, then bake the rigged splat sequence), **GSOPs** `gaussian_splats_deform` (mesh-driven deformation, Vellum
  "jellify").

For trees, the "rig" is the branch skeleton. That's exactly what this repo builds, either from a procedural generator
or by skeletonising the splats.

### A3. Generative 4D from a static asset (video-diffusion guided)
Animate3D, AnimateAnyMesh, DreamGaussian4D, Align Your Gaussians, STAG4D, SC4D, 4DGen and similar: a text, image or
video prompt drives a deformation field through score distillation or reconstruction from generated multi-view
videos. They are good for characters and short actions. Stochastic small-scale motion such as leaf flutter is
typically blurred away, and the results are hard to art-direct.

---

## B. Reconstructing 4DGS from video

### B1. Synchronised multi-view video (studio / volumetric capture)
* **4DGS** (Wu et al., CVPR 2024; HexPlane deformation field), **Deformable 3DGS** (CVPR 2024),
  **Real-time 4DGS** (4D Gaussian primitives, ICLR 2024), **Spacetime Gaussians** (CVPR 2024),
  **Dynamic 3D Gaussians** (tracking, 3DV 2024), Ex4DGS, 4D-Rotor GS, FreeTimeGS (CVPR 2025).
* Speed and compactness: **Disentangled 4DGS** (arXiv 2503.22159, 343 FPS), 4DGS-1K, **Multi4D** (2606.22197),
  MoRGS (2603.25042); streaming codecs 4DGCPro (2509.17513) and 4D-MoDe (2509.17506); survey: SUCCESS-GS (2512.07197).
* Fewer cameras: **4C4D** (arXiv 2604.04063, four cameras), 4D reconstruction from sparse dynamic cameras (2606.04593).
* Production: **Gracia** (keyframes + motion deltas, ~80 Mbit/s web streaming, MINT format, PLY flipbooks for post),
  4DV.ai / Mediastorm (SIGGRAPH 2025 Real-Time Live! "InfiniteStudio"), Effigy.

### B2. Monocular casual video
* **Shape of Motion** (ICCV 2025): SE(3) motion bases fused from depth and tracking priors. **MoSca** (CVPR 2025):
  a 4D motion scaffold from foundation-model priors. Gaussian Marbles; GFlow (AAAI 2025).
* 2025–26: Uncertainty Matters in Dynamic GS (2510.12768), **RiGS** (rigid-aware, 2605.23672), **Ground4D**
  (2606.28828), flow-splatting efficient 4DGS (2606.29976), Gaussian Sequences with Multi-Scale Dynamics (2602.13806),
  Lift4D (2606.23688).
* Camera and geometry priors used by these methods: VGGT, MonST3R, CUT3R, MegaSaM and similar feed-forward
  pose/depth models.

### B3. Feed-forward (seconds instead of hours)
**L4GM** (NeurIPS 2024; animated object from a single-view video in one pass), **4DGT** (NeurIPS 2025; 4D Gaussian
Transformer trained on real monocular video), **BTimer** (bullet-time reconstruction), **MoVieS**
("4D dynamic view synthesis in one second"), **Any4D** (2512.10935, metric 4D), **4DNeX** (2508.13154),
Forge4D (humans, 2509.24209). Survey: *Advances in Feed-Forward 3D Reconstruction and View Synthesis* (2507.14501).

### B4. Generative (video diffusion to 4D)
**CAT4D** (2411.18613; multi-view video diffusion, then 4DGS), **SV4D 2.0** (ICCV 2025), **Lyra** (NVIDIA,
2509.19296; self-distils a video model into 3D/4D Gaussians), **SS4D** (2512.14284, native 4D latents),
**ShapeGen4D** (2510.06208), **Turbo4DGen** (2603.29572), ST-Gen4D (2605.07390).

**Takeaway:** for a tree, the monocular route needs a *still* tree for geometry and a separate motion source.
Multi-view synchronised capture works but is expensive. Generative 4D still struggles with thousands of
independently moving leaves.

---

## C. Trees and vegetation specifically

| Work | What it gives you |
|---|---|
| **DynamicTree** (CVPR 2026, arXiv 2510.22213) *(code n/a)* | First feed-forward long-term *interactive* animation of 3DGS trees. A sparse-voxel "spectrum" (modal basis) generates mesh motion and Gaussians are bound to the mesh; it supports real-time modal response to forces. Introduces **4DTree** (8,786 animated tree meshes). |
| **Wind on Trees** (arXiv 2609.17810) | Physically parameterised deformation for wind-driven trees: one damped harmonic oscillator per rigid part, driven by observed wind, integrated with differentiable RK4. It extrapolates better than learned fields; physical-parameter recovery from monocular video remains weak. |
| **Gaussian Swaying** (WACV 2026) | Surface-based aerodynamic simulation on 3D Gaussians. |
| **PhysDreamer** (ECCV 2024) | Plants and flowers respond to pokes (learned materials plus MPM). |
| **LeafFit** (CGF 2026, arXiv 2602.11577) | Turns a 3DGS plant into an instanced template-leaf mesh asset with shader-based deformation (lightweight and animatable). |
| **GaussianPlant** (arXiv 2512.14087) | Structure-aligned, hierarchical GS that disentangles branching structure from appearance. |
| 3DWPGS (IJMSSC 2025) | End-to-end woody-plant GS modelling plus physical simulation. |
| Iterative Motion Compensation (arXiv 2510.15491) | Canonical reconstruction of plants imaged in windy conditions, useful when your capture day wasn't calm. |
| Classic CG | Stam 1997 (stochastic modal dynamics under turbulence); Habel et al. 2009 (physically guided tree animation); Diener et al. 2009 (wind projection basis); Sousa, *GPU Gems 3* ch. 16 (Crysis vegetation: main bending + detail bending + leaf edge flutter). This is the lineage this repo's model follows. |
| Skeletonisation | Verroust & Lazarus 2000; Xu, Gossett & Chen 2007 (level-set / geodesic slicing of laser-scanned trees), used here to get a skeleton from a captured splat tree. |

---

## D. Tools, pipelines and formats

### D1. DCC and viewer support
* **Houdini 22** (released July 2026): Gaussian splats are a first-class geometry type. You can train splats from
  photos or generate them from 3D renders; animate them with standard deformers (Bone Deform, Surface Deform);
  drive them with simulations (SideFX demos a **wind-animated palm tree** whose splat points follow a vegetation
  sim); relight them with SideFX Labs prototype tools (SH plus normals via Copernicus); and render them in **Karma**.
  Houdini 21 had a limited, read-only technical preview.
  This is most likely the "Houdini demo" you saw. A free SideFX tutorial by Bogdan Lazar covers
  import → segment → KineFX skeleton → APEX → deform → Karma.
* **GSOPs 2.9** (CG Nomads; Houdini 20.5/21): import/edit/export, isolation, floater removal, deformation, Vellum
  "jellify", relighting, **PLY animation sequences** (one `.ply` per frame), and a convert SOP to and from Houdini's
  native splat conventions. houdini-gsplat (Plattipus) provides USD-native splats in Solaris.
* **Blender**: **KIRI 3DGS Render** (v4 edit/render modes; v5.x, June 2026, adds proxy rigging, experimental light
  baking and a **4DGS mode** that exports animated PLY sequences; Blender 5.1+). Mediastorm's
  **Blender 3DGS/4DGS Viewer Node** (Geometry Nodes, uses Blender's *Import PLY* node per frame). Blender 5.0's
  native **Import PLY** geometry node reads all 3DGS attributes (`f_dc_*`, `opacity`, `scale_*`, `rot_*`). We
  verified this, and it is what `gs4d_import_sequence.py` builds on.
* **Nuke 17** (USD GeoImport and SplatRender; *Splat-PLY-to-Nuke* stitches PLY sequences into USD value clips),
  **NVIDIA Omniverse** (Particle Fields; ovrtx), **Unity** (aras-p UnityGaussianSplatting; GSOPs sequence player),
  **Unreal** (several 3DGS plug-ins).
* Viewers and editors: **SuperSplat** (PlayCanvas; plays PLY sequences), **Postshot**, **Brush**, **LichtFeld Studio**,
  **Spark 2.0** (World Labs; THREE.js renderer whose *Dyno* shader graphs can animate splats procedurally on the GPU
  in the browser).

### D2. File formats for animated splats
| Format | Status | Use it for |
|---|---|---|
| **PLY sequence** ("flipbook"): `name_0001.ply …` in the reference 3DGS layout | De-facto standard; works everywhere | Houdini, Blender, SuperSplat, Postshot, Brush, Unity, Nuke via conversion |
| **OpenUSD `UsdVolParticleField3DGaussianSplat`** (OpenUSD **26.03**, March 2026) | Official schema: linear scales, linear [0,1] opacity, SH coefficients, projection/sort hints. Time samples give animation. Reference Hydra renderer `hdParticleField`. Converters: NVIDIA `usd-convert-gsplat`, AOUSD example script | Solaris/Karma, Omniverse, Nuke 17, usdview |
| SPZ (Niantic), `.splat`, SOG/compressed PLY (PlayCanvas) | Static compression formats | Web delivery (static) |
| Gracia MINT and other proprietary streams | Keyframes plus deltas | Volumetric video streaming |

There is **no standard compressed 4D splat format yet**. PLY sequences for interchange plus USD for
USD-centric pipelines is the safe combination, and it's what this repo exports.

---

## E. What this repository implements (and why)

1. **Object-only tree**: a procedural splat tree (guaranteed clean), your scan cleaned by `gs4d.cleanup`, or your own
   masked training (`gs4d.train`, gsplat). See [object_only_gaussians.md](object_only_gaussians.md).
2. **Skeleton**: the exact one from the generator, or extracted from any splat tree by geodesic level-set slicing.
3. **Wind** (`gs4d/wind.py`):
   * The wind field has a mean flow, **gust fronts travelling downwind** (so gusts sweep across the crown) and
     smooth 3D turbulence. It can loop seamlessly.
   * Each joint is a 2-DOF **damped oscillator** driven by the drag torque of its whole subtree,
     `τ_j = Σ_i (x_i − p_j) × a_i |v_i| v_i`, computed for all joints at once with one sparse
     ancestor-matrix product per sub-step.
   * Compliance and natural frequency are interpolated from trunk to twig with a pipe-model thickness proxy
     (subtree leaf area). The result is a slow, stiff trunk and fast, flexible twigs.
   * **Forward kinematics** re-poses each Gaussian's centre and orientation (`q' = q_joint ⊗ q`).
   * **Leaf flutter**: a spatially coherent random-Fourier rotation and jitter field, scaled by local wind speed.
     Anisotropic leaf splats change footprint as they turn, which produces the characteristic shimmer.
4. **Export**: PLY sequence, animated USD `ParticleField3DGaussianSplat`, compact NPZ, and a Blender Geometry-Nodes
   importer tested headlessly on Blender 5.0.

Known limitations (and possible upgrades):
* Higher-order SH are not rotated with the splats. View-dependent colour is slightly off under large rotations
  (tree motion is small, and exporting with `SH_DEGREE=0` avoids it entirely).
* No collisions and no two-way coupling between branches (the usual trade-off of real-time tree animation).
* Extracted skeletons of very dense crowns merge neighbouring branches into coarse joints. The motion is still
  plausible, but for hero shots use Houdini/KineFX or, once released, DynamicTree.
* An upgrade path is to swap the analytic oscillators for DynamicTree's learned modal basis, or to fit oscillator
  parameters to a reference video as in *Wind on Trees*.
