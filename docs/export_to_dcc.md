# Exporting animated splats to Blender, Houdini and other tools

`gs4d.export_all(anim, "out/tree_wind")` (or notebook step 7) writes:

```
tree_wind/
├── tree_wind_rest.ply            # rest pose (re-simulate / re-rig it in your DCC)
├── ply_sequence/
│   ├── tree_wind_0001.ply        # one reference-layout 3DGS PLY per frame (frame numbers start at 1)
│   └── ...
├── tree_wind.usdc                # OpenUSD ParticleField3DGaussianSplat, time-sampled
├── gs4d_import_sequence.py       # Blender importer (native Geometry Nodes)
├── preview.mp4                   # notebook preview (when present)
└── manifest.json                 # frame count, fps, up axis, wind parameters
```

## Conventions

| | PLY sequence | USD (`ParticleField3DGaussianSplat`) |
|---|---|---|
| position | `x y z` | `positions` (time-sampled) |
| rotation | `rot_0..3` = quaternion **(w, x, y, z)**, un-normalised allowed | `orientations` (time-sampled, `quatf`) |
| scale | `scale_0..2` = **log** σ | `scales` = **linear** σ |
| opacity | `opacity` = **logit** | `opacities` = **linear** [0, 1] |
| colour | `f_dc_0..2` + `f_rest_*` (channel-major: all R, then G, then B) | `radiance:sphericalHarmonicsCoefficients` (per splat: DC, then bands; RGB triples) + `radiance:sphericalHarmonicsDegree` |
| up axis | whatever you chose (`REORIENT`) | stage `upAxis` metadata (Y or Z) |

`SH_DEGREE=0` gives the smallest files: 17 floats per splat per frame (including the zero normals most tools expect), about 3.7 MB per frame for 55k splats; SH degree 3 is 62 floats (13.6 MB).
Higher-order SH are copied from the rest pose and are **not rotated** with the splats (see limitations).

## Blender (4.5 LTS / 5.x)

**Option A: `gs4d_import_sequence.py`** (no add-on; tested headless on Blender 5.0.1)
1. Unzip the export. In Blender: *Scripting* workspace → *Text → Open* → `gs4d_import_sequence.py` → *Run Script*.
   The script looks for `ply_sequence/` next to itself; edit `SEQ_DIR` at the top if you moved things.
2. It creates an object `GS4D_Tree` with a Geometry Nodes modifier:
   *Scene Time → zero-padded index (loops over the clip) → Replace String in `…_####.ply` → **Import PLY** →
   delete faint splats → store colour/alpha → instance an ico-sphere per splat (rotation from `rot_*`, scale
   `exp(scale_*) × Splat Size`) → set material*.
3. The material is emissive colour × Gaussian falloff `exp(-½ k² (1 − (N·I)²))` × opacity, mixed with Transparent.
   Use Cycles for correct transparency ordering. EEVEE uses dithered transparency (more samples = smoother).
4. Modifier inputs: *Splat Size* (σ multiplier, 2 = 95% of the Gaussian), *Detail* (ico subdivisions),
   *Min Opacity*, *Start Frame*, *Frame Count* (the clip loops).
5. Command line / render farm:
   ```bash
   blender -b -P gs4d_import_sequence.py -- --dir ./ply_sequence --camera --render //render/f_#### --engine CYCLES --samples 64
   ```
Splat proxies are an approximation of true splat rasterisation: good for layout, compositing and motion,
slightly softer or harder than a real splat renderer.

**Option B: KIRI Engine 3DGS Render** (free add-on; v5.x for Blender 5.1+). Import individual frames or the rest
pose for high-quality splat rendering. Its **proxy-mesh rigging** can also re-animate `tree_wind_rest.ply` inside
Blender, and its **4DGS mode** exports PLY sequences back out.

**Option C: Mediastorm Blender 3DGS/4DGS Viewer Node**: a Geometry Nodes viewer that also reads PLY sequences
through *Import PLY*.

## Houdini

* **Houdini 22** (native splats): load `ply_sequence/tree_wind_$F4.ply` with a File SOP (or the splat import
  node), deform or simulate further with standard SOPs, and render in **Karma**. Or reference `tree_wind.usdc`
  in **Solaris**.
* **Houdini 20.5 / 21 + GSOPs**: *Gaussian Splats Import* SOP with `$F4` in the path; *Gaussian Splats Convert*
  switches between GSOPs and native H21 attribute conventions.
* Re-simulating in Houdini: import `tree_wind_rest.ply`, build a KineFX/Vellum or vegetation rig, and let it drive
  the splat points (the approach in SideFX's H22 palm-tree demo).

## USD-native tools

`tree_wind.usdc` follows the OpenUSD **26.03+** schema. Open it in usdview (with `hdParticleField`), NVIDIA
Omniverse (Particle Fields), Nuke 17 (GeoImport → SplatRender) or Houdini Solaris. Tools on older USD versions will
not recognise the prim type. Use the PLY sequence there.

## Viewers, game engines, web

* **SuperSplat**, **Postshot**, **Brush**, **LichtFeld Studio**: open the PLY files (sequence playback depends on
  the tool/version).
* **Unity**: aras-p UnityGaussianSplatting (static) or the GSOPs Unity sequence player.
* **Unreal**: 3DGS plug-ins import PLY; sequences need a flipbook player or a custom Niagara setup.
* **Web**: Spark (THREE.js) can animate splats procedurally in a *Dyno* shader. Port `gs4d/wind.py`'s joint FK
  and flutter to GLSL for real-time wind without shipping per-frame data.

## Limitations of the export

* **SH rotation:** view-dependent colour (SH degree ≥ 1) is not rotated per splat. Tree motion is small, so this is
  rarely visible. Export with `SH_DEGREE=0` to be exact.
* **Size:** a PLY sequence stores every splat every frame. USD stores appearance once and only positions + orientations per frame (28 bytes per splat): 2.5× smaller than PLY at SH 0, ~9× at SH 3. For long
  clips, export the rest pose and re-animate in the DCC.
* **Orientation:** exports use the tree's up axis unless you pick `REORIENT` (Z-up for Blender and Unreal, Y-up for
  Houdini, Maya and USD's default).
