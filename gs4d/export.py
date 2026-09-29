"""Export animated Gaussian splats for DCC tools.

Formats
-------
``ply_sequence``  One reference-layout 3DGS ``.ply`` per frame
                  (``<name>_0001.ply`` ...).  The de-facto interchange for
                  animated splats: Houdini (GSOPs / native H21-22 splats),
                  Blender (``gs4d/data/gs4d_import_sequence.py`` or the KIRI
                  3DGS Render add-on), SuperSplat, Postshot, Brush, Unity.
``usd``           One ``.usdc`` using OpenUSD's ``ParticleField3DGaussianSplat``
                  schema (OpenUSD >= 26.03) with time-sampled positions and
                  orientations; appearance is stored once.  For Houdini
                  Solaris/Karma, NVIDIA Omniverse, Nuke 17 and other
                  USD-native tools.
``npz``           Compact NumPy archive: rest pose + per-frame positions
                  (float32) and quaternions (float16).  For custom viewers.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .gaussians import GaussianModel, SH_C0
from .io_ply import save_ply


def export_ply_sequence(
    frames,
    out_dir: str | os.PathLike,
    name: str = "tree_wind",
    start_frame: int = 1,
    sh_degree: int | None = None,
    progress: bool = True,
) -> list[Path]:
    """Write one PLY per frame.  ``frames`` is a WindAnimation or a list of models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n = len(frames)
    for i in range(n):
        f = frames[i]
        p = out_dir / f"{name}_{i + start_frame:04d}.ply"
        save_ply(f, p, sh_degree=sh_degree)
        paths.append(p)
        if progress and (i % max(1, n // 10) == 0 or i == n - 1):
            print(f"  wrote {p.name} ({i + 1}/{n})")
    return paths


def export_usd(
    frames,
    path: str | os.PathLike,
    fps: float = 24.0,
    start_frame: int = 1,
    up_axis: str = "Z",
    meters_per_unit: float = 1.0,
    sh_degree: int | None = None,
    prim_path: str = "/Tree/Splats",
    progress: bool = True,
) -> Path:
    """Animated ``UsdVolParticleField3DGaussianSplat`` (requires ``usd-core>=26.3``).

    Static per-splat data (scales, opacities, SH radiance) is authored once;
    ``positions``, ``orientations`` and ``extent`` are time-sampled.
    Conventions follow the schema: linear scales, linear [0,1] opacity, raw
    SH coefficients (DC first, then bands in the reference 3DGS order).
    """
    from pxr import Gf, Usd, UsdGeom, UsdVol, Vt

    if not hasattr(UsdVol, "ParticleField3DGaussianSplat"):
        raise RuntimeError("usd-core >= 26.03 is required (pip install -U usd-core)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rest = frames[0]
    if sh_degree is not None and sh_degree != rest.sh_degree:
        rest = rest.with_sh_degree(sh_degree)
    n = len(frames)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis.upper() == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.SetStartTimeCode(start_frame)
    stage.SetEndTimeCode(start_frame + n - 1)
    root = UsdGeom.Xform.Define(stage, prim_path.rsplit("/", 1)[0] or "/Tree")
    stage.SetDefaultPrim(root.GetPrim())
    gs = UsdVol.ParticleField3DGaussianSplat.Define(stage, prim_path)

    gs.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(rest.scales.astype(np.float32)))
    gs.CreateOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(rest.opacities.astype(np.float32)))
    deg = rest.sh_degree
    sh = np.concatenate([rest.sh_dc[:, None, :], rest.sh_rest], 1).reshape(-1, 3).astype(np.float32)
    gs.CreateRadianceSphericalHarmonicsDegreeAttr().Set(int(deg))
    gs.CreateRadianceSphericalHarmonicsCoefficientsAttr().Set(Vt.Vec3fArray.FromNumpy(sh))
    gs.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(np.clip(rest.sh_dc * SH_C0 + 0.5, 0, 1).astype(np.float32))
    )
    gs.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(rest.opacities.astype(np.float32)))

    pos_attr = gs.CreatePositionsAttr()
    ori_attr = gs.CreateOrientationsAttr()
    ext_attr = gs.CreateExtentAttr()
    for i in range(n):
        f = frames[i]
        tc = Usd.TimeCode(start_frame + i)
        pos_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(f.means, np.float32)), tc)
        # GfQuatf memory layout is (x, y, z, w): reorder from our (w, x, y, z)
        q = np.ascontiguousarray(f.quats[:, [1, 2, 3, 0]], np.float32)
        ori_attr.Set(Vt.QuatfArray.FromNumpy(q), tc)
        pad = 3.0 * rest.scales.max(1)[:, None]
        lo = (f.means - pad).min(0)
        hi = (f.means + pad).max(0)
        ext_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]), tc)
        if progress and (i % max(1, n // 10) == 0 or i == n - 1):
            print(f"  USD frame {i + 1}/{n}")
    stage.GetRootLayer().Save()
    return path


def export_npz(frames, path: str | os.PathLike, fps: float = 24.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rest = frames[0]
    n = len(frames)
    pos = np.zeros((n, len(rest), 3), np.float32)
    rot = np.zeros((n, len(rest), 4), np.float16)
    for i in range(n):
        f = frames[i]
        pos[i] = f.means
        rot[i] = f.quats
    np.savez_compressed(
        path,
        positions=pos,
        quats_wxyz=rot,
        log_scales=rest.log_scales,
        opacity_logits=rest.opacity_logits,
        sh_dc=rest.sh_dc,
        sh_rest=rest.sh_rest,
        fps=np.float32(fps),
    )
    return path


def write_manifest(out_dir: str | os.PathLike, info: dict) -> Path:
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps(info, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return p


BLENDER_SCRIPT = Path(__file__).resolve().parent / "data" / "gs4d_import_sequence.py"


def copy_blender_script(out_dir: str | os.PathLike) -> Path | None:
    """Copy the Blender importer next to the exported sequence."""
    src = BLENDER_SCRIPT
    if not src.exists():
        return None
    dst = Path(out_dir) / "gs4d_import_sequence.py"
    shutil.copyfile(src, dst)
    return dst


def zip_dir(src_dir: str | os.PathLike, zip_path: str | os.PathLike) -> Path:
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(src_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src_dir.parent))
    return zip_path


def export_all(
    anim,
    out_dir: str | os.PathLike,
    name: str = "tree_wind",
    formats: Iterable[str] = ("ply_sequence", "usd", "blender"),
    sh_degree: int | None = None,
    up_axis: str = "Z",
    fps: float | None = None,
) -> dict:
    """Export a :class:`~gs4d.wind.WindAnimation` in several formats at once."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = fps or getattr(getattr(anim, "p", None), "fps", 24.0)
    formats = set(formats)
    result: dict = {}
    frames = _CachedFrames(anim)
    rest = getattr(anim, "model", None)
    save_ply(rest if rest is not None else frames[0], out_dir / f"{name}_rest.ply", sh_degree=sh_degree)
    result["rest_ply"] = out_dir / f"{name}_rest.ply"
    if "ply_sequence" in formats or "blender" in formats:
        print("PLY sequence ...")
        result["ply_sequence"] = export_ply_sequence(frames, out_dir / "ply_sequence", name, sh_degree=sh_degree)
    if "usd" in formats:
        print("USD ...")
        try:
            result["usd"] = export_usd(frames, out_dir / f"{name}.usdc", fps=fps, up_axis=up_axis, sh_degree=sh_degree)
        except Exception as e:  # keep the other exports even if usd-core is missing/old
            print(f"  USD export skipped: {e}")
    if "npz" in formats:
        print("NPZ ...")
        result["npz"] = export_npz(frames, out_dir / f"{name}.npz", fps=fps)
    if "blender" in formats:
        result["blender_script"] = copy_blender_script(out_dir)
    info = {
        "name": name,
        "frames": len(frames),
        "fps": fps,
        "start_frame": 1,
        "up_axis": up_axis,
        "num_gaussians": len(frames[0]),
        "sh_degree": sh_degree if sh_degree is not None else frames[0].sh_degree,
        "ply_pattern": f"ply_sequence/{name}_####.ply",
        "wind_params": anim.p.to_dict() if getattr(anim, "p", None) is not None else None,
    }
    result["manifest"] = write_manifest(out_dir, info)
    return result


class ReorientedFrames:
    """Rotate every frame (e.g. to make +Z or +Y the up axis for a DCC tool).

    Higher-order SH are not rotated; export with ``sh_degree=0`` if the
    rotation is large and view-dependent colour matters.
    """

    def __init__(self, frames, rotation: np.ndarray):
        self.frames = frames
        self.R = np.asarray(rotation, np.float64)
        self.p = getattr(frames, "p", None)
        self.model = frames.model.transformed(self.R) if hasattr(frames, "model") else None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> GaussianModel:
        return self.frames[i].transformed(self.R)


def up_rotation(up, target: str = "Z") -> np.ndarray:
    """Rotation taking the model's ``up`` vector to +Z (Blender) or +Y (Houdini, Maya, USD default)."""
    from .quaternion import rotation_between
    from .skeleton import parse_up

    return rotation_between(parse_up(up), [0.0, 0.0, 1.0] if target.upper() == "Z" else [0.0, 1.0, 0.0])


class _CachedFrames:
    """Pose each frame once even when several exporters iterate over them."""

    def __init__(self, anim, max_cached: int = 4):
        self.anim = anim
        self.cache: dict[int, GaussianModel] = {}
        self.max_cached = max_cached

    def __len__(self) -> int:
        return len(self.anim)

    def __getitem__(self, i: int) -> GaussianModel:
        if i not in self.cache:
            if len(self.cache) >= self.max_cached:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = self.anim[i]
        return self.cache[i]
