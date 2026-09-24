"""Read / write Gaussian splat files.

* ``.ply``   - the reference 3DGS layout (Kerbl et al. 2023), as written by
               graphdeco, gsplat, nerfstudio, Postshot, Brush, KIRI, Polycam,
               Luma, SuperSplat (uncompressed) ...  Also accepts plain coloured
               point clouds (x, y, z, red, green, blue) and 2DGS files.
* ``.splat`` - the 32-byte-per-splat format popularised by antimatter15.

Only numpy is required.  ASCII PLY files fall back to ``plyfile`` if installed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from .gaussians import GaussianModel, logit, rgb_to_sh0, sh0_to_rgb, degree_from_num_rest

_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def _parse_header(f) -> tuple[str, list[tuple[str, int, list[tuple[str, str]]]], int]:
    first = f.readline()
    if first.strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of PLY header")
        tokens = line.decode("ascii", errors="replace").strip().split()
        if not tokens or tokens[0] in ("comment", "obj_info"):
            continue
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            elements.append((tokens[1], int(tokens[2]), []))
        elif tokens[0] == "property":
            if tokens[1] == "list":
                elements[-1][2].append((tokens[4], "list"))
            else:
                elements[-1][2].append((tokens[2], _PLY_TYPES[tokens[1]]))
        elif tokens[0] == "end_header":
            break
    return fmt, elements, f.tell()


def read_ply_vertices(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Return the ``vertex`` element of a PLY file as a dict of 1-D arrays."""
    with open(path, "rb") as f:
        fmt, elements, offset = _parse_header(f)
        if fmt == "ascii":
            try:
                from plyfile import PlyData  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise ValueError("ASCII PLY needs `pip install plyfile`") from e
            v = PlyData.read(str(path))["vertex"].data
            return {name: np.asarray(v[name]) for name in v.dtype.names}
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"unsupported PLY format {fmt!r}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        names = [e[0] for e in elements]
        if "vertex" not in names:
            raise ValueError("PLY has no vertex element")
        vprops = [p[0] for p in elements[names.index("vertex")][2]]
        if "packed_position" in vprops or "chunk" in names:
            raise ValueError(
                "This is a SuperSplat *compressed* PLY. Re-export it uncompressed from "
                "SuperSplat, or convert with `npx @playcanvas/splat-transform in.compressed.ply out.ply`."
            )
        for name, count, props in elements:
            if any(t == "list" for _, t in props):
                if name == "vertex":
                    raise ValueError("list properties on vertices are not supported")
                if names.index(name) < names.index("vertex"):
                    raise ValueError(f"cannot skip list element {name!r} before vertices")
                continue
            dtype = np.dtype([(pn, endian + pt) for pn, pt in props])
            if name == "vertex":
                f.seek(offset)
                data = np.fromfile(f, dtype=dtype, count=count)
                if len(data) != count:
                    raise ValueError("PLY file is truncated")
                return {pn: np.asarray(data[pn]) for pn, _ in props}
            offset += dtype.itemsize * count
    raise AssertionError("unreachable")


def _sorted_props(cols: dict, prefix: str) -> list[str]:
    pat = re.compile(re.escape(prefix) + r"(\d+)$")
    found = [(int(m.group(1)), k) for k in cols if (m := pat.match(k))]
    return [k for _, k in sorted(found)]


def _points_to_gaussians(cols: dict) -> GaussianModel:
    """Turn a plain coloured point cloud into isotropic Gaussians."""
    from scipy.spatial import cKDTree

    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    if all(c in cols for c in ("red", "green", "blue")):
        rgb = np.stack([cols["red"], cols["green"], cols["blue"]], 1).astype(np.float64)
        rgb = rgb / (255.0 if rgb.max() > 1.0 else 1.0)
    else:
        rgb = np.full((len(xyz), 3), 0.6)
    k = min(4, len(xyz) - 1)
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    s = np.clip(d[:, 1:].mean(1), 1e-5, None) * 0.7
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(xyz), 1))
    return GaussianModel.from_activated(xyz, quats, np.repeat(s[:, None], 3, 1), np.full(len(xyz), 0.9), rgb)


def gaussians_from_columns(cols: dict[str, np.ndarray]) -> GaussianModel:
    if not all(c in cols for c in ("x", "y", "z")):
        raise ValueError("PLY vertices need x, y, z")
    if "opacity" not in cols or "scale_0" not in cols:
        return _points_to_gaussians(cols)
    n = len(cols["x"])
    means = np.stack([cols["x"], cols["y"], cols["z"]], 1)
    scale_names = _sorted_props(cols, "scale_")
    log_scales = np.stack([cols[k] for k in scale_names], 1).astype(np.float64)
    if log_scales.shape[1] == 2:  # 2DGS surfels: add a very thin third axis
        log_scales = np.concatenate([log_scales, log_scales.min(1, keepdims=True) - 4.0], 1)
    rot_names = _sorted_props(cols, "rot_")
    quats = np.stack([cols[k] for k in rot_names], 1) if len(rot_names) == 4 else np.tile([1.0, 0, 0, 0], (n, 1))
    if all(c in cols for c in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh_dc = np.stack([cols["f_dc_0"], cols["f_dc_1"], cols["f_dc_2"]], 1)
    elif all(c in cols for c in ("red", "green", "blue")):
        rgb = np.stack([cols["red"], cols["green"], cols["blue"]], 1).astype(np.float64)
        sh_dc = rgb_to_sh0(rgb / (255.0 if rgb.max() > 1.0 else 1.0))
    else:
        sh_dc = np.zeros((n, 3))
    rest_names = _sorted_props(cols, "f_rest_")
    k = len(rest_names) // 3
    degree_from_num_rest(k)
    if k:
        # Reference layout is channel-major: [R_1..R_k, G_1..G_k, B_1..B_k]
        rest = np.stack([cols[name] for name in rest_names], 1).reshape(n, 3, k).transpose(0, 2, 1)
    else:
        rest = np.zeros((n, 0, 3))
    known = {"x", "y", "z", "nx", "ny", "nz", "opacity", "f_dc_0", "f_dc_1", "f_dc_2", "red", "green", "blue"}
    known |= set(scale_names) | set(rot_names) | set(rest_names)
    extra = {k: v for k, v in cols.items() if k not in known}
    return GaussianModel(means, quats, log_scales, cols["opacity"], sh_dc, rest, extra)


def read_splat(path: str | os.PathLike) -> GaussianModel:
    """antimatter15 ``.splat``: pos f32x3, scale f32x3, rgba u8x4, quat u8x4."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 32:
        raise ValueError(".splat size is not a multiple of 32 bytes")
    rec = raw.reshape(-1, 32)
    pos = rec[:, 0:12].copy().view("<f4").reshape(-1, 3)
    scale = rec[:, 12:24].copy().view("<f4").reshape(-1, 3)
    rgba = rec[:, 24:28].astype(np.float64) / 255.0
    quat = (rec[:, 28:32].astype(np.float64) - 128.0) / 128.0
    return GaussianModel.from_activated(pos, quat, scale, rgba[:, 3], rgba[:, :3])


def load_gaussians(path: str | os.PathLike) -> GaussianModel:
    path = Path(path)
    if path.suffix.lower() == ".splat":
        return read_splat(path)
    if path.suffix.lower() == ".ply":
        return gaussians_from_columns(read_ply_vertices(path))
    raise ValueError(f"unsupported file type {path.suffix!r} (expected .ply or .splat)")


def save_ply(
    model: GaussianModel,
    path: str | os.PathLike,
    sh_degree: int | None = None,
    extra_attributes: list[str] | None = None,
    write_normals: bool = True,
) -> Path:
    """Write the reference binary-little-endian 3DGS PLY layout.

    ``sh_degree`` truncates / zero-pads the SH bands (``0`` gives the smallest
    file, 14 floats per splat).  ``extra_attributes`` lists keys of
    ``model.extra`` to append as additional float properties (most viewers
    ignore unknown properties, but keep it off for maximum compatibility).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sh_degree is not None and sh_degree != model.sh_degree:
        model = model.with_sh_degree(sh_degree)
    n = len(model)
    k = model.sh_rest.shape[1]
    names = ["x", "y", "z"]
    if write_normals:
        names += ["nx", "ny", "nz"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    extra_attributes = list(extra_attributes or [])
    names += extra_attributes

    cols = [model.means]
    if write_normals:
        cols.append(np.zeros((n, 3), np.float32))
    cols.append(model.sh_dc)
    if k:
        cols.append(model.sh_rest.transpose(0, 2, 1).reshape(n, 3 * k))
    cols.append(model.opacity_logits[:, None])
    cols.append(model.log_scales)
    cols.append(model.quats)
    for key in extra_attributes:
        cols.append(np.asarray(model.extra[key], dtype=np.float32).reshape(n, 1))
    data = np.ascontiguousarray(np.concatenate([np.asarray(c, np.float32) for c in cols], 1), dtype="<f4")
    assert data.shape[1] == len(names)

    header = "ply\nformat binary_little_endian 1.0\ncomment written by gs4d\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return path


def save_splat(model: GaussianModel, path: str | os.PathLike) -> Path:
    """Write antimatter15 ``.splat`` (degree-0 colour only, sorted by size*opacity)."""
    path = Path(path)
    order = np.argsort(-(np.exp(model.log_scales.sum(1)) * model.opacities))
    m = model.subset(order)
    rec = np.zeros((len(m), 32), np.uint8)
    rec[:, 0:12] = m.means.astype("<f4").view(np.uint8).reshape(-1, 12)
    rec[:, 12:24] = m.scales.astype("<f4").view(np.uint8).reshape(-1, 12)
    rgba = np.concatenate([np.clip(sh0_to_rgb(m.sh_dc), 0, 1), m.opacities[:, None]], 1)
    rec[:, 24:28] = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)
    rec[:, 28:32] = np.clip(np.round(m.quats * 128.0 + 128.0), 0, 255).astype(np.uint8)
    rec.tofile(path)
    return path


__all__ = [
    "load_gaussians",
    "save_ply",
    "save_splat",
    "read_ply_vertices",
    "read_splat",
    "gaussians_from_columns",
    "logit",
]
