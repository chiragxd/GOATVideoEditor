"""
Finishing: cinematic LUTs (generated locally as .cube files) and stabilization helpers.

Looks are built as math on RGB (no third-party LUT files), written once to luts/<name>.cube and applied
by FFmpeg's lut3d filter. Drop your own .cube files into luts/ and they appear in the UI too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LOOKS = ["Auto (by mood)", "None", "Teal & Orange", "Warm Film", "Moody", "Clean Bright", "Vintage Fade",
         "Punchy Pop", "B&W Contrast"]

MOOD_TO_LOOK = {
    "excited": "Punchy Pop", "happy": "Clean Bright", "funny": "Punchy Pop", "confident": "Teal & Orange",
    "curious": "Clean Bright", "surprised": "Teal & Orange", "serious": "Moody", "emotional": "Warm Film",
    "angry": "Moody", "calm": "Warm Film",
}


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _s_curve(x: np.ndarray, k: float) -> np.ndarray:
    return np.clip(x + k * (x - x ** 2) * (2 * x - 1) * -1, 0, 1)


def apply_look(rgb: np.ndarray, look: str) -> np.ndarray:
    y = _luma(rgb)[..., None]
    if look == "Teal & Orange":
        shadows = np.clip(1 - y * 2, 0, 1)
        highs = np.clip(y * 2 - 1, 0, 1)
        out = rgb + shadows * np.array([-0.04, 0.02, 0.06]) + highs * np.array([0.06, 0.02, -0.05])
        out = _s_curve(np.clip(out, 0, 1), 0.25)
        out = y + (out - y) * 1.15
    elif look == "Warm Film":
        out = rgb * np.array([1.04, 1.0, 0.93]) + 0.025
        out = _s_curve(np.clip(out, 0, 1), 0.15)
        out = y + (out - y) * 0.95
    elif look == "Moody":
        out = _s_curve(rgb, 0.35) * 0.94
        out = y * 0.94 + (out - y * 0.94) * 0.8 + np.array([-0.01, 0.0, 0.02])
    elif look == "Clean Bright":
        out = np.power(np.clip(rgb, 0, 1), 0.92) * 1.02
        out = y + (out - y) * 1.08
    elif look == "Vintage Fade":
        out = 0.06 + rgb * 0.88
        out = out * np.array([1.03, 1.0, 0.92])
        out = y + (out - y) * 0.8
    elif look == "Punchy Pop":
        out = _s_curve(rgb, 0.3)
        out = y + (out - y) * 1.3
    elif look == "B&W Contrast":
        out = np.repeat(_s_curve(y, 0.4), 3, axis=-1)
    else:
        out = rgb
    return np.clip(out, 0, 1)


def ensure_lut(luts_dir: Path, look: str, size: int = 33) -> Path | None:
    if look in ("None", "Auto (by mood)", ""):
        return None
    custom = luts_dir / f"{look}.cube"
    if custom.exists():
        return custom
    luts_dir.mkdir(exist_ok=True)
    grid = np.linspace(0, 1, size)
    # .cube order: red changes fastest, then green, then blue.
    b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    out = apply_look(rgb, look)
    lines = [f'TITLE "{look}"', f"LUT_3D_SIZE {size}"] + [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in out]
    custom.write_text("\n".join(lines) + "\n", encoding="ascii")
    return custom


def available_looks(luts_dir: Path) -> list[str]:
    extra = sorted(p.stem for p in luts_dir.glob("*.cube") if p.stem not in LOOKS) if luts_dir.exists() else []
    return LOOKS + extra


def resolve_look(look: str, mood: str) -> str:
    return MOOD_TO_LOOK.get(mood, "Clean Bright") if look == "Auto (by mood)" else look
