"""
Person cutout (U²-Net-p via onnxruntime, ~4.7 MB, CPU-friendly) for the "text behind subject" effect.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

class PersonMatte:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.prev: np.ndarray | None = None

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Soft person mask (0..1) at frame size, temporally smoothed to avoid flicker."""
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = x / max(float(x.max()), 1e-6)
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        out = self.session.run(None, {self.input: x.transpose(2, 0, 1)[None]})[0][0, 0]
        out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
        m = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        m = np.clip((m - 0.35) / 0.3, 0, 1)  # firm up edges
        m = cv2.GaussianBlur(m, (0, 0), 2)
        if self.prev is not None and self.prev.shape == m.shape:
            m = 0.6 * m + 0.4 * self.prev
        self.prev = m
        return m


def big_text_layer(text: str, size: tuple[int, int], font_loader, y_center: float, color: str = "#FFFFFF") -> np.ndarray:
    """Huge stacked title (RGBA, frame-sized) meant to sit behind the speaker."""
    w, h = size
    words = text.upper().split() or ["WATCH"]

    def split(n: int) -> list[str]:
        per = -(-len(words) // n)
        return [" ".join(words[i:i + per]) for i in range(0, len(words), per)]

    def fit_size(lines: list[str]) -> int:
        size_ = 340
        while size_ > 60 and max(font_loader(size_).getlength(l) for l in lines) > w * 0.92:
            size_ -= 10
        return size_

    # Choose the 1-3 line layout that allows the biggest letters (that's what makes the effect read).
    lines = max((split(n) for n in range(1, min(3, len(words)) + 1)), key=fit_size)
    fsize = fit_size(lines)
    font = font_loader(fsize)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    line_h = int(fsize * 0.98)
    y = int(h * y_center) - line_h * len(lines) // 2
    for l in lines:
        lw = font.getlength(l)
        d.text(((w - lw) / 2, y), l, font=font, fill=color)
        y += line_h
    shadow = img.filter(ImageFilter.GaussianBlur(10))
    sh = np.array(shadow)
    sh[..., :3] = 0
    sh[..., 3] = (sh[..., 3] * 0.5).astype(np.uint8)
    base = Image.alpha_composite(Image.fromarray(sh), img)
    return np.array(base)


class TextBehind:
    """Applies the title between background and person for the first `duration` seconds."""

    def __init__(self, matte: PersonMatte, layer: np.ndarray, duration: float, fade: float = 0.25):
        self.matte = matte
        self.rgb = layer[..., :3].astype(np.float32)
        self.alpha = layer[..., 3:4].astype(np.float32) / 255.0
        self.duration = duration
        self.fade = fade

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        if t >= self.duration:
            return frame
        k = min(1.0, t / self.fade, (self.duration - t) / self.fade)
        if k <= 0:
            return frame
        f = frame.astype(np.float32)
        a = self.alpha * k
        out = f * (1 - a) + self.rgb * a
        m = self.matte.mask(frame)[..., None]
        out = out * (1 - m) + f * m  # person back in front of the text
        return out.astype(np.uint8)
