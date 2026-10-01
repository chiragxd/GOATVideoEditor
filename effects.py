"""
Effects engine for Goat Reel Editor.

- Sound effects synthesized with numpy (no downloads, no licensing): pop, click, whoosh, swoosh, ding, boom, riser.
- Face tracking (OpenCV Haar cascade) so zooms and crops follow the speaker.
- Story-driven virtual camera: slow push, pull back, punch-in on emphasis, shake, zoom & whip transitions.
- Emoji pop-ins (Windows Segoe UI Emoji color font), hook title card, flash transitions.
- Audio mixer: speech + SFX + optional music with automatic ducking under speech.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SR = 48000
WIN_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
EMOJI_FONT = WIN_FONTS / "seguiemj.ttf"

# --------------------------------------------------------------------------------------
# Sound effects
# --------------------------------------------------------------------------------------

_rng = np.random.default_rng(7)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / max(decay, 1e-4))


def _sweep_sine(f0: float, f1: float, dur: float, curve: str = "exp") -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    if curve == "exp":
        f = f0 * (f1 / f0) ** (t / dur)
    else:
        f = f0 + (f1 - f0) * t / dur
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase)


def _swept_lowpass_noise(dur: float, cut_start: float, cut_peak: float, cut_end: float) -> np.ndarray:
    n = int(dur * SR)
    noise = _rng.standard_normal(n).astype(np.float32)
    x = np.linspace(0, 1, n)
    cutoff = np.where(x < 0.5, cut_start + (cut_peak - cut_start) * (x / 0.5),
                      cut_peak + (cut_end - cut_peak) * ((x - 0.5) / 0.5))
    alpha = 1 - np.exp(-2 * np.pi * cutoff / SR)
    out = np.empty(n, dtype=np.float32)
    y = 0.0
    for i in range(n):
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    return out


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * peak).astype(np.float32)


def synth_sfx(name: str) -> np.ndarray:
    if name == "pop":
        s = _sweep_sine(900, 250, 0.09) * _env(int(0.09 * SR), 0.002, 0.03)
    elif name == "click":
        n = int(0.035 * SR)
        s = (_rng.standard_normal(n) * 0.4 + np.sin(2 * np.pi * 2200 * np.arange(n) / SR)) * _env(n, 0.0005, 0.006)
    elif name == "whoosh":
        n = int(0.5 * SR)
        s = _swept_lowpass_noise(0.5, 300, 3500, 400) * np.hanning(n) ** 1.5
    elif name == "swoosh":
        n = int(0.28 * SR)
        s = _swept_lowpass_noise(0.28, 800, 6000, 1200) * np.hanning(n)
    elif name == "ding":
        n = int(1.3 * SR)
        t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * 1318.5 * t) + 0.45 * np.sin(2 * np.pi * 2637 * t)
             + 0.2 * np.sin(2 * np.pi * 3955.5 * t)) * _env(n, 0.002, 0.35)
    elif name == "boom":
        n = int(1.1 * SR)
        body = _sweep_sine(90, 32, 1.1) * _env(n, 0.004, 0.35)
        thump = _swept_lowpass_noise(1.1, 400, 200, 60) * _env(n, 0.001, 0.08) * 2.5
        s = np.tanh((body + thump) * 1.8)
    elif name == "riser":
        n = int(1.2 * SR)
        ramp = np.linspace(0, 1, n) ** 2
        s = (_swept_lowpass_noise(1.2, 300, 3000, 8000) * 0.8 + _sweep_sine(180, 1400, 1.2) * 0.35) * ramp
        s[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    else:
        return np.zeros(1, dtype=np.float32)
    return _norm(np.asarray(s, dtype=np.float32))


SFX_GAIN = {"pop": 0.45, "click": 0.35, "whoosh": 0.4, "swoosh": 0.35, "ding": 0.3, "boom": 0.55, "riser": 0.3}
SFX_LEAD = {"riser": 1.2, "whoosh": 0.2, "swoosh": 0.12}  # seconds the sound starts before the event


# --------------------------------------------------------------------------------------
# Pro voice processing: EQ, de-esser, loudness
# --------------------------------------------------------------------------------------

def _shelf_gain(freqs: np.ndarray, center: float, gain_db: float, width_oct: float) -> np.ndarray:
    """Smooth bell-shaped gain curve in dB on a log-frequency axis."""
    with np.errstate(divide="ignore"):
        octs = np.log2(np.maximum(freqs, 1.0) / center)
    return gain_db * np.exp(-0.5 * (octs / (width_oct / 2.355)) ** 2)


def voice_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Broadcast voice EQ: cut mud (250 Hz), add presence (3 kHz) and air (10 kHz)."""
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    gain_db = (_shelf_gain(freqs, 250, -2.5, 1.5) + _shelf_gain(freqs, 3000, 3.0, 1.6)
               + np.clip((freqs - 7000) / 5000, 0, 1) * 2.0)
    return np.fft.irfft(spec * 10 ** (gain_db / 20), n=audio.size).astype(np.float32)


def deess(audio: np.ndarray, sr: int, threshold: float = 0.35, max_cut_db: float = 8.0) -> np.ndarray:
    """Dynamic de-esser: turns down 5-9 kHz only in frames where sibilance dominates."""
    from scipy.signal import istft, stft

    f, _, z = stft(audio, fs=sr, nperseg=1024)
    band = (f >= 5000) & (f <= 9000)
    mag = np.abs(z)
    ratio = mag[band].sum(axis=0) / (mag.sum(axis=0) + 1e-9)
    over = np.clip((ratio - threshold) / threshold, 0, 1)
    gain = 10 ** (-(over * max_cut_db) / 20)
    z[band] *= gain[None, :]
    _, out = istft(z, fs=sr, nperseg=1024)
    out = out[: audio.size]
    if out.size < audio.size:
        out = np.pad(out, (0, audio.size - out.size))
    return out.astype(np.float32)


def loudness_normalize(stereo: np.ndarray, sr: int, target_lufs: float = -14.0, ceiling_db: float = -1.0) -> np.ndarray:
    """Normalize integrated loudness to the social-media standard (-14 LUFS) with a peak ceiling."""
    import pyloudnorm as pyln

    if stereo.shape[0] < sr * 0.5:
        return stereo
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(stereo)
    if not np.isfinite(loud):
        return stereo
    out = stereo * 10 ** ((target_lufs - loud) / 20)
    ceiling = 10 ** (ceiling_db / 20)
    over = np.abs(out) > ceiling * 0.8
    if over.any():  # soft-knee limiter only above 80% of the ceiling
        knee = ceiling * 0.8
        s = np.sign(out)
        a = np.abs(out)
        out = np.where(a > knee, s * (knee + (ceiling - knee) * np.tanh((a - knee) / (ceiling - knee))), out)
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Music library & B-roll library (local folders)
# --------------------------------------------------------------------------------------

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

MOOD_TO_MUSIC = {
    "excited": ["energetic", "upbeat"], "confident": ["energetic", "upbeat"], "happy": ["upbeat", "funny"],
    "funny": ["funny", "upbeat"], "curious": ["chill", "upbeat"], "calm": ["chill", "emotional"],
    "serious": ["dramatic", "chill"], "surprised": ["dramatic", "energetic"], "angry": ["dramatic", "energetic"],
    "emotional": ["emotional", "chill"],
}


def pick_music(library: Path, mood: str, seed: str) -> Path | None:
    """Pick a track for the mood from library/<category>/; falls back to any track in the library."""
    import random

    rnd = random.Random(seed)
    for cat in MOOD_TO_MUSIC.get(mood, ["chill"]):
        files = sorted(p for p in (library / cat).glob("*") if p.suffix.lower() in AUDIO_EXT)
        if files:
            return rnd.choice(files)
    files = sorted(p for p in library.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    return rnd.choice(files) if files else None


def _tokens(text: str) -> set[str]:
    import re

    return {t for t in re.split(r"[^a-z0-9\u0900-\u097f\u0a80-\u0aff]+", text.lower()) if len(t) > 2}


def find_broll(library: Path, idea: str, line_text: str, used: set[Path]) -> Path | None:
    """Match a B-roll file by filename keywords (e.g. 'money cash rupees.mp4') against the idea and line."""
    files = [p for p in library.rglob("*") if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT and p not in used]
    if not files:
        return None
    want_idea, want_line = _tokens(idea), _tokens(line_text)
    best, best_score = None, 0.0
    for p in files:
        name = _tokens(p.stem.replace("_", " ").replace("-", " ")) | _tokens(p.parent.name if p.parent != library else "")
        score = 2.0 * len(name & want_idea) + 1.0 * len(name & want_line)
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 1.0 else None


# --------------------------------------------------------------------------------------
# Audio mixing
# --------------------------------------------------------------------------------------

def load_music(path: Path, n: int) -> np.ndarray:
    from moviepy import AudioFileClip

    clip = AudioFileClip(str(path))
    try:
        arr = clip.to_soundarray(fps=SR)
    finally:
        clip.close()
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    if arr.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float32)
    reps = int(math.ceil(n / arr.shape[0]))
    return np.tile(arr, (reps, 1))[:n].astype(np.float32)


def speech_mask(words: Sequence, n: int, hold: float = 0.25) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float32)
    for w in words:
        a, b = int(max(0, w.start - 0.05) * SR), int(min(n / SR, w.end + hold) * SR)
        mask[a:b] = 1.0
    # Smooth to avoid pumping.
    k = int(0.15 * SR)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(mask, kernel, mode="same")


def mix_audio(speech: np.ndarray, events: list[tuple[float, str]], words: Sequence,
              music_path: Path | None, sfx_volume: float = 1.0, music_volume: float = 0.18) -> np.ndarray:
    if speech.ndim == 1:
        speech = np.stack([speech, speech], axis=1)
    n = speech.shape[0]
    out = speech.astype(np.float32).copy()
    cache: dict[str, np.ndarray] = {}
    for t, name in events:
        if name not in SFX_GAIN:
            continue
        s = cache.setdefault(name, synth_sfx(name)) * SFX_GAIN[name] * sfx_volume
        start = int(max(0.0, t - SFX_LEAD.get(name, 0.0)) * SR)
        end = min(n, start + s.size)
        if end > start:
            out[start:end] += s[: end - start, None]
    if music_path:
        music = load_music(music_path, n)
        duck = 1.0 - 0.7 * speech_mask(words, n)  # music drops to 30% under speech
        fade = np.ones(n, dtype=np.float32)
        f = min(n, int(1.5 * SR))
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.minimum(fade[-f:], np.linspace(1, 0, f))
        out += music * (music_volume * duck * fade)[:, None]
    return loudness_normalize(out, SR)


# --------------------------------------------------------------------------------------
# Face tracking
# --------------------------------------------------------------------------------------

@dataclass
class FaceTrack:
    times: np.ndarray
    cx: np.ndarray  # normalized 0..1
    cy: np.ndarray

    def at(self, t: float) -> tuple[float, float]:
        if self.times.size == 0:
            return 0.5, 0.42
        return float(np.interp(t, self.times, self.cx)), float(np.interp(t, self.times, self.cy))


def track_faces(clip, step: float = 0.25) -> FaceTrack:
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    times, xs, ys = [], [], []
    last = (0.5, 0.42)
    t = 0.0
    while t < clip.duration:
        frame = clip.get_frame(t)
        h, w = frame.shape[:2]
        scale = 360.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            last = ((fx + fw / 2) / small.shape[1], (fy + fh * 0.6) / small.shape[0])
        times.append(t)
        xs.append(last[0])
        ys.append(last[1])
        t += step
    if not times:
        return FaceTrack(np.array([]), np.array([]), np.array([]))
    # Heavy smoothing -> calm, professional camera movement.
    k = 7
    pad = lambda a: np.pad(np.array(a), (k // 2, k // 2), mode="edge")
    kernel = np.ones(k) / k
    return FaceTrack(np.array(times), np.convolve(pad(xs), kernel, "valid"), np.convolve(pad(ys), kernel, "valid"))


# --------------------------------------------------------------------------------------
# Virtual camera
# --------------------------------------------------------------------------------------

def _ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


@dataclass
class CamLine:
    start: float
    end: float
    camera: str
    transition: str
    energy: int
    hit: float  # time of the emphasized word (for punch/shake)


class VirtualCamera:
    """Computes zoom/offset per time from the edit plan, then applies it with one warpAffine per frame."""

    def __init__(self, lines: list[CamLine], face: FaceTrack | None, duration: float, intensity: float = 1.0):
        self.lines = lines
        self.face = face
        self.duration = duration
        self.intensity = intensity  # autopilot scales all moves (light for close-ups, stronger for wide shots)

    def _line_at(self, t: float) -> tuple[int, CamLine | None]:
        for k, l in enumerate(self.lines):
            nxt = self.lines[k + 1].start if k + 1 < len(self.lines) else self.duration
            if l.start <= t < nxt:
                return k, l
        return -1, None

    def state(self, t: float) -> tuple[float, float, float, float]:
        """Return zoom, shake_x(px frac), shake_y, whip_amount."""
        k, l = self._line_at(t)
        if l is None:
            return 1.0, 0.0, 0.0, 0.0
        base = 1.0 if k % 2 == 0 else 1.06  # alternate framing = jump-cut feel (a hard change at the cut)
        span = max(0.3, l.end - l.start)
        p = (t - l.start) / span
        zoom = base
        if l.camera == "slow_push":
            zoom = base + 0.12 * min(max(p, 0), 1)
        elif l.camera == "pull_back":
            zoom = base + 0.14 * (1 - min(max(p, 0), 1))
        elif l.camera in ("punch", "shake") and t >= l.hit:
            zoom = base + (0.14 + 0.02 * l.energy) * _ease_out((t - l.hit) / 0.09)

        sx = sy = 0.0
        if l.camera == "shake" and l.hit <= t < l.hit + 0.45:
            d = t - l.hit
            amp = 0.012 * math.exp(-d / 0.15)
            sx = amp * math.sin(2 * math.pi * 17 * d)
            sy = amp * math.cos(2 * math.pi * 13 * d)

        whip = 0.0
        d = t - l.start
        if 0 <= d < 0.3 and l.transition == "zoom":
            zoom += 0.3 * (1 - _ease_out(d / 0.3))
        if 0 <= d < 0.16 and l.transition == "whip":
            whip = 1 - d / 0.16
        k = self.intensity
        return 1 + (zoom - 1) * k, sx * k, sy * k, whip


# --------------------------------------------------------------------------------------
# Graphics
# --------------------------------------------------------------------------------------

def emoji_image(emoji: str, size: int = 160) -> np.ndarray | None:
    if not EMOJI_FONT.exists() or not emoji:
        return None
    font = ImageFont.truetype(str(EMOJI_FONT), 109)  # Segoe emoji renders crisply at 109, then scale
    probe = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    ImageDraw.Draw(probe).text((20, 20), emoji, font=font, embedded_color=True)
    bbox = probe.getbbox()
    if not bbox:
        return None
    img = probe.crop(bbox)
    scale = size / max(img.size)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    # Soft shadow.
    pad = 20
    canvas = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 140), (pad + 6, pad + 8), img)
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(8)))
    canvas.alpha_composite(img, (pad, pad))
    return np.array(canvas)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:3]


def _load(font, size: int) -> ImageFont.FreeTypeFont:
    """`font` is a path or a callable size -> font (lets the app pass variable-weight fonts)."""
    return font(size) if callable(font) else ImageFont.truetype(str(font), size)


def title_card_image(text: str, accent: str, font_path: Path, width: int = 1080) -> np.ndarray:
    size = 78
    font = _load(font_path, size)
    lines = _wrap(text.upper(), font, width - 240)
    while len(lines) > 2 and size > 50:
        size -= 6
        font = _load(font_path, size)
        lines = _wrap(text.upper(), font, width - 240)
    asc, desc = font.getmetrics()
    line_h = asc + desc + 8
    tw = int(max(font.getlength(l) for l in lines))
    box_w, box_h = tw + 80, line_h * len(lines) + 50
    img = Image.new("RGBA", (box_w + 40, box_h + 40), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((26, 30, 26 + box_w, 30 + box_h), 28, fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((20, 20, 20 + box_w, 20 + box_h), 28, fill=(255, 255, 255, 245))
    d.rounded_rectangle((20, 20, 32, 20 + box_h), 6, fill=accent)  # accent stripe
    y = 20 + 25
    for l in lines:
        lw = font.getlength(l)
        d.text((20 + (box_w - lw) / 2, y), l, font=font, fill=(15, 15, 15))
        y += line_h
    return np.array(img.rotate(-2, resample=Image.BICUBIC, expand=True))


def logo_image(path: Path, width: int = 150, opacity: float = 0.9) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    scale = width / img.width
    img = img.resize((width, max(1, int(img.height * scale))), Image.LANCZOS)
    alpha = np.array(img)[..., 3].astype(np.float32) * opacity
    arr = np.array(img)
    arr[..., 3] = alpha.astype(np.uint8)
    return arr


def end_card_image(line1: str, line2: str, accent: str, font_path: Path, width: int = 1080) -> np.ndarray:
    """Call-to-action card, e.g. 'FOLLOW @handle' / 'for more'."""
    l1, l2 = line1.upper(), line2
    size = 84
    big = _load(font_path, size)
    while big.getlength(l1) > width - 260 and size > 40:  # keep clear of the screen edges
        size -= 4
        big = _load(font_path, size)
    small = _load(font_path, max(30, int(size * 0.55)))
    w1 = big.getlength(l1)
    w2 = small.getlength(l2) if l2 else 0
    box_w = int(max(w1, w2)) + 110
    box_h = 84 + (70 if l2 else 0) + 80
    img = Image.new("RGBA", (box_w + 40, box_h + 40), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((26, 30, 26 + box_w, 30 + box_h), 36, fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(12)))
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((20, 20, 20 + box_w, 20 + box_h), 36, fill=accent)
    dr.text((20 + (box_w - w1) / 2, 50), l1, font=big, fill=(15, 15, 15))
    if l2:
        dr.text((20 + (box_w - w2) / 2, 50 + 100), l2, font=small, fill=(15, 15, 15))
    return np.array(img)


# --------------------------------------------------------------------------------------
# Fast frame pipeline: camera + crop + scale to 1080x1920 in a single OpenCV warp
# --------------------------------------------------------------------------------------

class FrameComposer:
    """Turns a source frame into the final 9:16 base frame in one affine warp per frame.

    mode "crop": a 9:16 window (following the face) is cut from the source and scaled to the output.
    mode "blur": the whole frame is fitted to the output width over a blurred, darkened copy of itself.
    The virtual camera's zoom/shake/whip are folded into the same warp, so no extra resizes happen.
    """

    def __init__(self, camera: "VirtualCamera | None", face: FaceTrack | None, mode: str,
                 out_size: tuple[int, int] = (1080, 1920)):
        self.camera = camera
        self.face = face
        self.mode = mode
        self.out_w, self.out_h = out_size

    def _state(self, t: float) -> tuple[float, float, float, float]:
        return self.camera.state(t) if self.camera else (1.0, 0.0, 0.0, 0.0)

    def _crop_warp(self, frame: np.ndarray, t: float, zoom: float, sx: float, sy: float, whip: float) -> np.ndarray:
        h, w = frame.shape[:2]
        target = self.out_w / self.out_h
        crop_w, crop_h = (h * target, h) if w / h > target else (w, w / target)
        view_w, view_h = crop_w / zoom, crop_h / zoom
        fx, fy = self.face.at(t) if self.face else (0.5, 0.42)
        cx = min(max(fx * w, view_w / 2), w - view_w / 2)
        cy = min(max(fy * h, view_h / 2), h - view_h / 2)
        s = self.out_w / view_w
        tx = self.out_w / 2 - s * cx + (sx + whip * 0.25) * self.out_w
        ty = self.out_h / 2 - s * cy + sy * self.out_h
        m = np.float32([[s, 0, tx], [0, s, ty]])
        return cv2.warpAffine(frame, m, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _blur_compose(self, frame: np.ndarray, t: float, zoom: float, sx: float, sy: float, whip: float) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, self.out_w // 10), max(1, self.out_h // 10)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        bg = (cv2.resize(small, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR) * 0.55).astype(np.uint8)
        s = self.out_w / w * zoom
        fg_h = int(round(h * self.out_w / w))
        fx, fy = self.face.at(t) if self.face else (0.5, 0.5)
        cx, cy = w / 2 + (fx - 0.5) * w * (1 - 1 / zoom), h / 2 + (fy - 0.5) * h * (1 - 1 / zoom)
        tx = self.out_w / 2 - s * cx + (sx + whip * 0.25) * self.out_w
        ty = fg_h / 2 - s * cy + sy * self.out_h
        fg = cv2.warpAffine(frame, np.float32([[s, 0, tx], [0, s, ty]]), (self.out_w, fg_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        y0 = (self.out_h - fg_h) // 2
        bg[y0:y0 + fg_h] = fg
        return bg

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        zoom, sx, sy, whip = self._state(t)
        render = self._crop_warp if self.mode == "crop" else self._blur_compose
        out = render(frame, t, zoom, sx, sy, whip)
        if self.camera is not None:
            z_prev = self._state(max(0.0, t - 1 / 60))[0]
            if abs(zoom - z_prev) > 0.004:  # motion blur while the zoom moves fast
                acc = out.astype(np.float32)
                for k in (1, 2):
                    acc += render(frame, t, zoom + (z_prev - zoom) * k / 2, sx, sy, whip).astype(np.float32)
                out = (acc / 3).astype(np.uint8)
        if whip > 0:
            k = max(3, int(whip * self.out_w * 0.12) | 1)
            out = cv2.blur(out, (k, 1))
        return out


# --------------------------------------------------------------------------------------
# Multi-speaker: follow whoever is talking (mouth-motion active speaker detection)
# --------------------------------------------------------------------------------------

def track_active_speaker(clip, words: Sequence, step: float = 0.2, hold: float = 1.2) -> tuple[FaceTrack, int]:
    """Find people (face clusters by horizontal position) and, while someone speaks, pick the face whose
    mouth region moves most. Returns a smoothed FaceTrack that cuts between speakers, and the people count."""
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    samples = []  # (t, [(cx, cy, w, mouth_gray)])
    t = 0.0
    while t < clip.duration:
        frame = clip.get_frame(t)
        h, w = frame.shape[:2]
        scale = 480.0 / max(h, w)
        gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_RGB2GRAY))
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        dets = []
        for fx, fy, fw, fh in faces:
            mouth = gray[fy + int(fh * 0.65): fy + fh, fx + fw // 4: fx + 3 * fw // 4]
            if mouth.size:
                dets.append(((fx + fw / 2) / gray.shape[1], (fy + fh * 0.6) / gray.shape[0], fw / gray.shape[1],
                             cv2.resize(mouth, (32, 16))))
        samples.append((t, dets))
        t += step

    xs = sorted(d[0] for _, dets in samples for d in dets)
    if not xs:
        return FaceTrack(np.array([]), np.array([]), np.array([])), 0
    # 1-D clustering of face x-positions: split where consecutive positions jump > 0.12.
    centers, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > 0.12:
            centers.append(float(np.mean(cur)))
            cur = []
        cur.append(x)
    centers.append(float(np.mean(cur)))
    n = len(samples)
    people = [c for c in centers if sum(1 for _, ds in samples if any(abs(d[0] - c) < 0.1 for d in ds)) > 0.3 * n]
    if len(people) < 2:
        return track_faces(clip), len(people)

    speaking = lambda tt: any(w.start - 0.1 <= tt <= w.end + 0.1 for w in words)  # noqa: E731
    activity = np.zeros((n, len(people)))
    pos = np.tile(np.array([[c, 0.42] for c in people]), (n, 1, 1)).astype(float)
    prev_mouth: dict[int, np.ndarray] = {}
    for i, (tt, dets) in enumerate(samples):
        for d in dets:
            k = int(np.argmin([abs(d[0] - c) for c in people]))
            pos[i, k] = (d[0], d[1])
            if k in prev_mouth and speaking(tt):
                activity[i, k] = float(np.mean(cv2.absdiff(d[3], prev_mouth[k])))
            prev_mouth[k] = d[3]
    # Smooth activity over ~1s and choose the active speaker with a minimum hold time.
    win = max(1, int(1.0 / step))
    kernel = np.ones(win) / win
    smooth = np.stack([np.convolve(activity[:, k], kernel, "same") for k in range(len(people))], axis=1)
    active, last_switch, cur_k = [], -1e9, int(np.argmax(smooth[0])) if n else 0
    for i in range(n):
        best = int(np.argmax(smooth[i]))
        if best != cur_k and smooth[i, best] > smooth[i, cur_k] * 1.25 and samples[i][0] - last_switch >= hold:
            cur_k, last_switch = best, samples[i][0]
        active.append(cur_k)
    times = np.array([s[0] for s in samples])
    cx = np.array([pos[i, k, 0] for i, k in enumerate(active)])
    cy = np.array([pos[i, k, 1] for i, k in enumerate(active)])
    return FaceTrack(times, cx, cy), len(people)
