"""
Video understanding: what the footage looks like, so the editor can decide what to do (and what NOT to do).

- Frame scan (OpenCV, fast): face presence / size / position, camera motion, existing scene cuts,
  exposure, sharpness, orientation.
- Speech coverage from the transcript (talking video vs. music/b-roll video).
- Optional scene descriptions from a local vision model (Ollama, e.g. moondream).
- Autopilot: turns the insight + the director's story analysis into concrete edit settings, each with a
  human-readable reason.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Sequence

import cv2
import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434"
VISION_MODEL = "moondream"


@dataclass
class VideoInsight:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    orientation: str = "vertical"      # vertical | horizontal | square
    face_presence: float = 0.0         # share of sampled frames with a face
    face_size: float = 0.0             # median face width / frame width
    face_cx: float = 0.5               # median face center (normalized)
    face_cy: float = 0.42
    motion: float = 0.0                # mean frame-to-frame change (0..1); handheld/vlog ~>0.05
    scene_cuts: int = 0                # hard cuts already in the footage
    brightness: float = 0.5
    sharpness: float = 0.0             # variance of Laplacian (higher = sharper)
    speech_coverage: float = 0.0       # share of the runtime with speech
    words_per_min: float = 0.0
    scenes: list[str] = field(default_factory=list)  # vision-model descriptions of key frames

    @property
    def shot(self) -> str:
        if self.face_presence < 0.3:
            return "no-person" if self.motion < 0.05 else "action / b-roll"
        if self.face_size > 0.33:
            return "close-up talking head"
        if self.face_size > 0.18:
            return "medium talking head"
        return "wide shot with person"

    def describe(self) -> str:
        """Compact text the LLM director reads to 'see' the video."""
        parts = [
            f"{self.orientation} video, {self.duration:.0f}s",
            f"shot: {self.shot}",
            f"camera: {'handheld/moving' if self.motion > 0.05 else 'static'}",
            f"{self.scene_cuts} existing cuts" if self.scene_cuts else "single continuous take",
            "dark/underexposed" if self.brightness < 0.3 else "bright" if self.brightness > 0.65 else "normal exposure",
            f"speech covers {self.speech_coverage:.0%} of the runtime at {self.words_per_min:.0f} words/min",
        ]
        text = "; ".join(parts)
        if self.scenes:
            text += "\nWhat the camera sees: " + " | ".join(self.scenes)
        return text


# --------------------------------------------------------------------------------------
# Frame scan
# --------------------------------------------------------------------------------------

def scan_video(clip, words: Sequence = (), step: float = 0.5, log: Callable[[str], None] = print) -> VideoInsight:
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    w, h = clip.size
    ins = VideoInsight(duration=float(clip.duration), width=w, height=h,
                       orientation="vertical" if h > w * 1.1 else "horizontal" if w > h * 1.1 else "square")
    faces, sizes, motions, bright, sharp = [], [], [], [], []
    prev_small, prev_hist, cuts = None, None, 0
    t = 0.0
    while t < clip.duration:
        frame = clip.get_frame(t)
        scale = 320.0 / max(w, h)
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        bright.append(float(gray.mean()) / 255.0)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        found = cascade.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(found):
            fx, fy, fw, fh = max(found, key=lambda f: f[2] * f[3])
            faces.append(((fx + fw / 2) / small.shape[1], (fy + fh / 2) / small.shape[0]))
            sizes.append(fw / small.shape[1])
        if prev_small is not None:
            motions.append(float(np.mean(cv2.absdiff(gray, prev_small))) / 255.0)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.55:
            cuts += 1
        prev_small, prev_hist = gray, hist
        t += step

    n = max(1, len(bright))
    ins.face_presence = len(faces) / n
    if faces:
        ins.face_cx = float(np.median([f[0] for f in faces]))
        ins.face_cy = float(np.median([f[1] for f in faces]))
        ins.face_size = float(np.median(sizes))
    ins.motion = float(np.median(motions)) if motions else 0.0
    ins.scene_cuts = cuts
    ins.brightness = float(np.mean(bright)) if bright else 0.5
    ins.sharpness = float(np.median(sharp)) if sharp else 0.0
    if words:
        spoken = sum(max(0.0, w.end - w.start) for w in words)
        ins.speech_coverage = min(1.0, spoken / max(ins.duration, 0.1))
        ins.words_per_min = len(words) / max(ins.duration / 60, 0.1)
    log(f"Visual scan: {ins.shot}, face in {ins.face_presence:.0%} of frames (size {ins.face_size:.0%}), "
        f"motion {ins.motion:.3f}, {cuts} existing cuts, exposure {ins.brightness:.2f}")
    return ins


# --------------------------------------------------------------------------------------
# Vision model (what is happening on screen)
# --------------------------------------------------------------------------------------

def vision_available(model: str = VISION_MODEL) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return any(n.split(":")[0] == model.split(":")[0] for n in names)
    except Exception:
        return False


def describe_frames(clip, count: int = 4, model: str = VISION_MODEL,
                    log: Callable[[str], None] = print) -> list[str]:
    out = []
    times = np.linspace(clip.duration * 0.1, clip.duration * 0.9, count)
    for k, t in enumerate(times):
        frame = clip.get_frame(float(t))
        h, w = frame.shape[:2]
        scale = 512.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        body = {"model": model, "stream": False, "options": {"temperature": 0.1, "num_predict": 80},
                "prompt": "Describe this image.",  # moondream answers best to its native short prompt
                "images": [base64.b64encode(buf.tobytes()).decode()]}
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read()).get("response", "").strip().replace("\n", " ")
        except Exception as exc:
            log(f"Vision model failed on frame {k + 1} ({exc}); continuing without it.")
            break
        if text:
            out.append(f"[{t:.0f}s] {text[:160]}")
            log(f"Vision [{t:.0f}s]: {text[:160]}")
    return out


# --------------------------------------------------------------------------------------
# Autopilot: decide the edit from understanding
# --------------------------------------------------------------------------------------

CONTENT_TYPES = ["tips_tutorial", "storytime", "comedy", "motivational", "podcast_interview", "vlog",
                 "product_review", "opinion_rant", "educational", "other"]
PACES = ["calm", "medium", "fast"]

# Edit style per content type: (silence threshold, words/caption, caption style, emoji, sfx, transitions,
#                               hook reorder, camera intensity, music volume)
STYLE_BOOK = {
    "tips_tutorial":     (0.6, 3, "pop", 0.6, True, True, False, 0.9, 0.12),
    "educational":       (0.7, 3, "pop", 0.5, True, False, False, 0.8, 0.12),
    "storytime":         (0.7, 3, "auto", 0.8, True, True, True, 1.0, 0.16),
    "comedy":            (0.5, 2, "box", 1.0, True, True, True, 1.2, 0.18),
    "motivational":      (0.8, 3, "auto", 0.7, True, True, True, 1.0, 0.22),
    "podcast_interview": (1.0, 4, "pop", 0.3, False, False, False, 0.6, 0.0),
    "vlog":              (0.8, 3, "auto", 0.9, True, True, False, 0.7, 0.20),
    "product_review":    (0.6, 3, "pop", 0.6, True, True, True, 0.9, 0.14),
    "opinion_rant":      (0.5, 2, "box", 0.8, True, True, True, 1.2, 0.14),
    "other":             (0.8, 3, "auto", 0.7, True, True, True, 1.0, 0.16),
}


def autopilot(settings, insight: VideoInsight, content_type: str, pace: str, mood: str,
              broll_available: bool) -> tuple[object, list[str]]:
    """Return (new settings, decisions). Every decision explains *why* — the editor's reasoning."""
    ct = content_type if content_type in STYLE_BOOK else "other"
    thr, wpc, style, emoji_rate, sfx, trans, hook, cam, music_vol = STYLE_BOOK[ct]
    d: list[str] = [f"Content type **{ct.replace('_', ' ')}**, pace **{pace}**, mood **{mood}** → "
                    f"using the {ct.replace('_', ' ')} edit style."]

    # Pacing
    thr = {"fast": thr - 0.15, "calm": thr + 0.2}.get(pace, thr)
    thr = round(float(np.clip(thr, 0.4, 1.4)), 2)
    cut = insight.speech_coverage > 0.25
    d.append(f"Silence cuts {'at ' + str(thr) + 's' if cut else 'OFF'} — "
             + (f"{pace} delivery." if cut else "this is mostly non-speech footage; cutting gaps would break it."))

    # Camera
    face_ok = insight.face_presence >= 0.4
    if not face_ok:
        cam_on, cam = False, 0.0
        d.append("Camera zooms OFF — no consistent face on screen; zooming would crop the action.")
    else:
        cam_on = True
        if insight.face_size > 0.33:
            cam *= 0.55
            d.append(f"Light zooms — face already fills {insight.face_size:.0%} of the frame width.")
        elif insight.face_size < 0.15:
            cam *= 1.25
            d.append(f"Stronger punch-ins — face is small ({insight.face_size:.0%} of width), zoom brings it closer.")
        else:
            d.append("Normal punch-ins and pushes on emphasis.")
        if insight.motion > 0.06:
            cam *= 0.6
            d.append("Camera is already moving (handheld) → softer virtual camera to avoid motion sickness.")
    if insight.scene_cuts > max(3, insight.duration / 6):
        trans = False
        d.append(f"Footage already has {insight.scene_cuts} cuts → no extra flash/whip transitions.")

    # Reframe
    reframe = settings.reframe_mode
    if insight.orientation != "vertical":
        if face_ok and insight.face_size > 0.12:
            reframe = "Center crop"
            d.append("Horizontal source with a speaker → crop to 9:16 following the face.")
        else:
            reframe = "Blurred background"
            d.append("Horizontal scene without a clear speaker → keep the full frame over a blurred background.")

    # Order & story
    if not hook:
        d.append("Keeping the original order — this kind of content needs its sequence (no cold open).")

    # Graphics & sound
    if mood in ("emotional", "serious"):
        emoji_rate *= 0.4
        sfx_on = sfx
        d.append("Serious/emotional mood → minimal emoji, subtle sound design.")
    else:
        sfx_on = sfx
    if not sfx_on:
        d.append("No sound effects — they would distract in a conversation format.")
    if music_vol == 0.0:
        d.append("No background music — speech-only format.")
    if broll_available:
        d.append("B-roll library found → cutaways where the director suggests a visual.")

    # Grade
    grade_strength = 1.0
    if 0.38 <= insight.brightness <= 0.62 and insight.sharpness > 150:
        grade_strength = 0.5
        d.append("Footage is already well exposed and sharp → light color grade only.")
    elif insight.brightness < 0.3:
        d.append("Footage is dark → exposure lift + stronger grade.")

    # Caption placement: keep text off the face.
    caption_y = 0.66
    if face_ok and insight.orientation == "vertical":
        if insight.face_cy > 0.55:
            caption_y = 0.30
            d.append("Face sits low in frame → captions moved to the upper third.")
        elif insight.face_cy < 0.35:
            caption_y = 0.72
    d.append(f"Captions: {wpc} words at a time, style '{style}', placed at {caption_y:.0%} height.")

    # Stabilize only genuinely shaky handheld footage (it crops slightly).
    stabilize = insight.motion > 0.035 and insight.scene_cuts <= max(2, insight.duration / 10)
    d.append("Handheld shake detected → 2-pass stabilization." if stabilize else
             "Camera is steady → no stabilization (keeps full sharpness).")

    # Hook title: behind the speaker when there is a clear, steady person; otherwise a title card.
    behind = face_ok and insight.motion < 0.03 and insight.face_size >= 0.12 and ct in (
        "storytime", "motivational", "vlog", "comedy", "opinion_rant", "product_review", "other")
    title_style = "behind" if behind else "card"
    d.append("Hook title placed BEHIND the speaker (steady shot, clear subject)." if behind else
             "Hook title as a clean card on top.")

    new = replace(
        settings,
        stabilize=stabilize or settings.stabilize,
        title_style=title_style if settings.title_style == "auto" else settings.title_style,
        cut_silences=cut, silence_threshold=thr, words_per_caption=wpc, caption_style=style,
        camera_moves=cam_on, face_tracking=face_ok, camera_intensity=round(cam, 2), transitions=trans,
        hook_reorder=hook, sfx=sfx_on, emoji=emoji_rate > 0.2, emoji_rate=round(emoji_rate, 2),
        music_volume=music_vol, auto_music=music_vol > 0, reframe_mode=reframe, grade_strength=grade_strength,
        caption_y=caption_y, broll=broll_available,
    )
    return new, d


def insight_from_dict(d: dict | None) -> VideoInsight:
    return VideoInsight(**d) if d else VideoInsight()


def insight_to_dict(ins: VideoInsight) -> dict:
    return asdict(ins)
