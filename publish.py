"""
Publishing kit: everything needed after the render.

- Cover / thumbnail: picks the best frame (face visible, sharp, eyes open, expressive) and adds a title.
- Post copy: Instagram caption + hashtags, YouTube Shorts title/description (local LLM, rules fallback).
- Subtitles: SRT / VTT of the final timeline, plus an English translation for Hindi/Gujlish videos.
- Pro handoff: Final Cut Pro 7 XML (imports into Premiere Pro and DaVinci Resolve) with every cut.
- Multi-format: 1:1, 4:5 and 16:9 versions derived from the 9:16 master.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Sequence
from xml.sax.saxutils import escape

import cv2
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------------------


def _ts(t: float, sep: str = ",") -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d}{sep}{ms % 1000:03d}"


def caption_blocks(words: Sequence, max_words: int = 7, max_gap: float = 0.6) -> list[tuple[float, float, str]]:
    """Readable subtitle blocks (longer than on-screen captions) for SRT/VTT."""
    blocks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w.start - cur[-1].end > max_gap):
            blocks.append(cur)
            cur = []
        cur.append(w)
        if w.text.rstrip().endswith((".", "?", "!", "।")):
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return [(b[0].start, b[-1].end, " ".join(x.text.strip() for x in b)) for b in blocks]


def write_srt(blocks: list[tuple[float, float, str]], path: Path) -> Path:
    lines = []
    for k, (s, e, text) in enumerate(blocks, 1):
        lines += [str(k), f"{_ts(s)} --> {_ts(max(e, s + 0.3))}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_vtt(blocks: list[tuple[float, float, str]], path: Path) -> Path:
    lines = ["WEBVTT", ""]
    for s, e, text in blocks:
        lines += [f"{_ts(s, '.')} --> {_ts(max(e, s + 0.3), '.')}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def translate_blocks(blocks: list[tuple[float, float, str]], llm_json: Callable, model: str,
                     log: Callable[[str], None]) -> list[tuple[float, float, str]] | None:
    """Translate subtitle blocks to natural English with the local LLM (same timings)."""
    if not blocks:
        return None
    schema = {"type": "object", "properties": {"lines": {"type": "array", "items": {
        "type": "object", "properties": {"i": {"type": "integer"}, "en": {"type": "string"}},
        "required": ["i", "en"]}}}, "required": ["lines"]}
    out = [b for b in blocks]
    full = "\n".join(f"{k + 1}. {b[2]}" for k, b in enumerate(blocks))
    for start in range(0, len(blocks), 20):
        chunk = range(start, min(len(blocks), start + 20))
        log(f"Translating subtitles {chunk[0] + 1}-{chunk[-1] + 1} to English...")
        try:
            res = llm_json(model, "You translate Hindi/Hinglish/Gujarati/Gujlish social-media speech into natural, "
                                  "short English subtitles. Keep meaning and tone, no extra words, no emoji.",
                           f"Full transcript:\n{full}\n\nTranslate lines {chunk[0] + 1}-{chunk[-1] + 1}.", schema,
                           max_tokens=1500)
        except Exception as exc:
            log(f"Translation failed ({exc}).")
            return None
        for item in res.get("lines", []):
            try:
                k = int(item["i"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if k in chunk and str(item.get("en", "")).strip():
                s, e, _ = blocks[k]
                out[k] = (s, e, str(item["en"]).strip())
    return out


# --------------------------------------------------------------------------------------
# Post copy
# --------------------------------------------------------------------------------------

POST_SCHEMA = {
    "type": "object",
    "properties": {
        "instagram_caption": {"type": "string", "maxLength": 600},
        "hashtags": {"type": "array", "items": {"type": "string", "maxLength": 40}, "minItems": 5, "maxItems": 15},
        "youtube_title": {"type": "string", "maxLength": 100},
        "youtube_description": {"type": "string", "maxLength": 400},
    },
    "required": ["instagram_caption", "hashtags", "youtube_title", "youtube_description"],
}


def post_copy(transcript: str, summary: str, content_type: str, handle: str, llm_json: Callable | None,
              model: str, log: Callable[[str], None]) -> dict:
    if llm_json:
        try:
            log("Writing the post caption, hashtags and YouTube title...")
            res = llm_json(
                model,
                "You are a social media manager for Indian creators. Write in the same language style as the "
                "speaker (English, Hinglish or Gujlish in Roman script). Instagram caption: a scroll-stopping first "
                "line, 2-3 short lines, one call to action, max 2 emoji. Hashtags: 10-15 relevant ones mixing broad "
                "and niche, no spaces, start with #. YouTube Shorts title under 60 characters ending with #shorts. "
                "Description: 2 sentences.",
                f"Content type: {content_type}\nSummary: {summary}\nCreator handle: {handle or 'n/a'}\n"
                f"Transcript:\n{transcript[:4000]}", POST_SCHEMA, max_tokens=700)
            import html
            import re

            res["hashtags"] = [("#" + h.lstrip("#").replace(" ", "")) for h in res.get("hashtags", []) if h.strip()]
            for k in ("instagram_caption", "youtube_title", "youtube_description"):
                res[k] = html.unescape(str(res.get(k, ""))).strip()
            # Hashtags are listed separately; keep the caption itself clean.
            res["instagram_caption"] = re.sub(r"(\s*#\w+)+\s*$", "", res["instagram_caption"]).strip()
            return res
        except Exception as exc:
            log(f"Post copy AI failed ({exc}); using a simple template.")
    first = transcript.split(".")[0][:90]
    return {"instagram_caption": f"{first}...\n\nFollow {handle} for more!".strip(),
            "hashtags": ["#reels", "#reelsindia", "#instagood", "#trending", "#viral"],
            "youtube_title": (first[:50] + " #shorts"), "youtube_description": summary or first}


def write_post_copy(copy: dict, path: Path) -> Path:
    text = (f"INSTAGRAM CAPTION\n{copy['instagram_caption']}\n\n{' '.join(copy['hashtags'])}\n\n"
            f"YOUTUBE SHORTS TITLE\n{copy['youtube_title']}\n\nYOUTUBE DESCRIPTION\n{copy['youtube_description']}\n")
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Cover / thumbnail
# --------------------------------------------------------------------------------------

def _cascade(name: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))


def best_frame_time(clip, step: float = 0.4, min_t: float = 0.5) -> float:
    """Score frames: face present & large, eyes open, smile, sharpness; avoid the very start/end."""
    face_c, eye_c, smile_c = (_cascade("haarcascade_frontalface_default.xml"), _cascade("haarcascade_eye.xml"),
                              _cascade("haarcascade_smile.xml"))
    best_t, best = clip.duration / 3, -1.0
    t = min(min_t, clip.duration / 2)
    while t < clip.duration - 0.3:
        frame = clip.get_frame(t)
        h, w = frame.shape[:2]
        scale = 480 / max(h, w)
        gray = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_RGB2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(sharp / 300.0, 1.0)
        faces = face_c.detectMultiScale(cv2.equalizeHist(gray), 1.1, 5, minSize=(30, 30))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            roi = gray[fy:fy + fh, fx:fx + fw]
            eyes = eye_c.detectMultiScale(roi[: fh // 2], 1.1, 6)
            smiles = smile_c.detectMultiScale(roi[fh // 2:], 1.6, 18)
            score += 2.0 + 1.5 * (fw / gray.shape[1]) + 0.8 * min(len(eyes), 2) + 1.2 * (len(smiles) > 0)
        if score > best:
            best, best_t = score, t
        t += step
    return best_t


def make_cover(clip, title: str, accent: str, font_loader: Callable[[int], ImageFont.FreeTypeFont],
               path: Path, size: tuple[int, int] = (1080, 1920), min_t: float = 0.5) -> Path:
    t = best_frame_time(clip, min_t=min_t)
    frame = Image.fromarray(clip.get_frame(t)).convert("RGB")
    frame = frame.resize(size, Image.LANCZOS) if frame.size != size else frame
    # Subtle darkening at the top so the title pops.
    grad = Image.new("L", (1, size[1]))
    for y in range(size[1]):
        grad.putpixel((0, y), int(150 * max(0.0, 1 - y / (size[1] * 0.45))))
    shade = Image.new("RGB", size, (0, 0, 0))
    frame = Image.composite(shade, frame, grad.resize(size))
    draw = ImageDraw.Draw(frame)
    text = title.upper().strip() or "WATCH THIS"
    fsize = 150
    font = font_loader(fsize)
    words, lines = text.split(), []
    while True:
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if font.getlength(trial) <= size[0] - 140 or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        if len(lines) <= 3 or fsize <= 70:
            break
        fsize -= 10
        font = font_loader(fsize)
    y = 170
    for k, line in enumerate(lines):
        lw = font.getlength(line)
        color = accent if k == len(lines) - 1 else "#FFFFFF"
        draw.text(((size[0] - lw) / 2, y), line, font=font, fill=color, stroke_width=max(6, fsize // 14),
                  stroke_fill="#000000")
        y += int(fsize * 1.12)
    frame.save(path, quality=95)
    return path


# --------------------------------------------------------------------------------------
# Multi-format versions
# --------------------------------------------------------------------------------------

def aspect_versions(master: Path, ffmpeg: str, formats: Sequence[str], log: Callable[[str], None]) -> list[Path]:
    """Derive 1:1, 4:5 and 16:9 versions from the 9:16 master (fast re-encode, audio copied)."""
    filters = {
        "1x1": "crop=1080:1080:0:(ih-1080)*0.55",
        "4x5": "crop=1080:1350:0:(ih-1350)*0.5",
        "16x9": ("split[a][b];[a]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=30:5,"
                 "eq=brightness=-0.15[bg];[b]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:0"),
    }
    out = []
    for f in formats:
        if f not in filters:
            continue
        dst = master.with_name(master.stem + f"_{f}.mp4")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(master), "-filter_complex", filters[f],
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-movflags", "+faststart", str(dst)]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            out.append(dst)
            log(f"Exported {f.replace('x', ':')} version: {dst.name}")
    return out


# --------------------------------------------------------------------------------------
# Pro handoff: FCP7 XML (Premiere Pro / DaVinci Resolve import)
# --------------------------------------------------------------------------------------

def write_fcp_xml(source: Path, segments: list[tuple[float, float]], fps: float, width: int, height: int,
                  duration: float, path: Path, name: str = "GOAT Studio edit") -> Path:
    """Sequence with one clip item per kept segment, in edit order, video + audio linked."""
    tb = int(round(fps)) or 30
    ntsc = "TRUE" if abs(fps - round(fps)) > 0.01 else "FALSE"
    f = lambda sec: int(round(sec * fps))  # noqa: E731
    src_url = "file://localhost/" + str(source.resolve()).replace("\\", "/").replace(" ", "%20")
    total_frames = f(duration)
    rate = f"<rate><timebase>{tb}</timebase><ntsc>{ntsc}</ntsc></rate>"
    file_el = (f'<file id="file-1"><name>{escape(source.name)}</name><pathurl>{escape(src_url)}</pathurl>{rate}'
               f"<duration>{total_frames}</duration><media><video><samplecharacteristics>"
               f"<width>{width}</width><height>{height}</height></samplecharacteristics></video>"
               "<audio><channelcount>2</channelcount></audio></media></file>")
    v_items, a_items, pos = [], [], 0
    for k, (s, e) in enumerate(segments, 1):
        length = f(e) - f(s)
        if length <= 0:
            continue
        fe = file_el if k == 1 else '<file id="file-1"/>'
        common = (f"<name>{escape(source.stem)} {k}</name><duration>{total_frames}</duration>{rate}"
                  f"<start>{pos}</start><end>{pos + length}</end><in>{f(s)}</in><out>{f(e)}</out>")
        v_items.append(f'<clipitem id="v{k}">{common}{fe}</clipitem>')
        a_items.append(f'<clipitem id="a{k}">{common}<file id="file-1"/><sourcetrack><mediatype>audio</mediatype>'
                       "<trackindex>1</trackindex></sourcetrack></clipitem>")
        pos += length
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4"><sequence id="seq-1">'
           f"<name>{escape(name)}</name><duration>{pos}</duration>{rate}<media>"
           f"<video><format><samplecharacteristics><width>{width}</width><height>{height}</height>"
           f"</samplecharacteristics></format><track>{''.join(v_items)}</track></video>"
           f"<audio><track>{''.join(a_items)}</track></audio></media></sequence></xmeml>")
    path.write_text(xml, encoding="utf-8")
    return path
