"""
Goat Reel Editor - Local, free, automated Reels / Shorts editor.

Pipeline:
  A. Audio extraction -> denoise (DeepFilterNet, fallback noisereduce) -> compression -> normalization
  B. Whisper transcription (auto language detect, word-level timestamps, Hinglish/Gujlish biasing)
  C. Kinetic 2-3 word captions with keyword highlighting, stroke + drop shadow (Pillow rendered)
  D. Silence jump-cuts (> threshold) and 9:16 1080x1920 reframe (center crop or blurred stack)

Run:  python app.py   ->  opens http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import ImageFont

import gradio as gr
import soundfile as sf
import torch
import whisper
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)

import analyzer
import captions_ass
import director
import effects
import finishing
import publish
import segment

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
FONTS_DIR = APP_DIR / "fonts"          # free creator fonts (Google Fonts, OFL)
BROLL_DIR = APP_DIR / "broll"          # your cutaway clips/images, named by keywords
MUSIC_DIR = APP_DIR / "music"          # background music by mood: music/<energetic|upbeat|funny|chill|dramatic|emotional>/
PRESETS_DIR = APP_DIR / "presets"      # saved brand presets
LUTS_DIR = APP_DIR / "luts"            # color looks (.cube); generated ones + your own
MODELS_DIR = APP_DIR / "models"        # small local AI models (person cutout)
PROJECTS_DIR = APP_DIR / "projects"    # saved edits you can reopen
for _d in (OUTPUT_DIR, FONTS_DIR, BROLL_DIR, MUSIC_DIR, PRESETS_DIR, LUTS_DIR, MODELS_DIR, PROJECTS_DIR):
    _d.mkdir(exist_ok=True)

TARGET_W, TARGET_H = 1080, 1920
AUDIO_SR = 48000  # DeepFilterNet native rate

def ensure_ffmpeg_on_path() -> None:
    """Whisper shells out to `ffmpeg`; expose imageio-ffmpeg's bundled binary if none is installed."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    bin_dir = APP_DIR / "bin"
    bin_dir.mkdir(exist_ok=True)
    target = bin_dir / "ffmpeg.exe"
    if not target.exists():
        shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), target)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


ensure_ffmpeg_on_path()

# Windows consoles default to cp1252; emoji in logs would crash print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))

COLOR_CHOICES = {
    "White": "#FFFFFF",
    "Target Yellow": "#FFE600",
    "Neon Green": "#39FF14",
    "Hot Pink": "#FF2E88",
    "Electric Cyan": "#00E5FF",
    "Orange": "#FF8A00",
}

# Windows font candidates, by script. Nirmala UI ships with Windows 10/11 and covers Indic scripts.
WIN_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FONT_CANDIDATES = {
    "latin": ["impact.ttf", "arialbd.ttf", "segoeuib.ttf", "arial.ttf"],
    "devanagari": ["NirmalaB.ttf", "Nirmala.ttc", "Nirmala.ttf", "mangalb.ttf", "mangal.ttf"],
    "gujarati": ["NirmalaB.ttf", "Nirmala.ttc", "Nirmala.ttf", "shrutib.ttf", "shruti.ttf"],
}

# Caption font families: display name -> (file in fonts/ or Windows Fonts, variable-font instance)
FONT_FAMILIES = {
    "Montserrat ExtraBold": ("Montserrat[wght].ttf", "ExtraBold"),
    "Montserrat Black": ("Montserrat[wght].ttf", "Black"),
    "Poppins Black": ("Poppins-Black.ttf", None),
    "Poppins ExtraBold": ("Poppins-ExtraBold.ttf", None),
    "Anton": ("Anton-Regular.ttf", None),
    "Bebas Neue": ("BebasNeue-Regular.ttf", None),
    "Impact (Windows)": ("impact.ttf", None),
}
DEFAULT_FONT = "Montserrat ExtraBold"

# Hint prompts bias Whisper toward natural code-switched spelling.
LANGUAGE_PROMPTS = {
    "hi": "Toh guys, aaj hum baat karenge ek important topic ke baare mein. Yeh video end tak dekhna, bahut value milegi.",
    "gu": "Kem cho friends, aaje aapde vaat karishu ek important topic vishe. Aa video puro jojo, bahu kaam aavse.",
    "en": "Hey guys, welcome back. Today we're talking about something really important.",
}

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "but", "so", "to", "of", "in", "on", "at", "for", "with", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "our", "their", "his", "her", "them", "us", "do", "does", "did",
    "have", "has", "had", "just", "like", "um", "uh", "okay", "ok", "yeah", "can", "will", "would",
    "if", "then", "there", "here", "what", "when", "how", "not", "no", "yes", "am", "as", "by", "from",
    # Romanized Hindi / Gujarati fillers
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "toh", "to", "bhi", "na", "ne",
    "ye", "yeh", "woh", "wo", "aur", "ek", "kya", "ho", "tha", "thi", "the", "raha", "rahe", "rahi",
    "che", "chhe", "nu", "ni", "no", "ma", "thi", "ane", "pan", "aa", "e", "tame", "hu", "haan",
    # Devanagari / Gujarati fillers
    "है", "हैं", "का", "की", "के", "को", "से", "में", "पर", "तो", "भी", "ना", "ने", "ये", "यह", "वो",
    "और", "एक", "क्या", "हो", "था", "थी", "छे", "છે", "નું", "ની", "નો", "માં", "અને", "પણ", "આ",
}

PUNCHY_WORDS = {
    "never", "always", "secret", "free", "money", "best", "worst", "stop", "now", "fast", "huge",
    "insane", "crazy", "must", "important", "mistake", "truth", "million", "billion", "win", "fail",
    "boom", "why", "wow", "power", "hack", "tip", "growth", "viral", "first", "last", "only",
    "zabardast", "bindaas", "ekdum", "bahut", "sabse", "kabhi", "mast", "jordaar", "bau", "bahu",
}

class PipelineError(RuntimeError):
    """Raised for user-facing, recoverable pipeline failures."""


@dataclass
class Word:
    text: str
    start: float
    end: float
    highlight: bool = False


@dataclass
class Settings:
    cut_silences: bool = True
    silence_threshold: float = 0.8
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    model_size: str = "small"
    language: str = "auto"
    reframe_mode: str = "Blurred background"
    words_per_caption: int = 3
    denoise: bool = True
    font_size: int = 96
    color_grade: bool = True
    camera_moves: bool = True
    progress_bar: bool = True
    use_ai: bool = True
    llm_model: str = director.DEFAULT_LLM
    caption_style: str = "auto"  # auto | pop | box
    emotion_colors: bool = True
    face_tracking: bool = True
    emoji: bool = True
    sfx: bool = True
    hook_title: bool = True
    transitions: bool = True
    music_path: str | None = None
    music_volume: float = 0.18
    font_family: str = DEFAULT_FONT
    ai_spellfix: bool = True
    remove_fillers: bool = True
    smart_cuts: bool = True       # cut retakes / false starts / mic checks chosen by the director
    hook_reorder: bool = True     # move the strongest line to the start (cold open)
    broll: bool = True
    voice_polish: bool = True     # EQ + de-esser (+ -14 LUFS on the final mix)
    auto_music: bool = True       # pick music from music/<mood> when none is uploaded
    logo_path: str | None = None
    handle: str = ""
    cta_text: str = ""
    end_card: bool = True
    draft: bool = False
    autopilot: bool = True        # AI decides every edit setting from what it understands
    use_vision: bool = True       # local vision model describes key frames
    camera_intensity: float = 1.0
    emoji_rate: float = 0.7       # share of director-suggested emoji that are shown
    grade_strength: float = 1.0
    caption_y: float = 0.66       # vertical caption position (fraction of height)
    look: str = "Auto (by mood)"  # cinematic LUT
    stabilize: bool = False       # manual switch; autopilot also turns it on for shaky footage
    title_style: str = "auto"     # auto | card | behind (text behind the speaker)
    multi_speaker: bool = True    # follow whoever is talking when several people are in frame
    export_srt: bool = True
    english_subs: bool = True
    export_xml: bool = True
    cover: bool = True
    post_copy: bool = True
    export_formats: tuple = ()    # extra aspect ratios: "1x1", "4x5", "16x9"


@dataclass
class JobContext:
    work_dir: Path
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.logs.append(line)
        print(line, flush=True)

    @property
    def text(self) -> str:
        return "\n".join(self.logs)


# --------------------------------------------------------------------------------------
# Phase A: Audio isolation, denoising, compression, normalization
# --------------------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path, ctx: JobContext) -> bool:
    """Extract mono 48 kHz audio. Returns False if the video has no audio stream."""
    clip = VideoFileClip(str(video_path))
    try:
        if clip.audio is None:
            ctx.log("No audio stream found; captions and silence cutting will be skipped.")
            return False
        clip.audio.write_audiofile(
            str(out_wav), fps=AUDIO_SR, nbytes=2, codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"], logger=None,
        )
    finally:
        clip.close()
    if not out_wav.exists() or out_wav.stat().st_size < 1024:
        ctx.log("Audio stream is empty; continuing without audio processing.")
        return False
    return True


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


DF_SIDECAR = APP_DIR / "tools" / "dfenv" / "Scripts" / "python.exe"


def denoise_deepfilternet(audio: np.ndarray, sr: int) -> np.ndarray:
    """Studio-grade speech cleanup. Runs in-process if DeepFilterNet is importable, otherwise through the
    Python 3.11 sidecar in tools/dfenv (DeepFilterNet doesn't support newer Pythons yet)."""
    import importlib.util

    if importlib.util.find_spec("df") is None:  # DeepFilterNet not importable here -> use the sidecar
        if not DF_SIDECAR.exists():
            raise ImportError("DeepFilterNet is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            sf.write(str(src), audio, sr, subtype="FLOAT")
            proc = subprocess.run([str(DF_SIDECAR), str(APP_DIR / "tools" / "df_denoise.py"), str(src), str(dst)],
                                  capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0 or not dst.exists():
                raise RuntimeError(proc.stderr.strip()[-300:] or "DeepFilterNet sidecar failed")
            out, _ = sf.read(str(dst), dtype="float32")
            return out[: audio.size] if out.size >= audio.size else np.pad(out, (0, audio.size - out.size))
    from df.enhance import enhance, init_df  # type: ignore

    model, df_state, _ = init_df()
    model_sr = df_state.sr()
    if sr != model_sr:
        audio = resample(audio, sr, model_sr)
    tensor = torch.from_numpy(audio).unsqueeze(0)
    enhanced = enhance(model, df_state, tensor).squeeze(0).cpu().numpy()
    if sr != model_sr:
        enhanced = resample(enhanced, model_sr, sr)
    return enhanced.astype(np.float32)


def denoise_noisereduce(audio: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr

    # Stationary pass kills hum/fan; a gentle non-stationary pass tames room tone.
    stage1 = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.9, n_jobs=1)
    stage2 = nr.reduce_noise(y=stage1, sr=sr, stationary=False, prop_decrease=0.6, n_jobs=1)
    return stage2.astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or audio.size == 0:
        return audio
    duration = audio.size / sr_in
    n_out = int(round(duration * sr_out))
    x_old = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def highpass_fast(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """FFT-domain high-pass (fast for long files)."""
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    ramp = np.clip((freqs - cutoff * 0.5) / (cutoff * 0.5), 0.0, 1.0)
    return np.fft.irfft(spectrum * ramp, n=audio.size).astype(np.float32)


def compress(audio: np.ndarray, sr: int, threshold_db: float = -20.0, ratio: float = 3.5,
             attack_ms: float = 5.0, release_ms: float = 80.0, makeup_db: float = 4.0) -> np.ndarray:
    """Feed-forward RMS compressor with block-wise envelope follower."""
    block = max(1, int(sr * 0.002))  # 2 ms blocks
    n_blocks = int(np.ceil(audio.size / block))
    padded = np.pad(audio, (0, n_blocks * block - audio.size))
    rms = np.sqrt(np.mean(padded.reshape(n_blocks, block) ** 2, axis=1) + 1e-12)
    level_db = 20 * np.log10(rms)

    over = np.maximum(level_db - threshold_db, 0.0)
    target_gain_db = -over * (1.0 - 1.0 / ratio)

    block_sec = block / sr
    a_att = np.exp(-block_sec / (attack_ms / 1000))
    a_rel = np.exp(-block_sec / (release_ms / 1000))
    smoothed = np.empty_like(target_gain_db)
    g = 0.0
    for i, tg in enumerate(target_gain_db):
        coeff = a_att if tg < g else a_rel
        g = coeff * g + (1 - coeff) * tg
        smoothed[i] = g

    gain = 10 ** ((smoothed + makeup_db) / 20)
    gain_samples = np.repeat(gain, block)[: audio.size]
    return (audio * gain_samples).astype(np.float32)


def normalize(audio: np.ndarray, target_rms_db: float = -16.0, peak_ceiling_db: float = -1.0) -> np.ndarray:
    """Speech-weighted loudness normalization with a soft limiter."""
    if audio.size == 0:
        return audio
    frame = 2048
    usable = audio[: (audio.size // frame) * frame]
    if usable.size == 0:
        usable = audio
        frames = usable[None, :]
    else:
        frames = usable.reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    voiced = frame_rms[frame_rms > np.percentile(frame_rms, 40)]
    speech_rms = float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else float(np.sqrt(np.mean(audio ** 2)))
    if speech_rms < 1e-6:
        return audio
    gain = 10 ** (target_rms_db / 20) / speech_rms
    out = audio * gain
    ceiling = 10 ** (peak_ceiling_db / 20)
    return (np.tanh(out / ceiling) * ceiling).astype(np.float32)


def process_audio(raw_wav: Path, clean_wav: Path, settings: Settings, ctx: JobContext) -> None:
    audio, sr = load_mono(raw_wav)
    if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-5:
        ctx.log("Audio is silent; skipping enhancement.")
        sf.write(str(clean_wav), audio, sr)
        return

    audio = highpass_fast(audio, sr)
    if settings.denoise:
        try:
            ctx.log("Denoising with DeepFilterNet...")
            audio = denoise_deepfilternet(audio, sr)
        except Exception as exc:  # ImportError or runtime failure -> fallback
            ctx.log(f"DeepFilterNet unavailable ({type(exc).__name__}); using noisereduce fallback.")
            try:
                audio = denoise_noisereduce(audio, sr)
            except Exception as exc2:
                ctx.log(f"noisereduce failed ({exc2}); keeping original audio.")
    if settings.voice_polish:
        ctx.log("Voice polish: broadcast EQ (less mud, more presence) + de-esser...")
        audio = effects.deess(effects.voice_eq(audio, sr), sr)
    ctx.log("Applying compression and loudness normalization...")
    audio = compress(audio, sr)
    audio = normalize(audio)
    sf.write(str(clean_wav), audio, sr, subtype="PCM_16")


# --------------------------------------------------------------------------------------
# Phase B: Language detection & transcription
# --------------------------------------------------------------------------------------

_MODEL_CACHE: dict[str, whisper.Whisper] = {}


def get_whisper_model(size: str) -> whisper.Whisper:
    if size not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # free VRAM when switching sizes
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        _MODEL_CACHE[size] = whisper.load_model(size, device=DEVICE)
    return _MODEL_CACHE[size]


def detect_language(model: whisper.Whisper, wav_path: Path) -> tuple[str, dict[str, float]]:
    audio = whisper.load_audio(str(wav_path))
    # Probe several windows so an intro in one language doesn't dominate.
    probs_total: dict[str, float] = {}
    window = whisper.audio.N_SAMPLES
    starts = [0] + [s for s in (window, 2 * window) if s + window // 2 < audio.size]
    for s in starts:
        seg = whisper.pad_or_trim(audio[s: s + window])
        mel = whisper.log_mel_spectrogram(seg, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        for lang, p in probs.items():
            probs_total[lang] = probs_total.get(lang, 0.0) + p / len(starts)
    # Code-switched Indic speech is often mis-labelled as ur/pa/mr/ne/bn; fold into hi.
    for alias in ("ur", "pa", "mr", "ne"):
        probs_total["hi"] = probs_total.get("hi", 0.0) + probs_total.pop(alias, 0.0)
    best = max(probs_total, key=probs_total.get)
    top = dict(sorted(probs_total.items(), key=lambda kv: kv[1], reverse=True)[:4])
    return best, top


def transcribe(wav_path: Path, settings: Settings, ctx: JobContext) -> tuple[list[Word], str]:
    ctx.log(f"Loading Whisper '{settings.model_size}' on {DEVICE.upper()}...")
    model = get_whisper_model(settings.model_size)

    requested = settings.language
    romanize = requested in ("hinglish", "gujlish")
    if requested == "auto":
        lang, probs = detect_language(model, wav_path)
        ctx.log("Language probabilities: " + ", ".join(f"{k}={v:.2f}" for k, v in probs.items()))
        # Indian creators almost always caption in Roman script (Hinglish/Gujlish), so
        # auto mode romanizes; pick "Hindi (Devanagari)" / "Gujarati" for native script.
        if lang in ("hi", "gu"):
            romanize = True
        if lang == "en" and max(probs.get("hi", 0), probs.get("gu", 0)) > 0.2:
            lang, romanize = ("hi" if probs.get("hi", 0) >= probs.get("gu", 0) else "gu"), True
    elif requested == "hinglish":
        lang = "hi"
    elif requested == "gujlish":
        lang = "gu"
    else:
        lang = requested

    label = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}.get(lang, lang)
    ctx.log(f"Detected language: {label}{' (code-switched, romanized)' if romanize else ''}")

    options = dict(
        language=lang,
        task="transcribe",
        word_timestamps=True,
        condition_on_previous_text=False,
        fp16=(DEVICE == "cuda"),
        temperature=(0.0, 0.2, 0.4),
        beam_size=5,
        best_of=5,
        no_speech_threshold=0.5,
        verbose=None,
    )
    if romanize:
        # Hindi/Gujarati decoding with a Latin-script prompt yields Hinglish/Gujlish spelling.
        options["initial_prompt"] = LANGUAGE_PROMPTS.get(lang, LANGUAGE_PROMPTS["en"])

    result = model.transcribe(str(wav_path), **options)

    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            text = w.get("word", "").strip()
            if not text:
                continue
            start, end = float(w["start"]), float(w["end"])
            if end <= start:
                end = start + 0.05
            words.append(Word(text=text, start=start, end=end))
    words = fix_word_overlaps(words)

    if not words:
        ctx.log("Whisper found no speech in the audio.")
    else:
        ctx.log(f"Transcribed {len(words)} words.")
    return words, lang


def fix_word_overlaps(words: list[Word]) -> list[Word]:
    words.sort(key=lambda w: w.start)
    for prev, cur in zip(words, words[1:]):
        if cur.start < prev.end:
            mid = (prev.end + cur.start) / 2
            prev.end = max(prev.start + 0.02, mid)
            cur.start = prev.end
            cur.end = max(cur.end, cur.start + 0.02)
    return words


def save_word_map(words: list[Word], lang: str, path: Path) -> None:
    payload = {
        "language": lang,
        "word_count": len(words),
        "words": [
            {"word": w.text, "start": round(w.start, 3), "end": round(w.end, 3), "highlight": w.highlight}
            for w in words
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Phase D (part 1): Silence detection & timeline remapping
# --------------------------------------------------------------------------------------

def compute_keep_segments(words: list[Word], duration: float, threshold: float,
                          pad: float = 0.12) -> list[tuple[float, float]]:
    """Return (start, end) ranges of source time to keep, dropping gaps > threshold."""
    if not words:
        return [(0.0, duration)]
    segments: list[tuple[float, float]] = []
    seg_start = max(0.0, words[0].start - pad)
    seg_end = words[0].end
    for w in words[1:]:
        if w.start - seg_end > threshold:
            segments.append((seg_start, min(duration, seg_end + pad)))
            seg_start = max(0.0, w.start - pad)
        seg_end = max(seg_end, w.end)
    segments.append((seg_start, min(duration, seg_end + pad)))

    # Merge segments that overlap after padding.
    merged: list[tuple[float, float]] = []
    for s, e in segments:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        elif e - s > 0.05:
            merged.append((s, e))
    return merged or [(0.0, duration)]


def remap_words(words: list[Word], segments: list[tuple[float, float]]) -> list[Word]:
    """Map word times from source timeline to the jump-cut timeline."""
    offsets = []
    acc = 0.0
    for s, e in segments:
        offsets.append((s, e, acc))
        acc += e - s
    out: list[Word] = []
    for w in words:
        for s, e, off in offsets:
            if s <= w.start < e:
                new_start = w.start - s + off
                new_end = min(w.end, e) - s + off
                out.append(Word(w.text, new_start, max(new_end, new_start + 0.05), w.highlight))
                break
    out.sort(key=lambda w: w.start)  # segments may be reordered (cold-open hook)
    return out


def subtract_ranges(segments: list[tuple[float, float]], cuts: list[tuple[float, float]],
                    min_len: float = 0.08) -> list[tuple[float, float]]:
    """Remove cut ranges from keep-segments (order of segments is preserved)."""
    out = []
    for s, e in segments:
        pieces = [(s, e)]
        for cs, ce in cuts:
            nxt = []
            for ps, pe in pieces:
                if ce <= ps or cs >= pe:
                    nxt.append((ps, pe))
                    continue
                if cs > ps:
                    nxt.append((ps, cs))
                if ce < pe:
                    nxt.append((ce, pe))
            pieces = nxt
        out.extend(p for p in pieces if p[1] - p[0] >= min_len)
    return out


def intersect_ranges(segments: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(s, lo), min(e, hi)) for s, e in segments if min(e, hi) - max(s, lo) > 0.05]


def leading_count_in(words: list[Word]) -> list[Word]:
    """Words at the very start that are only a count-in or mic check ("1, 2, 3", "testing", "hello hello")."""
    out = []
    for w in words:
        tok = clean_token(w.text)
        if tok and (tok.isdigit() or tok in director.MIC_CHECK_WORDS) and tok not in ("so", "ok", "okay"):
            out.append(w)
        else:
            break
    return out


FILLER_WORDS = {"um", "umm", "uh", "uhh", "uhm", "hmm", "hmmm", "erm", "er", "ah", "aah", "mm", "mmm"}


# --------------------------------------------------------------------------------------
# Phase C: Kinetic captions
# --------------------------------------------------------------------------------------

def clean_token(text: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F\u0A80-\u0AFF']", "", text.lower())


def mark_keywords(words: list[Word]) -> None:
    """Highlight punchy/informative words: numbers, emphasis words, long content words."""
    if not words:
        return
    tokens = [clean_token(w.text) for w in words]
    freq: dict[str, int] = {}
    for t in tokens:
        if t and t not in STOPWORDS:
            freq[t] = freq.get(t, 0) + 1

    last_highlight = -10
    for i, (w, tok) in enumerate(zip(words, tokens)):
        if not tok or tok in STOPWORDS:
            continue
        score = 0.0
        if tok in PUNCHY_WORDS:
            score += 3
        if any(ch.isdigit() for ch in tok) or "%" in w.text or "$" in w.text or "₹" in w.text:
            score += 3
        if w.text.strip().endswith(("!", "?")):
            score += 1.5
        if len(tok) >= 7:
            score += 1.5
        if w.text.strip().isupper() and len(tok) > 1:
            score += 1
        if freq.get(tok, 0) >= 3:
            score += 1  # recurring topic word
        dur = w.end - w.start
        if dur > 0.45:
            score += 1  # speaker stressed it
        if score >= 2.5 and i - last_highlight >= 2:
            w.highlight = True
            last_highlight = i


def group_words(words: list[Word], max_words: int, pause_break: float = 0.3,
                max_chars: int = 18) -> list[list[Word]]:
    """Chunk into 1..max_words groups, breaking on pauses, punctuation and length."""
    groups: list[list[Word]] = []
    current: list[Word] = []
    for i, w in enumerate(words):
        if current:
            gap = w.start - current[-1].end
            chars = sum(len(x.text) + 1 for x in current) + len(w.text)
            if len(current) >= max_words or gap > pause_break or chars > max_chars:
                groups.append(current)
                current = []
        current.append(w)
        if w.text.rstrip().endswith((".", "?", "!", ",", "।")):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def detect_script(text: str) -> str:
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gujarati"
    if re.search(r"[\u0900-\u097F]", text):
        return "devanagari"
    return "latin"


_FONT_CACHE: dict[tuple[str, int, str], ImageFont.FreeTypeFont] = {}


def load_family_font(family: str, size: int) -> ImageFont.FreeTypeFont | None:
    file, variation = FONT_FAMILIES.get(family, FONT_FAMILIES[DEFAULT_FONT])
    for folder in (FONTS_DIR, WIN_FONTS):
        p = folder / file
        if p.exists():
            try:
                font = ImageFont.truetype(str(p), size)
                if variation:
                    font.set_variation_by_name(variation)
                return font
            except OSError:
                continue
    return None


def get_font(script: str, size: int, family: str = DEFAULT_FONT) -> ImageFont.FreeTypeFont:
    key = (script, size, family)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = load_family_font(family, size) if script == "latin" else None
    for name in ([] if font else FONT_CANDIDATES[script]):
        path = WIN_FONTS / name
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size)
                break
            except OSError:
                continue
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            font = ImageFont.load_default(size=size)
    _FONT_CACHE[key] = font
    return font


def write_captions_ass(words: list[Word], settings: Settings, lines: list, duration: float, path: Path,
                       groups: list[list[Word]] | None = None) -> None:
    """Write the libass caption track for the edited timeline (groups = on-screen caption chunks)."""
    captions_ass.ensure_static_fonts(FONTS_DIR)
    groups = groups if groups is not None else group_words(words, settings.words_per_caption)

    def measure(text: str, size: int) -> float:
        return get_font("latin", size, settings.font_family).getlength(text)

    def upper(text: str) -> str:
        return text.upper() if detect_script(text) == "latin" else text

    doc = captions_ass.build_ass(groups, settings, make_line_lookup(lines) if lines else None, measure,
                                 TARGET_W, TARGET_H, total=duration, emotion_colors=director.EMOTION_COLORS,
                                 upper=upper)
    path.write_text(doc, encoding="utf-8")


def _scaled_entrance(clip: ImageClip, scale: Callable[[float], float]) -> ImageClip:
    base_pos = clip.pos(0)
    w, h = clip.size

    def position(t: float):
        s = scale(t)
        return (base_pos[0] + w * (1 - s) / 2, base_pos[1] + h * (1 - s) / 2)

    return clip.resized(scale).with_position(position)


def bounce_in(clip: ImageClip, dur: float = 0.28) -> ImageClip:
    """Overshooting elastic entrance for high-energy lines."""
    def scale(t: float) -> float:
        x = min(1.0, t / dur) - 1.0
        c1 = 1.9  # easeOutBack overshoot
        return 0.5 + 0.5 * (1 + (c1 + 1) * x ** 3 + c1 * x ** 2)
    return _scaled_entrance(clip, lambda t: max(0.4, scale(t)))



# --------------------------------------------------------------------------------------
# Visual polish: mood-aware color grade, virtual camera, overlays
# --------------------------------------------------------------------------------------

def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def measure_luma(path: Path, samples: int = 8) -> float:
    """Average luma (0..1) across evenly spaced frames."""
    clip = VideoFileClip(str(path), audio=False)
    try:
        times = np.linspace(0, max(0.0, clip.duration - 0.1), samples)
        vals = []
        for t in times:
            f = clip.get_frame(float(t)).astype(np.float32) / 255.0
            vals.append(float(np.mean(0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2])))
        return float(np.mean(vals))
    finally:
        clip.close()


# Grade "looks" per overall mood: (contrast, saturation, red shift, blue shift)
MOOD_LOOKS = {
    "excited": (1.10, 1.30, 0.04, -0.04), "happy": (1.06, 1.28, 0.05, -0.04), "funny": (1.06, 1.30, 0.03, -0.02),
    "confident": (1.12, 1.20, 0.03, -0.03), "curious": (1.08, 1.18, 0.0, 0.01), "surprised": (1.10, 1.22, 0.02, 0.0),
    "serious": (1.14, 1.02, -0.02, 0.03), "emotional": (1.05, 0.95, -0.01, 0.04), "angry": (1.16, 1.10, 0.04, -0.02),
    "calm": (1.05, 1.10, 0.01, 0.01),
}


def enhance_video(src: Path, dst: Path, ctx: JobContext, mood: str = "calm", strength: float = 1.0,
                  look: str = "None", stabilize: bool = False, grade: bool = True) -> Path:
    """Auto exposure + mood-based color look, light denoise and sharpening via one ffmpeg pass."""
    import subprocess

    luma = measure_luma(src)
    target = 0.45
    # eq gamma: out = in^(1/gamma). Solve for gamma that moves mean luma toward target.
    gamma = float(np.clip(np.log(max(luma, 1e-3)) / np.log(target), 0.85, 1.7))
    brightness = float(np.clip((target - luma) * 0.15, -0.05, 0.06))
    contrast, saturation, rs, bs = MOOD_LOOKS.get(mood, MOOD_LOOKS["calm"])
    contrast, saturation = round(1 + (contrast - 1) * strength, 3), round(1 + (saturation - 1) * strength, 3)
    rs, bs = round(rs * strength, 3), round(bs * strength, 3)
    ctx.log(f"Color grade: exposure {luma:.2f} -> gamma {gamma:.2f}; '{mood}' look "
            f"(contrast {contrast}, saturation {saturation})")
    chain = []
    if stabilize:
        ctx.log("Stabilizing shaky footage (2-pass vid.stab)...")
        detect = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf",
                  "vidstabdetect=shakiness=6:accuracy=12:result=transforms.trf", "-f", "null", "-"]
        if subprocess.run(detect, cwd=dst.parent, capture_output=True).returncode == 0:
            chain.append("vidstabtransform=input=transforms.trf:smoothing=18:zoom=3:optzoom=0")
        else:
            ctx.log("Stabilization analysis failed; continuing without it.")
    if grade:
        chain += [
            "hqdn3d=1.5:1.5:6:6",
            f"eq=gamma={gamma:.3f}:brightness={brightness:.3f}:contrast={contrast}:saturation={saturation}",
            f"colorbalance=rs={rs}:bs={bs}:rh={rs / 2:.3f}:bh={bs / 2:.3f}",
        ]
    lut = finishing.ensure_lut(LUTS_DIR, look)
    if lut:
        shutil.copy(lut, dst.parent / "look.cube")
        chain.append("lut3d=look.cube:interp=tetrahedral")
        ctx.log(f"Color look: {look}")
    if grade:
        chain += ["unsharp=5:5:0.7:5:5:0.0", *(["vignette=angle=PI/6"] if strength >= 0.8 else [])]
    if not chain:
        return src
    vf = ",".join(chain)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
           "-c:a", "copy", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=dst.parent)
    if proc.returncode != 0 or not dst.exists():
        ctx.log(f"Color grade skipped (ffmpeg error: {proc.stderr.strip()[:200]})")
        return src
    return dst


def remap_time(t: float, segments: list[tuple[float, float]]) -> float:
    """Source time -> edited timeline time. Works with reordered segments; times inside removed
    gaps snap to the start of the next kept segment (in source order)."""
    acc, offsets = 0.0, []
    for s, e in segments:
        offsets.append((s, e, acc))
        acc += e - s
    for s, e, off in offsets:
        if s <= t < e:
            return off + t - s
    later = [(s, off) for s, _, off in offsets if s >= t]
    return min(later)[1] if later else acc


def remap_plan(plan: director.EditPlan, segments: list[tuple[float, float]]) -> list[director.LinePlan]:
    out = []
    for l in plan.lines:
        if not l.keep:
            continue
        nl = director.LinePlan(**{**l.__dict__})
        nl.start = remap_time(l.start, segments)
        nl.end = nl.start + sum(e - s for s, e in intersect_ranges(segments, l.start, l.end))
        if nl.end - nl.start > 0.05:
            out.append(nl)
    out.sort(key=lambda l: l.start)
    return out


def make_line_lookup(lines: list[director.LinePlan]) -> Callable[[float], director.LinePlan | None]:
    def line_at(t: float) -> director.LinePlan | None:
        best = None
        for l in lines:
            if l.start - 0.05 <= t:
                best = l
            else:
                break
        return best
    return line_at


def emphasis_hit(line: director.LinePlan, words: list[Word]) -> float:
    for w in words:
        if line.start - 0.01 <= w.start <= line.end and w.highlight:
            return w.start
    return line.start + min(0.15, (line.end - line.start) / 3)


def progress_bar_clip(duration: float, color: str, height: int = 12) -> ImageClip:
    rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    bar = np.zeros((height, TARGET_W, 3), dtype=np.uint8)
    bar[:] = rgb
    return (ImageClip(bar).with_duration(duration)
            .with_position(lambda t: (int(-TARGET_W + TARGET_W * t / max(duration, 1e-3)), 0)))


def overlay_clips(lines: list[director.LinePlan], words: list[Word], plan: director.EditPlan,
                  settings: Settings, duration: float, ctx: JobContext) -> list:
    clips = []
    # Emoji pop-ins above the captions, alternating slightly left/right for life.
    if settings.emoji:
        side, budget = 1, 1.0
        for l in lines:
            if not l.emoji:
                continue
            if budget < 1.0:  # autopilot thins emoji for calmer content types
                budget += settings.emoji_rate
                continue
            budget = budget - 1.0 + settings.emoji_rate
            img = effects.emoji_image(l.emoji, size=170 if l.energy >= 4 else 140)
            if img is None:
                continue
            side = -side
            start = emphasis_hit(l, words)
            dur = min(1.6, max(0.6, l.end - start + 0.2))
            h, w = img.shape[:2]
            pos = (TARGET_W // 2 - w // 2 + side * 340, int(TARGET_H * 0.40) - h // 2)  # beside the face
            clip = ImageClip(img, transparent=True).with_start(start).with_duration(dur).with_position(pos)
            clips.append(bounce_in(clip).with_effects([vfx.CrossFadeOut(0.2)]))
        ctx.log(f"Emoji pop-ins: {sum(1 for l in lines if l.emoji)}")

    # Flash / zoom-flash transitions at story beats.
    if settings.transitions:
        for l in lines:
            if l.transition in ("flash", "zoom") and l.start > 0.1:
                opacity = 0.85 if l.transition == "flash" else 0.35
                flash = (ColorClip((TARGET_W, TARGET_H), color=(255, 255, 255)).with_duration(0.22)
                         .with_start(max(0.0, l.start - 0.04)).with_opacity(opacity)
                         .with_effects([vfx.CrossFadeOut(0.2)]))
                clips.append(flash)

    # Hook title card for the first seconds.
    if settings.hook_title and plan.title_hook:
        accent = director.EMOTION_COLORS.get(plan.mood, settings.highlight_color)
        font_path = lambda size: load_family_font(settings.font_family, size)  # noqa: E731
        if font_path(40):
            img = effects.title_card_image(plan.title_hook, accent, font_path)
            h, w = img.shape[:2]
            dur = min(3.0, duration)
            clip = (ImageClip(img, transparent=True).with_start(0.0).with_duration(dur)
                    .with_position(((TARGET_W - w) // 2, 250)))
            clips.append(bounce_in(clip).with_effects([vfx.CrossFadeOut(0.35)]))
            ctx.log(f"Hook title: “{plan.title_hook}”")

    # Brand: logo watermark (top-left, clear of Instagram's UI) and call-to-action end card.
    if settings.logo_path and Path(settings.logo_path).exists():
        logo = effects.logo_image(Path(settings.logo_path))
        if logo is not None:
            clips.append(ImageClip(logo, transparent=True).with_duration(duration).with_position((48, 170)))
    cta = end_card_lines(settings)
    if settings.end_card and cta and duration > 4:
        font_loader = lambda size: load_family_font(settings.font_family, size)  # noqa: E731
        img = effects.end_card_image(cta[0], cta[1], settings.highlight_color, font_loader)
        h, w = img.shape[:2]
        start = max(0.0, duration - 2.6)
        clip = (ImageClip(img, transparent=True).with_start(start).with_duration(duration - start)
                .with_position(((TARGET_W - w) // 2, 300)))
        clips.append(bounce_in(clip))

    if settings.progress_bar:
        clips.append(progress_bar_clip(duration, settings.highlight_color))
    return clips


def end_card_lines(settings: Settings) -> tuple[str, str] | None:
    handle = settings.handle.strip()
    if handle and not handle.startswith("@"):
        handle = "@" + handle
    if settings.cta_text.strip():
        return settings.cta_text.strip(), handle
    if handle:
        return f"Follow {handle}", "for more"
    return None


def broll_clips(lines: list[director.LinePlan], settings: Settings, ctx: JobContext) -> list:
    """Cutaways from the local B-roll folder, matched to what is being said."""
    if not settings.broll or not any(BROLL_DIR.iterdir()):
        return []
    used: set[Path] = set()
    clips, last_k = [], -99
    for k, l in enumerate(lines):
        if k < 1 or k - last_k < 3 or l.end - l.start < 1.2 or l.beat == "hook":
            continue  # never over the opening; leave room between cutaways
        p = effects.find_broll(BROLL_DIR, l.broll, l.text, used)
        if not p:
            continue
        start = l.start + 0.15
        dur = min(2.2, l.end - start)
        try:
            if p.suffix.lower() in effects.IMAGE_EXT:
                src = ImageClip(str(p)).with_duration(dur)
            else:
                src = VideoFileClip(str(p), audio=False)
                if src.duration < 0.6:
                    continue
                dur = min(dur, src.duration)
                src = src.subclipped(0, dur)
        except Exception as exc:
            ctx.log(f"B-roll skipped ({p.name}: {exc})")
            continue
        scale = max(TARGET_W / src.w, TARGET_H / src.h) * 1.03  # overscan: no edge slivers after rounding
        src = src.resized(scale)
        w, h = src.size
        zoom = lambda t, d=dur: 1.0 + 0.06 * t / max(d, 0.1)  # slow Ken Burns push-in  # noqa: E731
        clip = (src.resized(zoom)
                .with_position(lambda t, w=w, h=h, z=zoom: ((TARGET_W - w * z(t)) / 2, (TARGET_H - h * z(t)) / 2))
                .with_start(start).with_duration(dur)
                .with_effects([vfx.CrossFadeIn(0.12), vfx.CrossFadeOut(0.12)]))
        clips.append(clip)
        used.add(p)
        last_k = k
        ctx.log(f"B-roll: '{p.name}' over line {l.i} (\"{l.text[:40]}\")")
    return clips


def sfx_events(lines: list[director.LinePlan], plan: director.EditPlan, settings: Settings,
               duration: float) -> list[tuple[float, str]]:
    events: list[tuple[float, str]] = []
    if settings.hook_title and plan.title_hook:
        events.append((0.02, "whoosh"))
    if settings.end_card and end_card_lines(settings) and duration > 4:
        events.append((max(0.0, duration - 2.6), "ding"))
    for l in lines:
        if l.sfx != "none":
            events.append((l.start, l.sfx))
        elif settings.transitions and l.transition in ("whip", "zoom"):
            events.append((l.start, "swoosh"))
    return events


# --------------------------------------------------------------------------------------
# Editable transcript (review step)
# --------------------------------------------------------------------------------------

LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\|\s*(.*)$")


def words_to_script(words: list[Word], max_words: int) -> str:
    lines = []
    for group in group_words(words, max_words):
        text = " ".join(f"*{w.text.strip()}*" if w.highlight else w.text.strip() for w in group)
        lines.append(f"{group[0].start:.2f} - {group[-1].end:.2f} | {text}")
    return "\n".join(lines)


def script_to_words(script: str) -> list[Word]:
    """Parse edited caption lines; retimes words evenly (weighted by length) inside each line."""
    words: list[Word] = []
    for raw in script.splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        start, end, text = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        tokens = text.split()
        if not tokens or end <= start:
            continue
        weights = [max(1, len(t.strip("*"))) + 2 for t in tokens]
        total = sum(weights)
        t = start
        for tok, wgt in zip(tokens, weights):
            dur = (end - start) * wgt / total
            hl = len(tok) > 2 and tok.startswith("*") and tok.endswith("*")
            words.append(Word(tok.strip("*"), t, t + dur, hl))
            t += dur
    return fix_word_overlaps(words)


def text_edit(original: list[Word], script: str) -> tuple[list[Word], list[tuple[float, float]]]:
    """Descript-style editing: compare the edited caption script with the original transcript.

    Deleted words/lines become cuts in the video; changed words keep their original timing (spelling
    fixes); added words are caption-only. Returns (caption words, cut ranges on the source timeline).
    """
    import difflib

    parsed = []
    for raw in script.splitlines():
        m = LINE_RE.match(raw)
        if m:
            parsed.append((float(m.group(1)), float(m.group(2)), m.group(3).split()))
    out: list[Word] = []
    cuts: list[tuple[float, float]] = []
    used: set[int] = set()
    for start, end, tokens in parsed:
        idx = [k for k, w in enumerate(original) if start - 0.02 <= w.start < end - 0.01 and k not in used]
        used.update(idx)
        orig = [original[k] for k in idx]
        if not orig:
            out.extend(script_to_words(f"{start:.2f} - {end:.2f} | {' '.join(tokens)}"))
            continue
        a = [clean_token(w.text) for w in orig]
        b = [clean_token(t.strip("*")) for t in tokens]
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
                for oi, tj in zip(range(i1, i2), range(j1, j2)):
                    tok = tokens[tj]
                    hl = len(tok) > 2 and tok.startswith("*") and tok.endswith("*")
                    out.append(Word(tok.strip("*"), orig[oi].start, orig[oi].end, hl))
            elif op == "delete":
                cuts.append((orig[i1].start - 0.03, orig[i2 - 1].end + 0.03))
            elif op == "replace":
                span_s, span_e = orig[i1].start, orig[i2 - 1].end
                out.extend(retime_tokens([t.strip("*") for t in tokens[j1:j2]], span_s, span_e))
            elif op == "insert":
                anchor = orig[i1 - 1].end if i1 > 0 else orig[0].start
                out.extend(retime_tokens([t.strip("*") for t in tokens[j1:j2]], anchor, anchor + 0.25 * (j2 - j1)))
    # Lines removed from the script entirely -> cut those words.
    for k, w in enumerate(original):
        if k not in used:
            cuts.append((w.start - 0.03, w.end + 0.03))
    merged: list[tuple[float, float]] = []
    for c in sorted(cuts):
        if merged and c[0] <= merged[-1][1] + 0.15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], c[1]))
        else:
            merged.append(c)
    return fix_word_overlaps(out), merged


PLAN_COLUMNS = ["#", "time", "text", "keep", "emotion", "energy", "beat", "emoji", "sfx", "camera", "transition",
                "emphasis", "broll"]


def plan_to_rows(plan: director.EditPlan) -> list[list]:
    return [[l.i, f"{l.start:.1f}s", l.text, "yes" if l.keep else "CUT", l.emotion, l.energy, l.beat, l.emoji,
             l.sfx, l.camera, l.transition, ", ".join(l.emphasis), l.broll] for l in plan.lines]


def plan_from_dict(d: dict) -> director.EditPlan:
    return director.EditPlan(title_hook=d["title_hook"], mood=d["mood"], summary=d["summary"], source=d["source"],
                             lines=[director.LinePlan(**l) for l in d["lines"]], hook_line=d.get("hook_line", 0),
                             content_type=d.get("content_type", "other"), pace=d.get("pace", "medium"))


def merge_plan_edits(plan: director.EditPlan, rows, title: str | None) -> director.EditPlan:
    """Apply user edits from the plan table back onto the plan (values are validated)."""
    if rows is not None and hasattr(rows, "values"):
        rows = rows.values.tolist()
    by_i = {l.i: l for l in plan.lines}
    for r in rows or []:
        if len(r) < len(PLAN_COLUMNS):
            continue
        try:
            line = by_i.get(int(r[0]))
        except (TypeError, ValueError):
            continue
        if not line:
            continue
        director.apply_directions(line, {
            "keep": str(r[3]).strip().lower() not in ("cut", "no", "false", "0"),
            "emotion": str(r[4]).strip(), "energy": r[5], "beat": str(r[6]).strip(), "emoji": str(r[7] or "").strip(),
            "sfx": str(r[8]).strip(), "camera": str(r[9]).strip(), "transition": str(r[10]).strip(),
            "emphasis": [e.strip() for e in str(r[11] or "").split(",") if e.strip()],
            "broll": str(r[12] or "").strip(),
        })
    if title is not None:
        plan.title_hook = title.strip()
    return plan


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def validate_video(path: Path) -> float:
    if not path.exists() or path.stat().st_size == 0:
        raise PipelineError("Uploaded file is empty or missing.")
    try:
        clip = VideoFileClip(str(path))
    except Exception as exc:
        raise PipelineError(f"Could not open video: {exc}") from exc
    try:
        if clip.duration is None or clip.duration <= 0.1:
            raise PipelineError("Video stream is empty (zero duration).")
        if clip.size is None or clip.size[0] == 0:
            raise PipelineError("No video stream found in file.")
        clip.get_frame(0)
        return float(clip.duration)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Video stream is unreadable: {exc}") from exc
    finally:
        clip.close()


def analyze(video_path: Path, settings: Settings, ctx: JobContext,
            progress: Callable[[float, str], None], words_override: list[Word] | None = None,
            lang_override: str | None = None) -> dict:
    """Phases A + B: audio polish and transcription. Returns job state for the next steps."""
    work = ctx.work_dir
    raw_wav, clean_wav = work / "raw.wav", work / "clean.wav"

    progress(0.02, "Validating video")
    duration = validate_video(video_path)
    ctx.log(f"Loaded video ({duration:.1f}s). Compute device: {DEVICE.upper()}"
            + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

    progress(0.05, "Step 1: Denoising audio")
    ctx.log("Step 1: Extracting & denoising audio...")
    has_audio = extract_audio(video_path, raw_wav, ctx)
    if has_audio:
        process_audio(raw_wav, clean_wav, settings, ctx)

    words: list[Word] = []
    lang = "none"
    if has_audio and words_override is not None:
        words, lang = words_override, lang_override or "auto"
        ctx.log(f"Step 2: Reusing transcript ({len(words)} words).")
    elif has_audio:
        progress(0.3, "Step 2: Detecting language & transcribing")
        ctx.log("Step 2: Detecting language and transcribing...")
        # Whisper is more accurate on the raw track; denoising artifacts confuse it.
        words, lang = transcribe(raw_wav, settings, ctx)
        if settings.ai_spellfix and settings.use_ai and words and words_override is None:
            progress(0.7, "AI spell-fix")
            words = ai_spellfix(words, settings, ctx)
        mark_keywords(words)

    progress(0.8, "Watching the footage")
    ctx.log("Step 3: Watching the footage (faces, motion, cuts, lighting)...")
    clip = VideoFileClip(str(video_path), audio=False)
    try:
        insight = analyzer.scan_video(clip, words, log=ctx.log)
        if settings.use_vision and settings.use_ai:
            if analyzer.vision_available():
                progress(0.9, "Vision AI: describing scenes")
                ctx.log("Vision AI: describing key frames...")
                insight.scenes = analyzer.describe_frames(clip, count=4 if duration > 20 else 3, log=ctx.log)
            else:
                ctx.log("Vision AI not installed (ollama pull moondream) — using frame analysis only.")
    finally:
        clip.close()

    return {"video": str(video_path), "has_audio": has_audio, "clean_wav": str(clean_wav),
            "lang": lang, "words": words, "work_dir": str(work), "insight": analyzer.insight_to_dict(insight)}


def retime_tokens(tokens: list[str], start: float, end: float) -> list[Word]:
    weights = [max(1, len(t)) + 2 for t in tokens]
    total, t, out = sum(weights), start, []
    for tok, wgt in zip(tokens, weights):
        dur = (end - start) * wgt / total
        out.append(Word(tok, t, t + dur))
        t += dur
    return out


def ai_spellfix(words: list[Word], settings: Settings, ctx: JobContext) -> list[Word]:
    """Let the local LLM fix misheard words sentence by sentence, using the whole transcript as context."""
    ok, why = director.ollama_available(settings.llm_model)
    if not ok:
        ctx.log(f"AI spell-fix skipped: {why}")
        return words
    # Whole sentences (not 10-word chunks), so the model never "completes" a line with the next one.
    groups = director.sentence_groups(words, max_words=40, pause=1.0)
    before = [" ".join(w.text.strip() for w in g) for g in groups]
    after = director.correct_transcript(before, settings.llm_model, ctx.log)
    out: list[Word] = []
    changed = 0
    for g, old, new in zip(groups, before, after):
        if new.split() == old.split():
            out.extend(g)
            continue
        changed += 1
        new_tokens = new.split()
        if len(new_tokens) == len(g):  # same word count: keep Whisper's precise timings
            out.extend(Word(tok, w.start, w.end) for tok, w in zip(new_tokens, g))
        else:
            out.extend(retime_tokens(new_tokens, g[0].start, g[-1].end))
        ctx.log(f"  fixed: \"{old}\" -> \"{new}\"")
    ctx.log(f"AI spell-fix corrected {changed} of {len(groups)} lines.")
    return fix_word_overlaps(out)


def direct(state: dict, words: list[Word], settings: Settings, ctx: JobContext) -> director.EditPlan:
    ctx.log("AI Director: analyzing story, emotion and delivery...")
    audio = Path(state["clean_wav"]) if state.get("has_audio") else None
    insight = analyzer.insight_from_dict(state.get("insight"))
    plan = director.build_plan(words, audio, settings.use_ai, settings.llm_model, ctx.log, visual=insight.describe())
    counts: dict[str, int] = {}
    for l in plan.lines:
        counts[l.emotion] = counts.get(l.emotion, 0) + 1
    ctx.log(f"Director ({plan.source}) planned {len(plan.lines)} lines. Emotions: "
            + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return plan


# Toggles the user can veto: autopilot may turn them off, but never back on if the user switched them off.
VETO_FIELDS = ["cut_silences", "camera_moves", "face_tracking", "transitions", "hook_reorder", "sfx", "emoji",
               "auto_music", "broll"]


def decide(settings: Settings, state: dict, plan: director.EditPlan) -> tuple[Settings, list[str]]:
    """Autopilot: derive every edit setting from the footage + story analysis."""
    if not settings.autopilot:
        return settings, ["Manual mode — using your settings exactly."]
    insight = analyzer.insight_from_dict(state.get("insight"))
    has_broll = any(p.suffix.lower() in effects.VIDEO_EXT | effects.IMAGE_EXT for p in BROLL_DIR.rglob("*"))
    auto, decisions = analyzer.autopilot(settings, insight, plan.content_type, plan.pace, plan.mood, has_broll)
    vetoed = [f for f in VETO_FIELDS if not getattr(settings, f) and getattr(auto, f)]
    if vetoed:
        auto = replace(auto, **{f: False for f in vetoed})
        decisions.append("Respecting your OFF switches: " + ", ".join(f.replace("_", " ") for f in vetoed) + ".")
    if settings.music_path:
        auto = replace(auto, music_volume=settings.music_volume or auto.music_volume)
    return auto, decisions


def qa_report(out_path: Path, source_dur: float, timeline_words: list[Word], lines: list, settings: Settings) -> str:
    """Self-check of the finished reel, like an editor's final pass."""
    import pyloudnorm as pyln
    import subprocess

    checks = []
    clip = VideoFileClip(str(out_path))
    try:
        dur = clip.duration
        dark = 0
        for t in np.arange(0.2, dur, 1.0):
            if clip.get_frame(float(t)).mean() < 8:
                dark += 1
        w, h = clip.size
    finally:
        clip.close()
    wav = out_path.with_suffix(".qa.wav")
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out_path), "-vn", "-ac", "2", str(wav)])
    lufs = None
    if wav.exists():
        a, sr = sf.read(str(wav))
        wav.unlink(missing_ok=True)
        if a.size and a.shape[0] > sr // 2:
            lufs = pyln.Meter(sr).integrated_loudness(a)
    cut_share = 1 - dur / max(source_dur, 0.1)
    covered = sum(w.end - w.start for w in timeline_words) / max(dur, 0.1)

    def row(ok: bool, text: str) -> str:
        return f'<li class="{"ok" if ok else "warn"}"><span class="dot"></span>{text}</li>'

    checks.append(row((w, h) == (TARGET_W, TARGET_H), f"Format {w}×{h} (9:16)"))
    checks.append(row(dur <= 90, f"Length {dur:.1f}s" + ("" if dur <= 90 else " (over 90 s; some platforms trim longer Reels/Shorts)")))
    checks.append(row(cut_share < 0.45, f"Tightened by {cut_share:.0%} ({source_dur:.1f}s → {dur:.1f}s)"
                      + ("" if cut_share < 0.45 else " (a lot was removed; check the CUT lines)")))
    if lufs is not None and np.isfinite(lufs):
        checks.append(row(-16 <= lufs <= -12, f"Loudness {lufs:.1f} LUFS (target -14)"))
    checks.append(row(not timeline_words or covered > 0.3, f"Captions on screen {min(covered, 1):.0%} of the time"))
    checks.append(row(dark == 0, "No black frames" if dark == 0 else f"{dark} dark/black moments found"))
    if lines:
        checks.append(row(True, f"{len(lines)} story beats directed"))
    return '<div class="block-h" style="margin-top:12px">Checks</div><ul class="qa">' + "".join(checks) + "</ul>"


def render(state: dict, words: list[Word], settings: Settings, ctx: JobContext,
           progress: Callable[[float, str], None], plan: director.EditPlan | None = None,
           extra_cuts: list[tuple[float, float]] | None = None) -> dict:
    """Phases C + D: direct, grade, cut, camera, reframe, caption, overlays, mix, encode."""
    work = Path(state["work_dir"])
    video_path = Path(state["video"])

    if plan is None:
        progress(0.02, "AI Director: understanding the story")
        plan = direct(state, words, settings, ctx)
    director.apply_emphasis(words, plan)
    (work / "edit_plan.json").write_text(plan.to_json(), encoding="utf-8")
    settings, decisions = decide(settings, state, plan)
    if settings.autopilot:
        ctx.log("Autopilot decisions:")
        for dcs in decisions:
            ctx.log("  • " + dcs.replace("**", ""))

    source_path = video_path
    look = finishing.resolve_look(settings.look, plan.mood)
    if settings.color_grade or settings.stabilize or look != "None":
        progress(0.08, "Finishing: stabilize, grade, look")
        ctx.log("Finishing pass: " + ", ".join(x for x, on in (("stabilize", settings.stabilize),
                                                                ("grade", settings.color_grade),
                                                                (f"look '{look}'", look != "None")) if on))
        source_path = enhance_video(video_path, work / "graded.mp4", ctx, plan.mood, settings.grade_strength,
                                    look=look, stabilize=settings.stabilize, grade=settings.color_grade)

    source = VideoFileClip(str(source_path))
    try:
        if state["has_audio"]:
            source = source.with_audio(AudioFileClip(state["clean_wav"]))

        progress(0.15, "Smart cuts")
        segments = (compute_keep_segments(words, source.duration, settings.silence_threshold)
                    if settings.cut_silences and words else [(0.0, source.duration)])
        cuts: list[tuple[float, float]] = list(extra_cuts or [])
        if extra_cuts:
            ctx.log(f"Text edits: cutting {len(extra_cuts)} deleted word ranges.")
        if settings.smart_cuts:
            count_in = leading_count_in(words)
            if count_in:
                cuts.append((0.0, count_in[-1].end + 0.05))
                ctx.log(f"Trimming count-in / mic check: \"{' '.join(w.text for w in count_in)}\"")
        if settings.remove_fillers:
            fillers = [(w.start - 0.02, w.end + 0.02) for w in words if clean_token(w.text) in FILLER_WORDS]
            cuts += fillers
            if fillers:
                ctx.log(f"Removing {len(fillers)} filler sounds (um/uh/hmm).")
        if settings.smart_cuts:
            dropped = [l for l in plan.lines if not l.keep]
            cuts += [(l.start - 0.05, l.end + 0.1) for l in dropped]
            for l in dropped:
                ctx.log(f"Cutting retake / false start: \"{l.text[:60]}\"")
        else:
            for l in plan.lines:
                l.keep = True
        if cuts:
            segments = subtract_ranges(segments, cuts)
        if settings.hook_reorder and plan.hook_line:
            hook = plan.lines[plan.hook_line - 1]
            lo, hi = hook.start - 0.08, hook.end + 0.15
            hook_segs = intersect_ranges(segments, lo, hi)
            if hook_segs:
                segments = hook_segs + subtract_ranges(segments, [(lo, hi)])
                ctx.log(f"Cold open: moved line {hook.i} to the start as the hook: \"{hook.text[:60]}\"")
                after_hook = next((l for l in plan.lines if l.keep and l.i != hook.i), None)
                if after_hook and after_hook.transition == "none":
                    after_hook.transition = "whip"  # snap back into the story
        removed = source.duration - sum(e - s for s, e in segments)
        ctx.log(f"Edit: {len(segments)} segments, {removed:.1f}s removed "
                f"({source.duration:.1f}s -> {source.duration - removed:.1f}s).")
        if segments != [(0.0, source.duration)]:
            edited = concatenate_videoclips([source.subclipped(s, e) for s, e in segments])
            timeline_words = remap_words([w for w in words if clean_token(w.text) not in FILLER_WORDS
                                          or not settings.remove_fillers], segments)
        else:
            edited = source
            timeline_words = words
        lines = remap_plan(plan, segments)

        word_map_path = work / "word_map.json"
        save_word_map(timeline_words, state["lang"], word_map_path)

        face, camera = None, None
        if settings.face_tracking and (settings.camera_moves or settings.reframe_mode == "Center crop") \
                and not settings.draft:
            progress(0.2, "Tracking faces / active speaker")
            ctx.log("Tracking faces and who is speaking...")
            if settings.multi_speaker:
                face, people = effects.track_active_speaker(edited, timeline_words)
                if people >= 2:
                    ctx.log(f"{people} people on screen → camera follows whoever is talking.")
            else:
                face = effects.track_faces(edited)
        if settings.camera_moves and lines:
            cam_lines = [effects.CamLine(l.start, l.end, l.camera, l.transition if settings.transitions else "none",
                                         l.energy, emphasis_hit(l, timeline_words)) for l in lines]
            camera = effects.VirtualCamera(cam_lines, face, edited.duration, settings.camera_intensity)
            ctx.log("Virtual camera: " + ", ".join(f"{c}×{sum(1 for l in lines if l.camera == c)}"
                                                   for c in director.CAMERA if c != "none"
                                                   and any(l.camera == c for l in lines)))

        progress(0.3, "Reframing to 9:16")
        src_w, src_h = edited.size
        is_vertical = abs(src_w / src_h - TARGET_W / TARGET_H) < 0.03
        mode = "crop" if is_vertical or settings.reframe_mode == "Center crop" or src_w / src_h < TARGET_W / TARGET_H \
            else "blur"
        ctx.log(f"Reframing to {TARGET_W}x{TARGET_H} ({'face-following crop' if mode == 'crop' else 'blurred background'}"
                ", camera + scale in one pass)...")
        composer = effects.FrameComposer(camera, face, mode, (TARGET_W, TARGET_H))
        vertical = edited.transform(lambda gf, t: composer.apply(gf(t), t), apply_to=[])
        vertical.size = (TARGET_W, TARGET_H)

        behind_done = False
        if settings.hook_title and plan.title_hook and settings.title_style == "behind":
            model_path = MODELS_DIR / "u2netp.onnx"
            if model_path.exists():
                ins = analyzer.insight_from_dict(state.get("insight"))
                face_y = ins.face_cy if mode == "crop" else 0.5
                layer = segment.big_text_layer(plan.title_hook, (TARGET_W, TARGET_H),
                                               lambda sz: load_family_font(settings.font_family, sz),
                                               y_center=float(np.clip(face_y - 0.12, 0.16, 0.42)))
                behind = segment.TextBehind(segment.PersonMatte(model_path), layer, min(3.0, vertical.duration))
                vertical = vertical.transform(lambda gf, t: behind.apply(gf(t), t), apply_to=[])
                behind_done = True
                ctx.log(f"Hook title BEHIND the speaker: “{plan.title_hook}”")
            else:
                ctx.log("Person-cutout model missing (models/u2netp.onnx); using the title card instead.")

        progress(0.35, "Rendering captions & graphics")
        ctx.log("Rendering emotion-aware captions and graphics...")
        ass_path = work / "captions.ass"
        if timeline_words:
            write_captions_ass(timeline_words, settings, lines, vertical.duration, ass_path)
            ctx.log(f"Captions: libass, style '{settings.caption_style}', font {settings.font_family}")
        overlays = overlay_clips(lines, timeline_words, plan,
                                 replace(settings, hook_title=settings.hook_title and not behind_done),
                                 vertical.duration, ctx)
        cutaways = broll_clips(lines, settings, ctx)

        # The base video is the background clip: no full-frame blend under it.
        final = CompositeVideoClip([vertical, *cutaways, *overlays], size=(TARGET_W, TARGET_H),
                                   use_bgclip=True).with_duration(vertical.duration)

        progress(0.38, "Mixing sound design")
        events = sfx_events(lines, plan, settings, vertical.duration) if settings.sfx else []
        music = Path(settings.music_path) if settings.music_path else None
        if music is None and settings.auto_music:
            music = effects.pick_music(MUSIC_DIR, plan.mood, seed=video_path.name)
            if music:
                ctx.log(f"Music for a '{plan.mood}' mood: {music.parent.name}/{music.name}")
        if events or music or vertical.audio is not None:
            n = int(vertical.duration * effects.SR)
            speech = (vertical.audio.to_soundarray(fps=effects.SR) if vertical.audio is not None
                      else np.zeros((n, 2), dtype=np.float32))
            mixed = effects.mix_audio(speech, events, timeline_words, music, music_volume=settings.music_volume)
            mix_path = work / "mix.wav"
            sf.write(str(mix_path), mixed, effects.SR, subtype="PCM_16")
            final = final.with_audio(AudioFileClip(str(mix_path)))
            ctx.log(f"Sound design: {len(events)} SFX" + (" + ducked background music" if music else "")
                    + ", mastered to -14 LUFS")

        progress(0.4, "Encoding master reel")
        prefix = "draft" if settings.draft else "reel"
        out_path = OUTPUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
        fps = 15 if settings.draft else min(60, max(24, round(source.fps or 30)))
        ctx.log("Encoding DRAFT preview (15 fps, fast settings; face tracking skipped)..." if settings.draft
                else "Encoding final reel (this is the slow part)...")
        cover_path = None
        if settings.cover and not settings.draft:
            try:
                cover_path = publish.make_cover(vertical,
                                                plan.title_hook or "", settings.highlight_color,
                                                lambda sz: load_family_font(settings.font_family, sz),
                                                out_path.with_suffix(".cover.jpg"),
                                                min_t=3.3 if settings.hook_title else 0.5)  # skip the title section
                ctx.log(f"Cover image: {cover_path.name}")
            except Exception as exc:
                ctx.log(f"Cover skipped ({exc})")
        # 1) Clean master without captions (kept so captions can be edited later in seconds).
        CAPTION_CACHE.mkdir(parents=True, exist_ok=True)
        base_path = CAPTION_CACHE / f"{out_path.stem}.base.mp4"
        b_codec, b_preset, b_params = intermediate_profile(settings.draft)
        final.write_videofile(
            str(base_path), fps=fps, codec=b_codec, audio_codec="aac", audio_bitrate="192k", preset=b_preset,
            threads=os.cpu_count() or 4, ffmpeg_params=b_params,
            temp_audiofile=str(work / "temp_audio.m4a"), logger=None,
        )
        final.close()
        # 2) Burn the captions onto it with the delivery encoder settings.
        progress(0.85, "Burning captions")
        burn_captions(base_path, ass_path if timeline_words and ass_path.exists() else None, out_path, settings.draft)
        for c in cutaways:
            c.close()
        source_fps, source_duration = source.fps or 30, source.duration
        vertical_duration = vertical.duration
    finally:
        source.close()

    json_out = out_path.with_suffix(".json")
    shutil.copy(word_map_path, json_out)
    plan_out = out_path.with_suffix(".plan.json")
    shutil.copy(work / "edit_plan.json", plan_out)
    progress(0.93, "Publishing kit")
    extra_files, post_text = publishing_kit(out_path, video_path, state, plan, timeline_words, segments,
                                            source_fps, (src_w, src_h), source_duration, settings, ctx)
    if cover_path and cover_path.exists():
        extra_files.insert(0, cover_path)
    progress(0.97, "Quality check")
    ctx.log("Quality check...")
    try:
        qa = qa_report(out_path, source_duration, timeline_words, lines, settings)
    except Exception as exc:
        qa = f'<ul class="qa"><li class="warn"><span class="dot"></span>Checks could not run: {exc}</li></ul>'
    ctx.log(f"Done! Saved to {out_path}")
    progress(1.0, "Done")
    files = [out_path, *extra_files, json_out, plan_out]
    session = caption_session(out_path, base_path, timeline_words, lines, settings, vertical_duration, files)
    return {"video": out_path, "files": files, "qa": qa,
            "decisions": decisions, "plan": plan, "post": post_text, "captions": session}


CAPTION_CACHE = OUTPUT_DIR / ".cache"   # caption-free masters + caption sessions (for the caption editor)
CAPTION_FIELDS = ["font_family", "font_size", "primary_color", "highlight_color", "emotion_colors", "caption_style",
                  "caption_y", "words_per_caption"]


def intermediate_profile(draft: bool) -> tuple[str, str, list[str]]:
    """Encoder settings for the caption-free master: fast and visually lossless."""
    codec, _ = pick_encoder()
    if codec == "h264_nvenc":
        return codec, "p1", ["-rc", "vbr", "-cq", "16", "-pix_fmt", "yuv420p"]
    return "libx264", ("ultrafast" if draft else "veryfast"), ["-crf", "24" if draft else "14", "-pix_fmt", "yuv420p"]


def delivery_args(draft: bool) -> list[str]:
    """Final delivery encoder arguments (same quality as before the caption editor existed)."""
    codec, params = pick_encoder()
    if draft:
        params = [p for p in params if p not in ("-crf", "18")] + ["-crf", "30"]
        return ["-c:v", codec, "-preset", "ultrafast", "-b:v", "3M", *params]
    preset = "p5" if codec == "h264_nvenc" else "medium"
    return ["-c:v", codec, "-preset", preset, "-b:v", "10M", *params]


def burn_captions(base: Path, ass_path: Path | None, out_path: Path, draft: bool) -> None:
    vf = ["-vf", captions_ass.filter_arg(ass_path, FONTS_DIR)] if ass_path else []
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(base), *vf, *delivery_args(draft),
           "-c:a", "copy", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise PipelineError(f"Caption burn failed: {proc.stderr.strip()[-300:]}")


def caption_session(out_path: Path, base_path: Path, words: list[Word], lines: list, settings: Settings,
                    duration: float, files: list[Path]) -> dict | None:
    """Everything the caption editor needs, saved next to the clean master."""
    if not words:
        return None
    groups = group_words(words, settings.words_per_caption)
    session = {
        "video": str(out_path), "root": str(out_path), "base": str(base_path), "duration": duration,
        "draft": settings.draft,
        "srt": settings.export_srt, "files": [str(f) for f in files], "version": 1,
        "style": {k: getattr(settings, k) for k in CAPTION_FIELDS},
        "groups": [[w.__dict__.copy() for w in g] for g in groups],
        "lines": [asdict(l) for l in lines],
    }
    (CAPTION_CACHE / f"{out_path.stem}.captions.json").write_text(json.dumps(session, ensure_ascii=False),
                                                                     encoding="utf-8")
    return session


def caption_rows(session: dict | None) -> list[list]:
    if not session:
        return []
    rows = []
    for k, g in enumerate(session["groups"], 1):
        text = " ".join(f"*{w['text'].strip()}*" if w.get("highlight") else w["text"].strip() for w in g)
        rows.append([k, round(g[0]["start"], 2), round(g[-1]["end"], 2), text])
    return rows


def groups_from_rows(session: dict, rows) -> list[list[Word]]:
    """Rebuild caption groups from the editor table. Unchanged timing keeps Whisper's word timings."""
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    originals = {k: g for k, g in enumerate(session["groups"], 1)}
    groups = []
    for r in rows or []:
        if len(r) < 4 or not str(r[3] or "").strip():
            continue  # empty text = remove this caption
        try:
            start, end = float(r[1]), float(r[2])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        tokens = str(r[3]).split()
        orig = originals.get(int(r[0])) if str(r[0]).strip().isdigit() else None
        same_timing = orig and abs(orig[0]["start"] - start) < 0.02 and abs(orig[-1]["end"] - end) < 0.02
        if orig and same_timing and len(tokens) == len(orig):
            ws = [Word(t.strip("*"), w["start"], w["end"], len(t) > 2 and t.startswith("*") and t.endswith("*"))
                  for t, w in zip(tokens, orig)]
        else:
            ws = retime_tokens([t.strip("*") for t in tokens], start, end)
            for w, t in zip(ws, tokens):
                w.highlight = len(t) > 2 and t.startswith("*") and t.endswith("*")
        groups.append(ws)
    groups.sort(key=lambda g: g[0].start)
    return groups


def apply_caption_edits(session: dict, rows, style: str | None, size: int | None, pos: float | None) -> dict:
    """Re-burn edited captions onto the clean master. Returns the updated session (new version file)."""
    base = Path(session["base"])
    if not base.exists():
        raise PipelineError("The caption-free master for this render is gone — render the video again.")
    groups = groups_from_rows(session, rows)
    st = dict(session["style"])
    if style:
        st["caption_style"] = CAPTION_STYLES.get(style, st["caption_style"])
    if size:
        st["font_size"] = int(size)
    if pos is not None:
        st["caption_y"] = float(pos)
    settings = replace(Settings(), **st)
    lines = [director.LinePlan(**l) for l in session.get("lines", [])]
    version = int(session.get("version", 1)) + 1
    root = Path(session.get("root") or session["video"])  # versions are named after the original render
    out_path = root.with_name(f"{root.stem}_v{version}.mp4")
    ass_path = CAPTION_CACHE / f"{out_path.stem}.ass"
    if groups:
        write_captions_ass([w for g in groups for w in g], settings, lines, session["duration"], ass_path,
                           groups=groups)
    burn_captions(base, ass_path if groups else None, out_path, session.get("draft", False))
    new_files = [out_path]
    words = [w for g in groups for w in g]
    if session.get("srt") and words:
        blocks = publish.caption_blocks(words)
        new_files += [publish.write_srt(blocks, out_path.with_suffix(".srt")),
                      publish.write_vtt(blocks, out_path.with_suffix(".vtt"))]
    new = {**session, "video": str(out_path), "version": version, "style": st,
           "groups": [[w.__dict__.copy() for w in g] for g in groups],
           "files": [str(f) for f in new_files] + [f for f in session["files"] if f not in map(str, new_files)]}
    (CAPTION_CACHE / f"{root.stem}.captions.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    return new


def publishing_kit(out_path: Path, original: Path, state: dict, plan: director.EditPlan, words: list[Word],
                   segments: list[tuple[float, float]], fps: float, src_size: tuple[int, int], src_duration: float,
                   settings: Settings, ctx: JobContext) -> tuple[list[Path], str]:
    """Subtitles, English translation, post copy, pro XML and extra aspect ratios."""
    files: list[Path] = []
    post_text = ""
    if settings.draft:
        return files, post_text
    ai_ok = settings.use_ai and director.ollama_available(settings.llm_model)[0]
    blocks = publish.caption_blocks(words)
    if settings.export_srt and blocks:
        files.append(publish.write_srt(blocks, out_path.with_suffix(".srt")))
        files.append(publish.write_vtt(blocks, out_path.with_suffix(".vtt")))
        if settings.english_subs and state.get("lang") in ("hi", "gu") and ai_ok:
            en = publish.translate_blocks(blocks, director.ollama_json, settings.llm_model, ctx.log)
            if en:
                files.append(publish.write_srt(en, out_path.with_name(out_path.stem + ".en.srt")))
        ctx.log("Subtitles exported (SRT/VTT" + (" + English" if len(files) > 2 else "") + ").")
    if settings.post_copy:
        transcript = " ".join(w.text.strip() for w in words)
        copy = publish.post_copy(transcript, plan.summary, plan.content_type, settings.handle,
                                 director.ollama_json if ai_ok else None, settings.llm_model, ctx.log)
        files.append(publish.write_post_copy(copy, out_path.with_suffix(".post.txt")))
        post_text = (f"**Instagram caption**\n\n{copy['instagram_caption']}\n\n{' '.join(copy['hashtags'])}\n\n"
                     f"**YouTube Shorts title:** {copy['youtube_title']}\n\n{copy['youtube_description']}")
    if settings.export_xml and segments:
        files.append(publish.write_fcp_xml(original, segments, fps, src_size[0], src_size[1], src_duration,
                                           out_path.with_suffix(".premiere.xml")))
        ctx.log("Premiere / DaVinci XML exported (cuts on the original footage).")
    if settings.export_formats:
        files += publish.aspect_versions(out_path, ffmpeg_exe(), settings.export_formats, ctx.log)
    return files, post_text


def run_pipeline(video_path: Path, settings: Settings, ctx: JobContext,
                 progress: Callable[[float, str], None]) -> dict:
    """One-shot pipeline: listen, watch, understand, decide, edit, check."""
    state = analyze(video_path, settings, ctx, lambda f, d: progress(f * 0.35, d))
    plan = direct(state, state["words"], settings, ctx)
    result = render(state, state["words"], settings, ctx, lambda f, d: progress(0.35 + f * 0.65, d), plan)
    return {**result, "state": state}


def pick_encoder() -> tuple[str, list[str]]:
    """Use NVENC when an NVIDIA GPU + capable ffmpeg is present, else libx264."""
    base = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if DEVICE == "cuda":
        try:
            import subprocess

            out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=10).stdout
            if "h264_nvenc" in out:
                return "h264_nvenc", base + ["-rc", "vbr", "-cq", "20"]
        except Exception:
            pass
    return "libx264", base + ["-crf", "18"]


# --------------------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------------------

LANGUAGE_OPTIONS = {
    "Auto-detect (Roman script for Hindi/Gujarati)": "auto",
    "English": "en",
    "Hinglish (Roman script)": "hinglish",
    "Gujlish (Roman script)": "gujlish",
    "Hindi (Devanagari)": "hi",
    "Gujarati (Gujarati script)": "gu",
}
CAPTION_STYLES = captions_ass.CAPTION_STYLES

JOBS_ROOT = Path(tempfile.gettempdir()) / "goat_reel_jobs"


def new_work_dir() -> Path:
    JOBS_ROOT.mkdir(exist_ok=True)
    # Drop jobs older than a day.
    for old in JOBS_ROOT.iterdir():
        try:
            if time.time() - old.stat().st_mtime > 86400:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    d = JOBS_ROOT / uuid.uuid4().hex[:10]
    d.mkdir()
    return d


# Order of the settings controls passed from the UI into make_settings().
OPTION_FIELDS = ["cut_silences", "silence_threshold", "primary_color", "highlight_color", "model_size", "language",
                 "reframe_mode", "words_per_caption", "font_size", "denoise", "color_grade", "camera_moves",
                 "progress_bar", "use_ai", "llm_model", "caption_style", "emotion_colors", "face_tracking", "emoji",
                 "sfx", "hook_title", "transitions", "music_path", "music_volume", "font_family", "ai_spellfix",
                 "remove_fillers", "smart_cuts", "hook_reorder", "broll", "voice_polish", "auto_music", "logo_path",
                 "handle", "cta_text", "end_card", "draft", "autopilot", "use_vision", "look", "stabilize",
                 "title_style", "multi_speaker", "export_srt", "english_subs", "export_xml", "cover", "post_copy",
                 "export_formats"]
TITLE_STYLES = {"Auto": "auto", "Card on top": "card", "Behind the speaker": "behind"}


def make_settings(*values) -> Settings:
    v = dict(zip(OPTION_FIELDS, values))
    v["primary_color"] = COLOR_CHOICES.get(v["primary_color"], "#FFFFFF")
    v["highlight_color"] = COLOR_CHOICES.get(v["highlight_color"], "#FFE600")
    v["language"] = LANGUAGE_OPTIONS.get(v["language"], "auto")
    v["caption_style"] = CAPTION_STYLES.get(v["caption_style"], "auto")
    v["words_per_caption"] = int(v["words_per_caption"])
    v["font_size"] = int(v["font_size"])
    v["llm_model"] = (v["llm_model"] or director.DEFAULT_LLM).strip()
    v["music_path"] = v["music_path"] or None
    v["logo_path"] = v["logo_path"] or None
    v["handle"] = v["handle"] or ""
    v["cta_text"] = v["cta_text"] or ""
    v["draft"] = v["draft"] == "Draft preview" if isinstance(v["draft"], str) else bool(v["draft"])
    v["autopilot"] = "autopilot" in str(v["autopilot"]).lower() if isinstance(v["autopilot"], str) else \
        bool(v["autopilot"])
    v["title_style"] = TITLE_STYLES.get(v["title_style"], v["title_style"] if v["title_style"] in
                                        TITLE_STYLES.values() else "auto")
    v["export_formats"] = tuple(f.split(" ")[0].replace(":", "x") for f in (v["export_formats"] or []))
    v["look"] = v["look"] or "Auto (by mood)"
    return Settings(**v)


# Brand presets store the UI values of these controls.
PRESET_FIELDS = ["primary_color", "highlight_color", "caption_style", "font_family", "emotion_colors", "font_size",
                 "words_per_caption", "music_volume", "logo_path", "handle", "cta_text", "end_card"]


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def save_preset_ui(name, *values):
    name = re.sub(r"[^\w\- ]", "", (name or "").strip())[:40]
    if not name:
        return gr.update(), "Give the preset a name first."
    data = dict(zip(PRESET_FIELDS, values))
    if data.get("logo_path"):  # copy the logo next to the preset so it survives restarts
        src = Path(data["logo_path"])
        if src.exists():
            dst = PRESETS_DIR / f"{name}_logo{src.suffix.lower()}"
            shutil.copy(src, dst)
            data["logo_path"] = str(dst)
    (PRESETS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return gr.update(choices=list_presets(), value=name), f"Saved brand preset '{name}'."


def load_preset_ui(name):
    p = PRESETS_DIR / f"{name}.json"
    if not name or not p.exists():
        return [gr.update() for _ in PRESET_FIELDS] + ["Pick a preset to load."]
    data = json.loads(p.read_text(encoding="utf-8"))
    return [gr.update(value=data[k]) if k in data else gr.update() for k in PRESET_FIELDS] + [f"Loaded '{name}'."]


def stream_job(ctx: JobContext, job: Callable[[Callable[[float, str], None]], object], progress):
    """Run `job` in a thread, yielding ('log', text) while it runs and ('done', result) at the end."""
    import threading

    result: dict = {}
    state = {"frac": 0.0, "desc": "Starting"}

    def report(frac: float, desc: str) -> None:
        state["frac"], state["desc"] = frac, desc

    def worker() -> None:
        try:
            result["value"] = job(report)
        except BaseException as exc:
            result["error"] = exc
            result["tb"] = traceback.format_exc()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_len = -1
    while thread.is_alive():
        progress(state["frac"], desc=state["desc"])
        if len(ctx.logs) != last_len:
            last_len = len(ctx.logs)
            yield "log", ctx.text
        time.sleep(0.5)
    thread.join()
    if "error" in result:
        exc = result["error"]
        ctx.log(f"ERROR: {exc}" if isinstance(exc, PipelineError) else f"UNEXPECTED ERROR: {exc}\n{result['tb']}")
        yield "error", ctx.text
    else:
        yield "done", result["value"]


def stage_upload(video_file, ctx: JobContext) -> Path:
    if not video_file:
        raise PipelineError("Please upload a video first.")
    src = Path(video_file if isinstance(video_file, str) else video_file.name)
    if src.suffix.lower() not in (".mp4", ".mov", ".m4v", ".mkv", ".webm"):
        raise PipelineError(f"Unsupported file type: {src.suffix}")
    local = ctx.work_dir / f"input{src.suffix.lower()}"
    shutil.copy(src, local)
    return local


DOWNLOAD_DIR = APP_DIR / "downloads"


def download_from_link(url: str, cookies_file: str | None, ctx: JobContext,
                       on_progress: Callable[[dict], None] | None = None) -> Path:
    """Download a video from Instagram / YouTube / TikTok / X / Facebook etc. with yt-dlp."""
    import yt_dlp

    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        raise PipelineError("Paste a full link starting with http:// or https://")
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    report = on_progress or (lambda info: None)
    parts_done = {"n": 0}

    def hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            fmt = (d.get("info_dict") or {})
            kind = "audio" if fmt.get("vcodec") == "none" else "video"
            report({"stage": "downloading", "part": parts_done["n"] + 1, "kind": kind,
                    "fraction": (got / total) if total else None, "downloaded": got, "total": total,
                    "speed": d.get("speed"), "eta": d.get("eta"), "title": fmt.get("title", "")})
        elif status == "finished":
            parts_done["n"] += 1
            report({"stage": "processing"})
            ctx.log("Download finished, preparing video...")

    def postprocessor_hook(d: dict) -> None:
        if d.get("status") == "started":
            report({"stage": "processing"})

    opts = {
        # Best MP4-compatible video+audio, capped at 1080p (Reels/Shorts never need more).
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / "%(extractor)s_%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(Path(ffmpeg_exe()).parent),
        "progress_hooks": [hook],
        "postprocessor_hooks": [postprocessor_hook],
        "restrictfilenames": True,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    ctx.log(f"Fetching video from {url} ...")
    report({"stage": "connecting"})
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = info["entries"][0]
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc)
        if any(k in msg.lower() for k in ("login", "private", "cookies", "rate-limit", "authentication")):
            msg += ("\nThis post needs a logged-in session. Export cookies from your browser with a "
                    "'Get cookies.txt' extension and upload the file in 'Cookies (optional)'.")
        raise PipelineError(f"Could not download: {msg}") from exc
    if not path.exists():
        candidates = sorted(DOWNLOAD_DIR.glob(f"*{info.get('id', '')}*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise PipelineError("Download finished but the video file was not found.")
        path = candidates[-1]
    title = (info.get("title") or "").strip()
    ctx.log(f"Downloaded: {title[:80] or path.name} ({info.get('duration') or '?'}s, "
            f"{info.get('extractor_key', 'web')})")
    return path


def _mb(n: float | None) -> str:
    return f"{(n or 0) / 1_048_576:.1f} MB"


def import_progress_html(p: dict) -> str:
    """Progress card for link imports (connecting -> downloading -> merging -> done / failed)."""
    stage = p.get("stage")
    if stage == "connecting":
        return ('<div class="imp"><div class="imp-row"><span>Connecting and reading the link…</span></div>'
                '<div class="imp-bar indet"><i></i></div></div>')
    if stage == "downloading":
        frac = p.get("fraction")
        part = f"Part {p['part']} · {p['kind']}" if p.get("part", 1) > 1 or p.get("kind") == "audio" else p.get("kind", "video").capitalize()
        left = f"{_mb(p.get('downloaded'))} of {_mb(p.get('total'))}" if p.get("total") else _mb(p.get("downloaded"))
        speed = f"{_mb(p.get('speed'))}/s" if p.get("speed") else ""
        eta = p.get("eta")
        eta_s = f"{int(eta) // 60}:{int(eta) % 60:02d} left" if eta is not None else ""
        width = f"{frac * 100:.1f}%" if frac is not None else "100%"
        pct = f"{frac * 100:.0f}%" if frac is not None else ""
        title = f'<div class="imp-sub">{_esc(p.get("title", "")[:90])}</div>' if p.get("title") else ""
        return (f'<div class="imp"><div class="imp-row"><span>Downloading {_esc(part.lower())} <b>{pct}</b></span>'
                f'<span>{" · ".join(x for x in (left, speed, eta_s) if x)}</span></div>'
                f'<div class="imp-bar{"" if frac is not None else " indet"}"><i style="width:{width}"></i></div>'
                f"{title}</div>")
    if stage == "processing":
        return ('<div class="imp"><div class="imp-row"><span>Merging audio and video…</span></div>'
                '<div class="imp-bar indet"><i></i></div></div>')
    if stage == "done":
        return (f'<div class="imp done"><div class="imp-row"><span><span class="dot"></span> Imported '
                f'<b>{_esc(p.get("title", "")[:80])}</b></span><span>{_esc(p.get("meta", ""))}</span></div></div>')
    if stage == "failed":
        return (f'<div class="imp failed"><div class="imp-row"><span><span class="dot"></span> '
                f'{_esc(p.get("error", "Import failed")[:220])}</span></div></div>')
    return ""


def friendly_import_error(msg: str) -> str:
    m = msg.lower()
    if "http:// or https://" in m:
        return "Paste a full link that starts with https://"
    if "getaddrinfo" in m or "failed to resolve" in m or "timed out" in m or "connection" in m:
        return "Couldn't reach that link. Check the URL and your internet connection."
    if "unsupported url" in m:
        return "This site or type of link isn't supported."
    if any(k in m for k in ("login", "private", "cookies", "authentication", "sign in", "rate-limit")):
        return "This post needs a logged-in session. Add a cookies.txt file below and try again."
    if "not available" in m or "removed" in m or "404" in m:
        return "That video isn't available (removed, private or region-locked)."
    return "Import failed. Details are in the activity log."


def fetch_link_ui(url, cookies):
    """Download in a background thread and stream a live progress card to the UI."""
    import threading

    ctx = JobContext(work_dir=DOWNLOAD_DIR)
    latest: dict = {"stage": "connecting"}
    result: dict = {}

    def worker() -> None:
        try:
            result["path"] = download_from_link(url, cookies, ctx, on_progress=latest.update)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    shown = None
    while thread.is_alive():
        card = import_progress_html(dict(latest))
        if card != shown:
            shown = card
            yield gr.update(), ctx.text, card
        time.sleep(0.3)
    thread.join()
    if "error" in result:
        exc = result["error"]
        msg = str(exc) if isinstance(exc, PipelineError) else f"Unexpected error: {exc}"
        ctx.log(f"ERROR: {msg}")  # full technical detail stays in the activity log
        yield gr.update(), ctx.text, import_progress_html({"stage": "failed", "error": friendly_import_error(msg)})
        return
    path = result["path"]
    size = path.stat().st_size if path.exists() else 0
    title = ctx.logs[-1].split("Downloaded: ", 1)[-1].rsplit(" (", 1)[0] if ctx.logs else path.name
    yield str(path), ctx.text, import_progress_html({"stage": "done", "title": title, "meta": _mb(size)})


def _esc(text: str) -> str:
    import html

    return html.escape(str(text or ""))


def _bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _esc(text))


def analysis_html(state: dict | None, plan: director.EditPlan | None = None, decisions: list[str] | None = None) -> str:
    """What the editor understood: summary grid, story, scenes, cuts and the reasoning behind each choice."""
    if not state:
        return ('<div class="empty">Run an edit to see how the video was read: format, mood, pacing, '
                'what is on screen, and the reason behind every editing choice.</div>')
    ins = analyzer.insight_from_dict(state.get("insight"))
    lang = {"hi": "Hindi / Hinglish", "gu": "Gujarati / Gujlish", "en": "English"}.get(state.get("lang"), state.get("lang"))
    fmt = lambda v: str(v).replace("_", " ").capitalize() if v else "—"  # noqa: E731
    cells = [
        ("Format", fmt(plan.content_type) if plan else "—"), ("Mood", fmt(plan.mood) if plan else "—"),
        ("Pace", fmt(plan.pace) if plan else "—"), ("Shot", fmt(ins.shot)),
        ("Language", lang or "—"), ("Duration", f"{ins.duration:.0f} s"),
        ("Speech", f"{ins.speech_coverage:.0%} · {ins.words_per_min:.0f} wpm"),
        ("Director", (plan.source.split(":", 1)[1] if plan and plan.source.startswith("ai:") else "Rules")
         if plan else "—"),
    ]
    out = ['<div class="grid">']
    out += [f'<div class="cell"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>' for k, v in cells]
    out.append("</div>")
    if plan and plan.summary:
        out.append(f'<div class="block-h">Story</div><div class="prose">{_esc(plan.summary)}</div>')
    items = []
    if plan and plan.title_hook:
        items.append(f"Hook title: <b>{_esc(plan.title_hook)}</b>")
    if plan and plan.hook_line:
        items.append(f"Cold open from line {plan.hook_line}: “{_esc(plan.lines[plan.hook_line - 1].text[:90])}”")
    if items:
        out.append('<ul class="list" style="margin-top:12px">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    if ins.scenes:
        out.append('<div class="block-h">On screen</div><ul class="list">'
                   + "".join(f"<li>{_esc(sc)}</li>" for sc in ins.scenes) + "</ul>")
    if plan:
        cut = [l for l in plan.lines if not l.keep]
        if cut:
            out.append('<div class="block-h">Removed lines</div><ul class="list">'
                       + "".join(f"<li>{_esc(l.text[:110])}</li>" for l in cut) + "</ul>")
    if decisions:
        out.append('<div class="block-h">Decisions</div><ul class="list">'
                   + "".join(f"<li>{_bold(dc)}</li>" for dc in decisions) + "</ul>")
    return "".join(out)


def list_outputs() -> list[str]:
    files = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files[:50]]


# Components every handler can update, in this order (see pack()).
OUT_NAMES = ["result", "files", "logs", "status", "analysis", "qa", "script", "plan_table", "title_box", "state",
             "post", "cap_state", "cap_table", "cap_player", "cap_style", "cap_size", "cap_pos"]


def caption_outputs(session: dict | None) -> dict:
    """Values that load a finished render into the caption editor."""
    if not session:
        return {"cap_state": None, "cap_table": [], "cap_player": None}
    st = session["style"]
    label = next((k for k, v in CAPTION_STYLES.items() if v == st["caption_style"]), "Auto (by emotion)")
    return {"cap_state": session, "cap_table": caption_rows(session), "cap_player": session["video"],
            "cap_style": label, "cap_size": st["font_size"], "cap_pos": round(st["caption_y"], 2)}


def pack(**kw) -> list:
    return [kw.get(n, gr.update()) for n in OUT_NAMES]


def status_line(ctx: JobContext) -> str:
    last = ctx.logs[-1].split("] ", 1)[-1] if ctx.logs else "Starting..."
    return f'<div class="status running"><span class="dot"></span>{_esc(last[:140])}</div>'


DONE = '<div class="status done"><span class="dot"></span>Finished. Files are listed under the preview.</div>'
FAILED = '<div class="status failed"><span class="dot"></span>Stopped. Details are in the activity log.</div>'


def serial_state(state: dict, plan: director.EditPlan | None = None) -> dict:
    st = {**state, "words": [w.__dict__ if isinstance(w, Word) else w for w in state.get("words", [])]}
    if plan is not None:
        st["plan"] = asdict(plan)
    return st


def caption_apply_ui(session, rows, style, size, pos, progress=gr.Progress()):
    if not session:
        return (gr.update(), gr.update(), gr.update(), session,
                '<div class="status failed"><span class="dot"></span>Render a video first.</div>')
    progress(0.2, desc="Burning captions")
    t0 = time.time()
    try:
        new = apply_caption_edits(session, rows, style, size, pos)
    except Exception as exc:
        return (gr.update(), gr.update(), gr.update(), session,
                f'<div class="status failed"><span class="dot"></span>{_esc(str(exc)[:200])}</div>')
    msg = (f'<div class="status done"><span class="dot"></span>Saved {_esc(Path(new["video"]).name)} '
           f'in {time.time() - t0:.0f} s. The previous version is kept.</div>')
    return new["video"], new["video"], new["files"], new, msg


def caption_reset_ui(session):
    return caption_rows(session)


def caption_seek(rows, evt: gr.SelectData):
    """Row clicked in the caption table -> time to jump the preview to."""
    try:
        r = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        data = rows.values.tolist() if hasattr(rows, "values") else rows
        return float(data[r][1]) + 0.01
    except Exception:
        return gr.update()


def auto_ui(video_file, *opts, progress=gr.Progress()):
    """The main button: listen, watch, understand, decide, edit, check — no questions asked."""
    settings = make_settings(*opts)
    ctx = JobContext(work_dir=new_work_dir())
    try:
        local = stage_upload(video_file, ctx)
    except PipelineError as exc:
        yield pack(logs=str(exc), status=FAILED)
        return
    for kind, payload in stream_job(ctx, lambda rep: run_pipeline(local, settings, ctx, rep), progress):
        if kind == "log":
            yield pack(logs=payload, status=status_line(ctx))
        elif kind == "error":
            yield pack(logs=payload, status=FAILED)
        else:
            st, plan = payload["state"], payload["plan"]
            yield pack(result=str(payload["video"]), files=[str(f) for f in payload["files"]], logs=ctx.text,
                       status=DONE, analysis=analysis_html(st, plan, payload["decisions"]), qa=payload["qa"],
                       post=payload.get("post", ""), **caption_outputs(payload.get("captions")),
                       script=words_to_script(st["words"], settings.words_per_caption),
                       plan_table=plan_to_rows(plan), title_box=plan.title_hook, state=serial_state(st, plan))


def transcribe_ui(video_file, *opts, progress=gr.Progress()):
    settings = make_settings(*opts)
    ctx = JobContext(work_dir=new_work_dir())
    try:
        local = stage_upload(video_file, ctx)
    except PipelineError as exc:
        yield pack(logs=str(exc), status=FAILED)
        return
    for kind, payload in stream_job(ctx, lambda rep: analyze(local, settings, ctx, rep), progress):
        if kind == "log":
            yield pack(logs=payload, status=status_line(ctx))
        elif kind == "error":
            yield pack(logs=payload, status=FAILED)
        else:
            ctx.log("Transcript ready — fix any wrong words, then run '2 · Direct'.")
            yield pack(logs=ctx.text, status=DONE, script=words_to_script(payload["words"], settings.words_per_caption),
                       analysis=analysis_html(payload), state=serial_state(payload))


def direct_ui(state, script, *opts, progress=gr.Progress()):
    if not state:
        yield pack(logs="Run '1 · Transcribe' first.", status=FAILED)
        return
    settings = make_settings(*opts)
    ctx = JobContext(work_dir=Path(state["work_dir"]))
    words = script_to_words(script or "")
    for kind, payload in stream_job(ctx, lambda rep: direct(state, words, settings, ctx), progress):
        if kind == "log":
            yield pack(logs=payload, status=status_line(ctx))
        elif kind == "error":
            yield pack(logs=payload, status=FAILED)
        else:
            _, decisions = decide(settings, state, payload)
            ctx.log("Edit plan ready — tweak anything in the table, then '3 · Render'.")
            yield pack(logs=ctx.text, status=DONE, analysis=analysis_html(state, payload, decisions),
                       plan_table=plan_to_rows(payload), title_box=payload.title_hook,
                       state={**state, "plan": asdict(payload)})


def render_ui(state, script, plan_rows, title, *opts, progress=gr.Progress()):
    if not state:
        yield pack(logs="Run Auto Edit or '1 · Transcribe' first.", status=FAILED)
        return
    settings = make_settings(*opts)
    ctx = JobContext(work_dir=Path(state["work_dir"]))
    original = [Word(**w) if isinstance(w, dict) else w for w in state.get("words", [])]
    if original and script and script.strip():
        words, text_cuts = text_edit(original, script)
    else:
        words, text_cuts = script_to_words(script or ""), []
    if script and script.strip() and not words:
        yield pack(logs="Could not read the caption lines. Keep the format:  0.00 - 1.20 | your words", status=FAILED)
        return
    plan = merge_plan_edits(plan_from_dict(state["plan"]), plan_rows, title) if state.get("plan") else None
    ctx.log(f"Re-rendering with your edits ({len(words)} caption words, {len(text_cuts)} text cuts).")
    for kind, payload in stream_job(ctx, lambda rep: render(state, words, settings, ctx, rep, plan, text_cuts),
                                    progress):
        if kind == "log":
            yield pack(logs=payload, status=status_line(ctx))
        elif kind == "error":
            yield pack(logs=payload, status=FAILED)
        else:
            yield pack(result=str(payload["video"]), files=[str(f) for f in payload["files"]], logs=ctx.text,
                       status=DONE, qa=payload["qa"], post=payload.get("post", ""),
                       **caption_outputs(payload.get("captions")),
                       analysis=analysis_html(state, payload["plan"], payload["decisions"]),
                       state={**state, "plan": asdict(payload["plan"])})


# ---------------- Projects (save / reopen an edit)

def list_projects() -> list[str]:
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "project.json").exists())


def save_project_ui(name, state, script, plan_rows, title, *opts):
    name = re.sub(r"[^\w\- ]", "", (name or "").strip())[:50]
    if not name or not state:
        return gr.update(), "Name the project, and transcribe or auto-edit a video first."
    pdir = PROJECTS_DIR / name
    media = pdir / "media"
    media.mkdir(parents=True, exist_ok=True)
    st = dict(state)
    for key in ("video", "clean_wav"):  # copy media so the project survives temp cleanup
        src = Path(st.get(key, ""))
        if src.exists():
            dst = media / src.name
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)
            st[key] = str(dst)
    st["work_dir"] = str(media)
    rows = plan_rows.values.tolist() if hasattr(plan_rows, "values") else plan_rows
    data = {"state": st, "script": script, "plan_rows": rows, "title": title,
            "options": [o if not isinstance(o, tuple) else list(o) for o in opts], "saved": time.strftime("%Y-%m-%d %H:%M")}
    (pdir / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return gr.update(choices=list_projects(), value=name), f"Saved project '{name}'."


def open_project_ui(name):
    p = PROJECTS_DIR / str(name) / "project.json"
    if not name or not p.exists():
        return [gr.update()] * (6 + len(OPTION_FIELDS)) + ["Pick a project."]
    data = json.loads(p.read_text(encoding="utf-8"))
    st = data["state"]
    plan = plan_from_dict(st["plan"]) if st.get("plan") else None
    opts = data.get("options", [])
    opt_updates = [gr.update(value=v) for v in opts] if len(opts) == len(OPTION_FIELDS) else \
        [gr.update()] * len(OPTION_FIELDS)
    return ([st, data.get("script", ""), data.get("plan_rows") or [], data.get("title", ""),
             analysis_html(st, plan), st.get("video")] + opt_updates + [f"Opened '{name}' (saved {data.get('saved')})."])


# ---------------- Batch processing (+ watch folder)

BATCH_INDEX = OUTPUT_DIR / "batch_index.json"


def _batch_key(p: Path) -> str:
    st = p.stat()
    return f"{p.name}|{st.st_size}|{int(st.st_mtime)}"


def batch_ui(files, folder, watch, *opts, progress=gr.Progress()):
    settings = make_settings(*opts)
    index = json.loads(BATCH_INDEX.read_text(encoding="utf-8")) if BATCH_INDEX.exists() else {}
    rows: list[list] = []
    seen: set[str] = set()

    def pending() -> list[Path]:
        cands = [Path(f if isinstance(f, str) else f.name) for f in (files or [])]
        if folder and Path(folder).is_dir():
            cands += sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in effects.VIDEO_EXT)
        return [p for p in cands if p.exists() and _batch_key(p) not in seen]

    while True:
        queue = pending()
        for p in queue:
            key = _batch_key(p)
            seen.add(key)
            if key in index and Path(index[key]).exists():  # resume: already done
                rows.append([p.name, "Done earlier", Path(index[key]).name, ""])
                yield rows, gr.update(choices=list_outputs())
                continue
            rows.append([p.name, "Editing", "", ""])
            yield rows, gr.update()
            ctx = JobContext(work_dir=new_work_dir())
            t0 = time.time()
            try:
                local = ctx.work_dir / f"input{p.suffix.lower()}"
                shutil.copy(p, local)
                res = run_pipeline(local, settings, ctx, lambda f, d: None)
                index[key] = str(res["video"])
                BATCH_INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
                rows[-1] = [p.name, "Done", Path(res["video"]).name, f"{time.time() - t0:.0f}s"]
            except Exception as exc:
                rows[-1] = [p.name, f"Failed: {str(exc)[:80]}", "", f"{time.time() - t0:.0f}s"]
            yield rows, gr.update(choices=list_outputs())
        if not watch:
            break
        time.sleep(10)  # watch folder: check for new files until Stop is pressed
        if not rows or rows[-1][1] != "Watching folder":
            rows.append(["", "Watching folder", "", ""])
            yield rows, gr.update()
    if not rows:
        rows = [["", "No videos found. Add files or a folder path.", "", ""]]
    yield rows, gr.update(choices=list_outputs())


# ---------------- Long video -> viral clips

def clips_ui(video_file, n_clips, *opts, progress=gr.Progress()):
    settings = make_settings(*opts)
    ctx = JobContext(work_dir=new_work_dir())
    try:
        local = stage_upload(video_file, ctx)
    except PipelineError as exc:
        yield [], ctx.text + str(exc), gr.update()
        return
    ok, why = director.ollama_available(settings.llm_model)
    if not ok:
        yield [], f"The clip finder needs the AI director: {why}", gr.update()
        return

    def job(report):
        report(0.02, "Transcribing the long video")
        duration = validate_video(local)
        ctx.log(f"Long video: {duration / 60:.1f} min. Transcribing (this is the long part)...")
        raw = ctx.work_dir / "raw.wav"
        if not extract_audio(local, raw, ctx):
            raise PipelineError("The video has no audio to find clips in.")
        words, lang = transcribe(raw, settings, ctx)
        report(0.4, "Finding viral moments")
        ideas = director.find_viral_moments(words, settings.llm_model, ctx.log)[: int(n_clips)]
        if not ideas:
            raise PipelineError("No standalone 20-60s moments found.")
        ctx.log(f"Found {len(ideas)} clips: " + "; ".join(f"{i.title} ({i.score})" for i in ideas))
        results = []
        for k, idea in enumerate(ideas, 1):
            report(0.4 + 0.6 * (k - 1) / len(ideas), f"Editing clip {k}/{len(ideas)}: {idea.title}")
            ctx.log(f"--- Clip {k}: {idea.title} [{idea.start:.0f}s-{idea.end:.0f}s] score {idea.score}")
            clip_path = ctx.work_dir / f"clip_{k}.mp4"
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{idea.start:.2f}", "-to",
                            f"{idea.end:.2f}", "-i", str(local), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                            "-c:a", "aac", "-b:a", "192k", str(clip_path)], check=True)
            sub_words = [Word(w.text, w.start - idea.start, w.end - idea.start, w.highlight) for w in words
                         if idea.start <= w.start < idea.end]
            sub_ctx = JobContext(work_dir=new_work_dir())
            sub_ctx.logs = ctx.logs  # one shared log
            st = analyze(clip_path, settings, sub_ctx, lambda f, d: None, words_override=sub_words, lang_override=lang)
            plan = direct(st, st["words"], settings, sub_ctx)
            if idea.title and not plan.title_hook:
                plan.title_hook = idea.title
            res = render(st, st["words"], settings, sub_ctx, lambda f, d: None, plan)
            results.append((idea, res))
        return results

    for kind, payload in stream_job(ctx, job, progress):
        if kind in ("log", "error"):
            yield gr.update(), payload, gr.update()
        else:
            rows = [[k, idea.score, idea.title, f"{idea.end - idea.start:.0f}s", idea.reason, Path(r["video"]).name]
                    for k, (idea, r) in enumerate(payload, 1)]
            yield rows, ctx.text, gr.update(choices=list_outputs(), value=Path(payload[0][1]["video"]).name)


ICONS = APP_DIR / "assets" / "icons"


def icon(name: str, variant: str = "light") -> str:
    return str(ICONS / f"{name}-{variant}.svg")


CSS = """
:root {
  --bg: #09090B; --panel: #111113; --panel-2: #16161A; --line: #232328; --line-2: #2E2E34;
  --text: #EDEDEF; --muted: #8B8B94; --faint: #5F5F68; --accent: #FF5C35; --ok: #3DD68C; --warn: #F5A524;
}
body, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; padding: 0 20px 40px !important; }
footer { display: none !important; }

/* ---------- top bar ---------- */
#topbar { display:flex; align-items:center; justify-content:space-between; padding: 18px 2px 16px;
  border-bottom: 1px solid var(--line); margin-bottom: 18px; }
#topbar .brand { display:flex; align-items:center; gap:10px; font-size:15px; font-weight:600; color:var(--text);
  letter-spacing:-0.01em; }
#topbar .brand .sub { color: var(--faint); font-weight: 500; }
#topbar .chips { display:flex; gap:16px; font-size:12.5px; color: var(--muted); }
#topbar .chip { display:flex; align-items:center; gap:7px; }
.dot { width:7px; height:7px; border-radius:50%; background: var(--faint); display:inline-block; flex: none; }
.dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(61,214,140,.12); }
.dot.off { background: #55555c; }

/* ---------- section headers ---------- */
.sec { font-size: 13px; font-weight: 600; color: var(--text); margin: 2px 0 10px; display:flex;
  align-items:baseline; gap:8px; }
.sec span { font-weight: 400; color: var(--faint); font-size: 12.5px; }
.muted, .muted p { color: var(--muted) !important; font-size: 13px !important; line-height: 1.55; }

/* ---------- surfaces ---------- */
.surface { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
  padding: 16px !important; }
.block, .form { border-color: var(--line) !important; }

/* ---------- buttons ---------- */
button.primary, .primary > button { background: #EDEDEF !important; color: #0B0B0D !important; border: none !important;
  font-weight: 600 !important; box-shadow: none !important; }
button.primary:hover { background: #FFFFFF !important; }
button.secondary { background: var(--panel-2) !important; border: 1px solid var(--line-2) !important;
  color: var(--text) !important; font-weight: 500 !important; box-shadow: none !important; }
button.secondary:hover { border-color: #3d3d45 !important; background: #1b1b20 !important; }
#go { height: 46px; font-size: 14.5px !important; border-radius: 10px !important; }
button img, button svg { width: 16px !important; height: 16px !important; }

/* ---------- segmented controls (radios) ---------- */
.seg .wrap { display:flex !important; gap:3px !important; padding:3px !important; background: #0D0D10 !important;
  border: 1px solid var(--line) !important; border-radius: 9px !important; }
.seg .wrap label { flex: 1; justify-content: center; margin: 0 !important; border: none !important;
  background: transparent !important; border-radius: 6px !important; padding: 7px 10px !important;
  color: var(--muted) !important; font-size: 13px !important; box-shadow: none !important; }
.seg .wrap label.selected { background: #232329 !important; color: var(--text) !important; }
.seg .wrap label input { display: none !important; }

/* ---------- tabs ---------- */
.tabs .tab-wrapper, .tab-nav { border-bottom: 1px solid var(--line) !important; }
button[role="tab"] { color: var(--muted) !important; font-weight: 500 !important; font-size: 13.5px !important;
  border: none !important; background: transparent !important; }
button[role="tab"]:hover { color: var(--text) !important; }
button[role="tab"].selected { color: var(--text) !important; }
button[role="tab"].selected::after { background: var(--accent) !important; height: 2px !important; }

/* ---------- status line ---------- */
.status { display:flex; align-items:center; gap:9px; padding: 10px 12px; border-radius: 9px; font-size: 13px;
  color: var(--muted); background: #0D0D10; border: 1px solid var(--line); }
.status.running { color: var(--text); }
.status.running .dot { background: var(--warn); animation: pulse 1.4s ease-in-out infinite; }
.status.done .dot { background: var(--ok); } .status.failed .dot { background: #F04438; }
@keyframes pulse { 50% { opacity: .35; } }

/* ---------- analysis ---------- */
.grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
.cell { background: var(--panel); padding: 14px 16px; }
.cell .k { font-size: 12px; color: var(--faint); }
.cell .v { font-size: 15px; font-weight: 500; color: var(--text); margin-top: 4px; }
.block-h { font-size: 13px; font-weight: 600; color: var(--text); margin: 18px 0 8px; }
.prose { color: #C9C9CF; font-size: 14px; line-height: 1.6; }
.list { list-style: none; padding: 0; margin: 0; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.list li { padding: 10px 14px; font-size: 13.5px; color: #C9C9CF; border-top: 1px solid var(--line); line-height: 1.5; }
.list li:first-child { border-top: none; }
.list b { color: var(--text); font-weight: 600; }
.empty { color: var(--muted); font-size: 13.5px; padding: 36px; text-align: center; border: 1px dashed var(--line-2);
  border-radius: 12px; }

/* ---------- quality check ---------- */
.qa { list-style: none; padding: 0; margin: 4px 0 0; }
.qa li { display:flex; align-items:center; gap:9px; font-size: 13px; color: #C9C9CF; padding: 5px 0; }
.qa li.warn { color: var(--warn); } .qa li .dot { background: var(--ok); } .qa li.warn .dot { background: var(--warn); }

/* ---------- link import progress ---------- */
.imp { border: 1px solid var(--line); background: #0D0D10; border-radius: 10px; padding: 11px 13px; margin-top: 4px; }
.imp-row { display:flex; justify-content:space-between; gap:10px; font-size: 13px; color: var(--muted); }
.imp-row b { color: var(--text); font-weight: 600; }
.imp-sub { font-size: 12px; color: var(--faint); margin-top: 6px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.imp-bar { height: 4px; background: #1d1d22; border-radius: 99px; overflow: hidden; margin-top: 9px; position: relative; }
.imp-bar i { display:block; height:100%; background: var(--accent); border-radius: 99px; transition: width .3s ease; }
.imp-bar.indet i { width: 35% !important; position:absolute; animation: indet 1.2s ease-in-out infinite; }
@keyframes indet { 0% { left: -35%; } 100% { left: 100%; } }
.imp.done .dot { background: var(--ok); } .imp.failed { border-color: #5a2a2a; }
.imp.failed .dot { background: #F04438; } .imp.failed .imp-row { color: #f3a3a3; }

#logs textarea, #script textarea { font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: 12.5px !important;
  line-height: 1.55 !important; }
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0,1fr)); } #topbar .chips { display:none; } }
"""

FORCE_DARK = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); window.location.replace(u.href); }
  // "Edit captions" button: open the Captions tab and bring it into view.
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#edit_caps')) return;
    const tab = [...document.querySelectorAll('button[role="tab"]')].find(b => b.textContent.trim() === 'Captions');
    if (tab) tab.click();
    setTimeout(() => document.getElementById('workspace')?.scrollIntoView({behavior: 'smooth'}), 80);
  });
}
"""

LOGO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none"><rect x="1" y="1" width="22" height="22" rx="6" '
        'fill="#EDEDEF"/><path d="M9.5 7.5v9l7-4.5z" fill="#0B0B0D"/></svg>')


def build_theme() -> gr.themes.Base:
    accent = gr.themes.Color(c50="#fff1ec", c100="#ffe0d6", c200="#ffc2ad", c300="#ff9f80", c400="#ff7a55",
                             c500="#FF5C35", c600="#e84a24", c700="#c23a1a", c800="#9a2f17", c900="#7d2915",
                             c950="#431208")
    return gr.themes.Base(
        primary_hue=accent, neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_md, text_size=gr.themes.sizes.text_sm,
    ).set(
        body_background_fill_dark="#09090B", background_fill_primary_dark="#111113",
        background_fill_secondary_dark="#0D0D10", block_background_fill_dark="#111113",
        block_border_color_dark="#232328", border_color_primary_dark="#232328", block_border_width="1px",
        block_label_background_fill_dark="#111113", block_label_text_color_dark="#8B8B94",
        block_title_text_color_dark="#C9C9CF", block_title_text_weight="500", block_label_text_weight="500",
        input_background_fill_dark="#0D0D10", input_border_color_dark="#2A2A30", input_border_color_focus_dark="#4a4a52",
        checkbox_background_color_selected_dark="#FF5C35", checkbox_border_color_selected_dark="#FF5C35",
        slider_color_dark="#FF5C35", block_shadow="none", block_shadow_dark="none",
        button_primary_background_fill_dark="#EDEDEF", button_primary_text_color_dark="#0B0B0D",
        button_primary_background_fill_hover_dark="#FFFFFF", button_secondary_background_fill_dark="#16161A",
        button_secondary_text_color_dark="#EDEDEF", button_secondary_border_color_dark="#2E2E34",
        table_even_background_fill_dark="#111113", table_odd_background_fill_dark="#0F0F12",
        table_border_color_dark="#232328", body_text_color_dark="#EDEDEF", body_text_color_subdued_dark="#8B8B94",
    )


def header_html() -> str:
    ai_ok, _ = director.ollama_available(director.DEFAULT_LLM)
    vision_ok = analyzer.vision_available()

    def chip(ok: bool, label: str) -> str:
        return f'<span class="chip"><span class="dot {"ok" if ok else "off"}"></span>{label}</span>'

    return (f'<div id="topbar"><div class="brand">{LOGO}GOAT <span class="sub">Studio</span></div>'
            f'<div class="chips">{chip(True, DEVICE.upper())}{chip(ai_ok, "Director")}{chip(vision_ok, "Vision")}'
            f'{chip(True, "Local only")}</div></div>')


def sec(title: str, sub: str = "") -> gr.HTML:
    return gr.HTML(f'<div class="sec">{_esc(title)}{f"<span>{_esc(sub)}</span>" if sub else ""}</div>')


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GOAT Studio", css=CSS, theme=build_theme(), js=FORCE_DARK) as demo:
        gr.HTML(header_html())
        job_state = gr.State(None)

        with gr.Row(equal_height=False):
            # ---------------- Source
            with gr.Column(scale=5):
                sec("Source", "Upload a file or import from a link")
                with gr.Tabs():
                    with gr.Tab("Upload"):
                        original = gr.Video(show_label=False, sources=["upload"], height=470)
                    with gr.Tab("Import link"):
                        link = gr.Textbox(placeholder="Paste a YouTube, Instagram, TikTok, X or Facebook URL",
                                          show_label=False)
                        fetch_btn = gr.Button("Import", variant="secondary", icon=icon("link"))
                        import_status = gr.HTML("")
                        cookies = gr.File(label="cookies.txt (private posts only)", file_types=[".txt"],
                                          type="filepath", height=110)
                        gr.Markdown("Imported videos open in the Upload tab.", elem_classes="muted")

            # ---------------- Controls
            with gr.Column(scale=3, min_width=290):
                sec("Edit")
                mode = gr.Radio(["Autopilot", "Manual"], value="Autopilot", label="Mode", elem_classes="seg",
                                info="Autopilot chooses pacing, framing, captions and sound for this video.")
                quality = gr.Radio(["Final 1080p", "Draft preview"], value="Final 1080p", label="Quality",
                                   elem_classes="seg", info="Draft is about twice as fast.")
                go = gr.Button("Auto edit", variant="primary", elem_id="go", icon=icon("sparkle", "dark"))
                status = gr.HTML('<div class="status"><span class="dot"></span>Ready</div>')
                with gr.Accordion("Step by step", open=False):
                    gr.Markdown("Review the transcript and the edit plan between steps.", elem_classes="muted")
                    transcribe_btn = gr.Button("Transcribe", variant="secondary", size="sm", icon=icon("text"))
                    direct_btn = gr.Button("Plan edit", variant="secondary", size="sm", icon=icon("clapper"))
                    render_btn = gr.Button("Render", variant="secondary", size="sm", icon=icon("play"))

            # ---------------- Output
            with gr.Column(scale=5):
                sec("Output", "1080 × 1920 · -14 LUFS")
                result = gr.Video(show_label=False, interactive=False, show_download_button=True, height=470)
                out_files = gr.File(label="Files", file_count="multiple", interactive=False, height=140)
                # Opens the Captions tab via the page script in FORCE_DARK (no server round-trip).
                gr.Button("Edit captions", variant="secondary", icon=icon("text"), elem_id="edit_caps")
                qa = gr.HTML("")
                with gr.Accordion("Post copy", open=False):
                    post = gr.Markdown("Caption, hashtags and a YouTube title appear here after a final render.",
                                       elem_classes="muted")

        gr.HTML('<div id="workspace" style="height:10px"></div>')
        with gr.Tabs():
            with gr.Tab("Analysis", id="analysis"):
                analysis = gr.HTML(analysis_html(None))
            with gr.Tab("Captions", id="captions"):
                cap_state = gr.State(None)
                cap_seek_t = gr.Number(visible=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        cap_player = gr.Video(show_label=False, interactive=False, height=560, elem_id="cap_player")
                    with gr.Column(scale=3):
                        gr.Markdown("Click a caption to jump to it. Change the words or timing, wrap a word in "
                                    "*stars* to highlight it, clear a caption's text to remove it, or add a row. "
                                    "Apply re-draws only the captions, so it takes seconds, and every version "
                                    "is kept.", elem_classes="muted")
                        cap_table = gr.Dataframe(headers=["#", "Start", "End", "Caption"], interactive=True,
                                                 datatype=["number", "number", "number", "str"], wrap=True,
                                                 column_widths=["8%", "13%", "13%", "66%"], label="Captions",
                                                 elem_id="cap_table")
                        with gr.Row():
                            cap_style = gr.Dropdown(list(CAPTION_STYLES), value="Auto (by emotion)", label="Animation")
                            cap_size = gr.Slider(60, 140, value=96, step=4, label="Size")
                            cap_pos = gr.Slider(0.15, 0.85, value=0.66, step=0.01, label="Position (from top)")
                        with gr.Row():
                            cap_apply = gr.Button("Apply changes", variant="primary", icon=icon("play", "dark"))
                            cap_reset = gr.Button("Reset table", variant="secondary", icon=icon("refresh"))
                        cap_status = gr.HTML('<div class="status"><span class="dot"></span>Render a video, then '
                                             'edit its captions here.</div>')
            with gr.Tab("Review"):
                with gr.Row():
                    with gr.Column(scale=2):
                        script = gr.Textbox(label="Transcript", lines=16, max_lines=40, elem_id="script",
                                            info="Delete words or lines to cut them. Wrap a word in *stars* to "
                                                 "highlight it. Retyped words only change spelling.")
                    with gr.Column(scale=3):
                        title_box = gr.Textbox(label="Hook title", info="Shown during the first three seconds.")
                        plan_table = gr.Dataframe(
                            headers=PLAN_COLUMNS, interactive=True, wrap=True, label="Edit plan",
                            datatype=["number", "str", "str", "str", "str", "number", "str", "str", "str", "str", "str",
                                      "str", "str"],
                        )
                        with gr.Accordion("Allowed values", open=False):
                            gr.Markdown(
                                f"**keep** yes, CUT · **emotion** {', '.join(director.EMOTIONS)} · "
                                f"**beat** {', '.join(director.BEATS)} · **sfx** {', '.join(director.SFX)} · "
                                f"**camera** {', '.join(director.CAMERA)} · **transition** "
                                f"{', '.join(director.TRANSITIONS)} · **energy** 1–5 · **broll** words from a file "
                                "name in the B-roll folder", elem_classes="muted")
                with gr.Row(equal_height=True):
                    project_name = gr.Textbox(label="Project", placeholder="Untitled project", scale=3)
                    save_project_btn = gr.Button("Save", variant="secondary", scale=1, icon=icon("save"))
                    project_pick = gr.Dropdown(list_projects(), label="Open project", scale=3)
                    open_project_btn = gr.Button("Open", variant="secondary", scale=1, icon=icon("folder"))
                project_status = gr.Markdown("", elem_classes="muted")
            with gr.Tab("Style"):
                with gr.Row():
                    with gr.Column():
                        sec("Captions")
                        font_family = gr.Dropdown(list(FONT_FAMILIES), value=DEFAULT_FONT, label="Font")
                        style = gr.Dropdown(list(CAPTION_STYLES), value="Auto (by emotion)", label="Animation")
                        with gr.Row():
                            primary = gr.Dropdown(list(COLOR_CHOICES), value="White", label="Text")
                            highlight = gr.Dropdown(list(COLOR_CHOICES), value="Target Yellow", label="Highlight")
                        emotion_colors = gr.Checkbox(value=True, label="Match highlight color to emotion")
                        with gr.Row():
                            font_size = gr.Slider(60, 140, value=96, step=4, label="Size")
                            wpc = gr.Slider(1, 4, value=3, step=1, label="Words on screen")
                    with gr.Column():
                        sec("Brand")
                        with gr.Row():
                            logo = gr.Image(label="Logo", type="filepath", height=140)
                            with gr.Column():
                                handle = gr.Textbox(label="Handle", placeholder="@yourname")
                                cta_text = gr.Textbox(label="End card", placeholder="Follow for part 2")
                                end_card = gr.Checkbox(value=True, label="Show end card")
                        with gr.Row(equal_height=True):
                            preset_pick = gr.Dropdown(list_presets(), label="Preset", scale=3)
                            load_btn = gr.Button("Load", variant="secondary", scale=1)
                        with gr.Row(equal_height=True):
                            preset_name = gr.Textbox(label="Save current style as", placeholder="My brand", scale=3)
                            save_btn = gr.Button("Save", variant="secondary", scale=1, icon=icon("save"))
                        preset_status = gr.Markdown("", elem_classes="muted")
            with gr.Tab("Settings"):
                gr.Markdown("In Autopilot, anything you switch off here stays off. Everything else is chosen per "
                            "video. In Manual mode these settings are used as they are.", elem_classes="muted")
                with gr.Row():
                    with gr.Column():
                        sec("Understanding")
                        model = gr.Dropdown(["tiny", "base", "small", "medium"], value="small", label="Speech model")
                        language = gr.Dropdown(list(LANGUAGE_OPTIONS), value=list(LANGUAGE_OPTIONS)[0],
                                               label="Language and script")
                        llm_model = gr.Textbox(value=director.DEFAULT_LLM, label="Director model")
                        use_ai = gr.Checkbox(value=True, label="Story and emotion analysis")
                        use_vision = gr.Checkbox(value=True, label="Scene understanding")
                        spellfix = gr.Checkbox(value=True, label="Transcript correction")
                    with gr.Column():
                        sec("Editing")
                        cut = gr.Checkbox(value=True, label="Remove silences")
                        threshold = gr.Slider(0.3, 2.0, value=0.8, step=0.1, label="Silence threshold (s)")
                        fillers = gr.Checkbox(value=True, label="Remove filler sounds")
                        smart_cuts = gr.Checkbox(value=True, label="Remove retakes and mic checks")
                        hook_reorder = gr.Checkbox(value=True, label="Cold open")
                        reframe = gr.Dropdown(["Blurred background", "Center crop"], value="Blurred background",
                                              label="Landscape footage")
                    with gr.Column():
                        sec("Picture")
                        look = gr.Dropdown(finishing.available_looks(LUTS_DIR), value="Auto (by mood)", label="Look")
                        title_style = gr.Dropdown(list(TITLE_STYLES), value="Auto", label="Hook title")
                        grade = gr.Checkbox(value=True, label="Color correction")
                        stabilize = gr.Checkbox(value=False, label="Stabilize")
                        camera = gr.Checkbox(value=True, label="Camera moves")
                        face = gr.Checkbox(value=True, label="Face tracking")
                        multi_speaker = gr.Checkbox(value=True, label="Follow active speaker")
                        emoji = gr.Checkbox(value=True, label="Emoji")
                        hook = gr.Checkbox(value=True, label="Hook title")
                        transitions = gr.Checkbox(value=True, label="Transitions")
                        broll = gr.Checkbox(value=True, label="B-roll")
                        pbar = gr.Checkbox(value=True, label="Progress bar")
                    with gr.Column():
                        sec("Sound")
                        denoise = gr.Checkbox(value=True, label="Noise removal")
                        voice_polish = gr.Checkbox(value=True, label="Voice EQ and loudness")
                        sfx = gr.Checkbox(value=True, label="Sound effects")
                        auto_music = gr.Checkbox(value=True, label="Music from library")
                        music = gr.Audio(label="Music track", type="filepath", sources=["upload"])
                        music_vol = gr.Slider(0.05, 0.5, value=0.18, step=0.01, label="Music level")
                    with gr.Column():
                        sec("Export")
                        cover = gr.Checkbox(value=True, label="Cover image")
                        post_copy = gr.Checkbox(value=True, label="Post copy")
                        export_srt = gr.Checkbox(value=True, label="Subtitles (SRT, VTT)")
                        english_subs = gr.Checkbox(value=True, label="English subtitles")
                        export_xml = gr.Checkbox(value=True, label="Premiere / Resolve XML")
                        formats = gr.CheckboxGroup(["1:1 (feed)", "4:5 (portrait feed)", "16:9 (YouTube)"],
                                                   label="Additional formats")
            with gr.Tab("Clips"):
                gr.Markdown("Finds the strongest standalone 20–60 second moments in a long video, scores them and "
                            "edits each one into a reel with your current settings.", elem_classes="muted")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row(equal_height=True):
                            clips_link = gr.Textbox(placeholder="Paste a URL", show_label=False, scale=4)
                            clips_fetch = gr.Button("Import", variant="secondary", scale=1, icon=icon("link"))
                        clips_import_status = gr.HTML("")
                        long_video = gr.Video(label="Long video", sources=["upload"], height=300)
                        n_clips = gr.Slider(1, 8, value=3, step=1, label="Number of clips")
                        clips_btn = gr.Button("Find clips", variant="primary", icon=icon("search", "dark"))
                    with gr.Column(scale=3):
                        clips_table = gr.Dataframe(headers=["#", "Score", "Title", "Length", "Why", "File"],
                                                   interactive=False, wrap=True, label="Results")
                        clips_log = gr.Textbox(label="Progress", lines=8, max_lines=20, interactive=False,
                                               elem_id="logs")
            with gr.Tab("Batch"):
                gr.Markdown("Edits a list of videos one after another with Autopilot. Finished files are remembered, "
                            "so a restarted batch picks up where it stopped. Watch mode keeps checking the folder "
                            "until you stop it.", elem_classes="muted")
                with gr.Row():
                    with gr.Column(scale=2):
                        batch_files = gr.File(label="Videos", file_count="multiple", file_types=["video"], height=160)
                        batch_folder = gr.Textbox(label="Or a folder", placeholder=r"C:\Users\me\Videos\raw")
                        batch_watch = gr.Checkbox(value=False, label="Watch folder for new files")
                        with gr.Row():
                            batch_btn = gr.Button("Start", variant="primary", icon=icon("play", "dark"))
                            batch_stop = gr.Button("Stop", variant="secondary", icon=icon("stop"))
                    with gr.Column(scale=3):
                        batch_table = gr.Dataframe(headers=["Video", "Status", "Output", "Time"], interactive=False,
                                                   wrap=True, label="Queue")
            with gr.Tab("Library"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row(equal_height=True):
                            lib_pick = gr.Dropdown(list_outputs(), label="Renders", scale=4)
                            lib_refresh = gr.Button("", variant="secondary", scale=0, min_width=44, icon=icon("refresh"))
                        gr.Markdown(f"Renders · `{OUTPUT_DIR}`  \nB-roll · `{BROLL_DIR}`  \nMusic · `{MUSIC_DIR}`  \n"
                                    f"Looks · `{LUTS_DIR}`  \nProjects · `{PROJECTS_DIR}`", elem_classes="muted")
                    with gr.Column(scale=3):
                        lib_video = gr.Video(show_label=False, interactive=False, height=420)

        with gr.Accordion("Activity log", open=False):
            logs = gr.Textbox(show_label=False, lines=14, max_lines=40, elem_id="logs", interactive=False)

        # ---------------- wiring
        opts = [cut, threshold, primary, highlight, model, language, reframe, wpc, font_size, denoise, grade,
                camera, pbar, use_ai, llm_model, style, emotion_colors, face, emoji, sfx, hook, transitions,
                music, music_vol, font_family, spellfix, fillers, smart_cuts, hook_reorder, broll, voice_polish,
                auto_music, logo, handle, cta_text, end_card, quality, mode, use_vision, look, stabilize,
                title_style, multi_speaker, export_srt, english_subs, export_xml, cover, post_copy, formats]
        assert len(opts) == len(OPTION_FIELDS)
        outs = [result, out_files, logs, status, analysis, qa, script, plan_table, title_box, job_state, post,
                cap_state, cap_table, cap_player, cap_style, cap_size, cap_pos]
        assert len(outs) == len(OUT_NAMES)

        go.click(auto_ui, inputs=[original, *opts], outputs=outs)
        transcribe_btn.click(transcribe_ui, inputs=[original, *opts], outputs=outs)
        direct_btn.click(direct_ui, inputs=[job_state, script, *opts], outputs=outs)
        render_btn.click(render_ui, inputs=[job_state, script, plan_table, title_box, *opts], outputs=outs)
        fetch_btn.click(fetch_link_ui, inputs=[link, cookies], outputs=[original, logs, import_status])
        link.submit(fetch_link_ui, inputs=[link, cookies], outputs=[original, logs, import_status])

        preset_controls = [primary, highlight, style, font_family, emotion_colors, font_size, wpc, music_vol, logo,
                           handle, cta_text, end_card]
        assert len(preset_controls) == len(PRESET_FIELDS)
        save_btn.click(save_preset_ui, inputs=[preset_name, *preset_controls], outputs=[preset_pick, preset_status])
        load_btn.click(load_preset_ui, inputs=[preset_pick], outputs=[*preset_controls, preset_status])

        save_project_btn.click(save_project_ui, inputs=[project_name, job_state, script, plan_table, title_box, *opts],
                               outputs=[project_pick, project_status])
        open_project_btn.click(open_project_ui, inputs=[project_pick],
                               outputs=[job_state, script, plan_table, title_box, analysis, original, *opts,
                                        project_status])
        clips_btn.click(clips_ui, inputs=[long_video, n_clips, *opts], outputs=[clips_table, clips_log, lib_pick])
        # Same downloader as the source panel (uses its cookies file for private posts).
        clips_fetch.click(fetch_link_ui, inputs=[clips_link, cookies],
                          outputs=[long_video, clips_log, clips_import_status])
        clips_link.submit(fetch_link_ui, inputs=[clips_link, cookies],
                          outputs=[long_video, clips_log, clips_import_status])
        batch_ev = batch_btn.click(batch_ui, inputs=[batch_files, batch_folder, batch_watch, *opts],
                                   outputs=[batch_table, lib_pick])
        batch_stop.click(None, cancels=[batch_ev])

        # Caption editor
        # "Edit captions" opens the Captions tab via a native listener installed in FORCE_DARK (page JS).
        cap_table.select(caption_seek, inputs=cap_table, outputs=cap_seek_t)
        cap_seek_t.change(None, inputs=cap_seek_t, js="""(t) => {
            const v = document.querySelector('#cap_player video');
            if (v && t !== null && t !== undefined) { v.currentTime = t; v.pause(); }
            return t; }""")
        cap_apply.click(caption_apply_ui, inputs=[cap_state, cap_table, cap_style, cap_size, cap_pos],
                        outputs=[cap_player, result, out_files, cap_state, cap_status])
        cap_reset.click(caption_reset_ui, inputs=cap_state, outputs=cap_table)

        lib_refresh.click(lambda: gr.update(choices=list_outputs()), outputs=lib_pick)
        lib_pick.change(lambda n: str(OUTPUT_DIR / n) if n else None, inputs=lib_pick, outputs=lib_video)
        result.change(lambda: gr.update(choices=list_outputs()), outputs=lib_pick)
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                                 allowed_paths=[str(APP_DIR / "assets")],
                                 favicon_path=str(APP_DIR / "assets" / "favicon.png"))
