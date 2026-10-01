"""
AI Director: understands the story & emotion of a transcript and produces an edit plan.

Two layers of understanding:
  1. Prosody (how it was said): loudness, pitch movement and speaking rate per sentence, from the audio.
  2. Meaning (what was said): a local LLM via Ollama reads the whole transcript, summarizes the story,
     then directs every sentence (emotion, energy, story beat, emphasis words, emoji, SFX, camera, transition).

If Ollama is not running, a rule-based director uses prosody + keywords so the app still works.
Words are duck-typed: any object with .text, .start, .end, .highlight works.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LLM = "qwen2.5:7b"

EMOTIONS = ["excited", "happy", "funny", "curious", "surprised", "serious", "emotional", "angry", "calm", "confident"]
BEATS = ["hook", "setup", "context", "tension", "reveal", "punchline", "lesson", "cta"]
SFX = ["none", "pop", "whoosh", "ding", "boom", "riser", "click", "swoosh"]
CAMERA = ["none", "punch", "slow_push", "shake", "pull_back"]
CONTENT_TYPES = ["tips_tutorial", "storytime", "comedy", "motivational", "podcast_interview", "vlog",
                 "product_review", "opinion_rant", "educational", "other"]
PACES = ["calm", "medium", "fast"]
TRANSITIONS = ["none", "flash", "zoom", "whip"]

# Caption accent color per emotion (used when "emotion colors" is on).
EMOTION_COLORS = {
    "excited": "#FFE600", "happy": "#FFD23F", "funny": "#39FF14", "curious": "#00E5FF",
    "surprised": "#FF8A00", "serious": "#FF3B3B", "emotional": "#7FB2FF", "angry": "#FF2E2E",
    "calm": "#A8FFDF", "confident": "#FFE600",
}

KEYWORD_EMOJI = {
    "money": "💰", "paisa": "💰", "paise": "💰", "rupee": "💰", "rich": "🤑", "profit": "📈", "growth": "📈",
    "fire": "🔥", "crazy": "🤯", "insane": "🤯", "mind": "🤯", "secret": "🤫", "love": "❤️", "pyaar": "❤️",
    "heart": "❤️", "time": "⏰", "time's": "⏰", "fast": "⚡", "quick": "⚡", "idea": "💡", "tip": "💡",
    "think": "🤔", "why": "🤔", "kyun": "🤔", "kyu": "🤔", "sad": "😢", "cry": "😢", "laugh": "😂", "funny": "😂",
    "win": "🏆", "best": "🏆", "goal": "🎯", "target": "🎯", "stop": "🛑", "no": "❌", "wrong": "❌",
    "yes": "✅", "right": "✅", "correct": "✅", "phone": "📱", "work": "💼", "gym": "💪", "strong": "💪",
    "food": "🍔", "khana": "🍛", "friend": "🤝", "dost": "🤝", "family": "👨‍👩‍👧", "world": "🌍", "eyes": "👀",
    "look": "👀", "dekho": "👀", "joke": "😂", "weird": "🤨", "testing": "🧪", "test": "🧪", "music": "🎵",
    "fear": "😱", "dar": "😱", "shock": "😱", "happy": "😄", "khush": "😄", "angry": "😤", "gussa": "😤",
}

MIC_CHECK_WORDS = {"testing", "test", "mic", "check", "hello", "one", "two", "three", "ek", "do", "teen",
                   "um", "uh", "umm", "hmm", "okay", "ok", "so", "recording", "chalu"}

EXCITED_WORDS = {"wow", "amazing", "insane", "crazy", "best", "awesome", "zabardast", "mast", "kamaal", "jordaar",
                 "omg", "boom", "unbelievable", "huge", "massive"}
SERIOUS_WORDS = {"never", "mistake", "important", "warning", "problem", "truth", "careful", "danger", "must",
                 "galti", "dhyan", "sach"}
EMOTIONAL_WORDS = {"sad", "miss", "lost", "cry", "dukh", "pain", "hurt", "alone", "sorry", "yaad"}


@dataclass
class LinePlan:
    i: int
    start: float
    end: float
    text: str
    emotion: str = "calm"
    energy: int = 3
    beat: str = "context"
    emphasis: list[str] = field(default_factory=list)
    emoji: str = ""
    sfx: str = "none"
    camera: str = "none"
    transition: str = "none"
    keep: bool = True       # False = retake / false start / off-topic -> cut from the reel
    broll: str = ""         # short visual idea for a cutaway (matched against the B-roll folder)
    loudness: float = 0.0   # z-score vs. the rest of the video
    pitch_var: float = 0.0  # pitch movement, z-score vs. the rest of the video
    rate: float = 0.0       # words / second


@dataclass
class EditPlan:
    title_hook: str
    mood: str
    summary: str
    lines: list[LinePlan]
    source: str  # "ai:<model>" or "rules"
    hook_line: int = 0  # line number to move to the very start as a cold-open hook (0 = keep order)
    content_type: str = "other"
    pace: str = "medium"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------------------

def sentence_groups(words: Sequence, max_words: int = 10, pause: float = 0.55) -> list[list]:
    groups: list[list] = []
    cur: list = []
    for w in words:
        if cur and (w.start - cur[-1].end > pause or len(cur) >= max_words):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w.text.rstrip().endswith((".", "?", "!", "।")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


# --------------------------------------------------------------------------------------
# Prosody analysis (how it was said)
# --------------------------------------------------------------------------------------

def _frame_pitch(frame: np.ndarray, sr: int) -> float:
    """Autocorrelation pitch estimate in Hz (0 if unvoiced)."""
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 0.01:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
    lo, hi = int(sr / 400), int(sr / 75)
    if hi >= corr.size:
        return 0.0
    lag = lo + int(np.argmax(corr[lo:hi]))
    if corr[lag] < 0.3 * corr[0]:
        return 0.0
    return sr / lag


def prosody_features(audio_path: Path | None, groups: list[list]) -> list[dict]:
    feats = [{"loudness": 0.0, "pitch_var": 0.0, "rate": 0.0} for _ in groups]
    for f, g in zip(feats, groups):
        dur = max(0.2, g[-1].end - g[0].start)
        f["rate"] = len(g) / dur
    if not audio_path or not Path(audio_path).exists():
        return feats
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr > 16000:  # decimate for speed; speech pitch is well below 8 kHz
        step = sr // 16000
        audio, sr = audio[::step], sr // step

    loud = []
    for f, g in zip(feats, groups):
        seg = audio[int(g[0].start * sr): int(g[-1].end * sr)]
        if seg.size < sr * 0.1:
            loud.append(-60.0)
            continue
        loud.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
        n = int(0.04 * sr)
        pitches = [_frame_pitch(seg[i:i + n], sr) for i in range(0, seg.size - n, n)]
        voiced = np.array([p for p in pitches if p > 0])
        if voiced.size > 3:
            semis = 12 * np.log2(voiced / np.median(voiced))
            f["pitch_var"] = float(np.std(semis))
    # Relative to this speaker: z-scores make "loud"/"expressive" mean louder/livelier than their own baseline.
    loud_arr = np.array(loud)
    pitch_arr = np.array([f["pitch_var"] for f in feats])
    for f, l, p in zip(feats, loud_arr, pitch_arr):
        f["loudness"] = float((l - loud_arr.mean()) / (loud_arr.std() or 1.0))
        f["pitch_var"] = float((p - pitch_arr.mean()) / (pitch_arr.std() or 1.0))
    return feats


def prosody_label(f: dict) -> str:
    loud = "loud" if f["loudness"] > 0.7 else "quiet" if f["loudness"] < -0.7 else "normal volume"
    pitch = "very expressive" if f["pitch_var"] > 0.8 else "flat tone" if f["pitch_var"] < -0.8 else "normal tone"
    rate = "fast" if f["rate"] > 3.5 else "slow" if f["rate"] < 1.8 else "normal pace"
    return f"{loud}, {pitch}, {rate}"


# --------------------------------------------------------------------------------------
# LLM director (what was said)
# --------------------------------------------------------------------------------------

def ollama_available(model: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        names = [m["name"] for m in tags.get("models", [])]
        if not any(n == model or n.split(":")[0] == model for n in names):
            return False, f"Ollama is running but model '{model}' is not pulled (run: ollama pull {model})."
        return True, ""
    except (urllib.error.URLError, OSError, ValueError):
        return False, "Ollama is not running (install from ollama.com and start it)."


def ollama_json(model: str, system: str, prompt: str, schema: dict, timeout: int = 1800,
                max_tokens: int = 2500) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        # num_predict caps runaway generations (small models occasionally loop inside a JSON string).
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": max_tokens, "repeat_penalty": 1.15},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    return json.loads(content)


STORY_SYSTEM = """You are the creative director of a top Instagram Reels / YouTube Shorts editing studio.
The speaker may talk in English, Hindi, Hinglish (Hindi+English) or Gujlish (Gujarati+English), usually
written in Roman script. The transcript comes from speech recognition and may contain misspelled words:
infer the intended meaning from context. Understand the story, the speaker's intent and emotional arc."""

STORY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "mood": {"type": "string", "enum": EMOTIONS},
        "title_hook": {"type": "string"},
        "hook_line": {"type": "integer"},
        "content_type": {"type": "string", "enum": CONTENT_TYPES},
        "pace": {"type": "string", "enum": PACES},
    },
    "required": ["summary", "mood", "title_hook", "hook_line", "content_type", "pace"],
}

LINES_SYSTEM = STORY_SYSTEM + """

You now direct the edit sentence by sentence. For each numbered line decide:
- emotion: the feeling the viewer should get from this line.
- energy: 1 (quiet, reflective) .. 5 (peak moment). Use the prosody hints (how it was said) AND the meaning.
- beat: its role in the story arc (hook = first grabbing line, reveal = key information or twist,
  punchline = funny/payoff moment, lesson = takeaway, cta = asking to follow/comment/share).
- emphasis: 0-2 words copied EXACTLY from that line that carry the meaning (they get highlighted).
- emoji: ONE emoji that fits the meaning, or "" (use on roughly one line in three, never on consecutive lines).
- sfx: sound effect at the start of the line. pop = playful/emoji, whoosh/swoosh = topic change,
  ding = insight/lesson, boom = big reveal or shock, riser = building suspense before a reveal,
  click = list items. Use "none" on most calm lines. Never the same sfx on consecutive lines.
- camera: punch = quick zoom on an emphasized word, slow_push = slow zoom-in for serious/emotional lines,
  shake = only for shock/anger/peak excitement, pull_back = relief or zooming out to the big picture.
- keep: false ONLY for lines a professional editor would cut: a retake (the speaker repeats the same thing
  again right after - keep the LAST, cleanest version), a false start, mic checks / counting ("1, 2, 3",
  "testing", "is it recording"), or pure filler with no meaning. Everything with meaning stays true.
- broll: a 1-3 word concrete visual that could illustrate the line as a cutaway (e.g. "money", "gym",
  "city night", "coffee"), or "" when the speaker's face matters more. Use on about one line in four.
- transition (into this line): flash for big reveals/punchlines, zoom for topic changes, whip for energetic
  switches, none otherwise. Use transitions sparingly (at most one in four lines).
Pacing targets for a retention-optimized reel (a boring edit loses viewers in 2 seconds):
- camera move on about 60% of lines, emoji on about 35%, sfx on about 40%, transitions on about 20%.
- The emotion must follow the content line by line. A monologue shifts between curious, funny, serious,
  confident etc. Do not label every line with the same emotion.
- Only line 1 can be the hook. Energy must vary: the peak moment of the story gets energy 5."""


def _lines_schema(n: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "emotion": {"type": "string", "enum": EMOTIONS},
            "energy": {"type": "integer", "minimum": 1, "maximum": 5},
            "beat": {"type": "string", "enum": BEATS},
            "emphasis": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
            "emoji": {"type": "string"},
            "sfx": {"type": "string", "enum": SFX},
            "camera": {"type": "string", "enum": CAMERA},
            "transition": {"type": "string", "enum": TRANSITIONS},
            "keep": {"type": "boolean"},
            "broll": {"type": "string"},
        },
        "required": ["i", "emotion", "energy", "beat", "emphasis", "emoji", "sfx", "camera", "transition",
                     "keep", "broll"],
    }
    return {"type": "object", "properties": {"lines": {"type": "array", "items": item, "minItems": n, "maxItems": n}},
            "required": ["lines"]}


def ai_direct(lines: list[LinePlan], model: str, log: Callable[[str], None], batch: int = 12,
              visual: str = "") -> tuple:
    transcript = "\n".join(f"{l.i}. [{l.start:.1f}s] {l.text}" for l in lines)
    seen = f"What the footage looks like: {visual}\n\n" if visual else ""
    log(f"AI Director ({model}): reading the whole story...")
    story = ollama_json(model, STORY_SYSTEM,
                        seen + "Transcript:\n" + transcript + "\n\nReturn: content_type (the format of this video), "
                        "pace (how fast the edit should feel for this content), summary (2-3 sentences in English describing "
                        "what happens and the emotional arc), mood (overall), title_hook (a punchy 3-6 word on-screen "
                        "hook title in the same language style the speaker uses, Roman script, no hashtags), hook_line (the number of "
                        "the single most attention-grabbing line to move to the very start as a cold open; 0 if line 1 "
                        "is already the best hook or if moving a line would confuse the story).",
                        STORY_SCHEMA)
    summary = str(story.get("summary", "")).strip()
    mood = story.get("mood", "calm") if story.get("mood") in EMOTIONS else "calm"
    title = str(story.get("title_hook", "")).strip().strip('"')[:60]
    try:
        hook_line = int(story.get("hook_line", 0))
    except (TypeError, ValueError):
        hook_line = 0
    hook_line = hook_line if 1 < hook_line <= len(lines) else 0
    content_type = story.get("content_type") if story.get("content_type") in CONTENT_TYPES else "other"
    pace = story.get("pace") if story.get("pace") in PACES else "medium"
    log(f"Content type: {content_type} · pace: {pace}")
    log(f"Story: {summary}")
    log(f"Overall mood: {mood} | Hook title: {title}")

    for b in range(0, len(lines), batch):
        chunk = lines[b:b + batch]
        log(f"AI Director: directing lines {chunk[0].i}-{chunk[-1].i} of {len(lines)}...")
        numbered = "\n".join(f"{l.i}. \"{l.text}\"  (prosody: {prosody_label(asdict(l))})" for l in chunk)
        prompt = (f"{seen}Content type: {content_type}, pace: {pace}\nStory summary: {summary}\n"
                  f"Overall mood: {mood}\n\nFull transcript for context:\n{transcript}\n\n"
                  f"Direct ONLY these lines (return exactly {len(chunk)} items with the same i):\n{numbered}")
        try:
            out = ollama_json(model, LINES_SYSTEM, prompt, _lines_schema(len(chunk)))
        except Exception as exc:
            log(f"AI Director batch failed ({exc}); using rules for these lines.")
            for l in chunk:
                rule_line(l, first=(l.i == 1))
            continue
        by_i = {int(d.get("i", -1)): d for d in out.get("lines", []) if isinstance(d, dict)}
        for pos, l in enumerate(chunk):
            d = by_i.get(l.i) or (out["lines"][pos] if pos < len(out.get("lines", [])) else None)
            if d is None:
                rule_line(l, first=(l.i == 1))
            else:
                apply_directions(l, d)
    return title, mood, summary, hook_line, content_type, pace


def apply_directions(line: LinePlan, d: dict) -> None:
    line.emotion = d.get("emotion") if d.get("emotion") in EMOTIONS else line.emotion
    try:
        line.energy = int(np.clip(int(d.get("energy", line.energy)), 1, 5))
    except (TypeError, ValueError):
        pass
    line.beat = d.get("beat") if d.get("beat") in BEATS else line.beat
    line_tokens = {_norm(t): t for t in line.text.split()}
    line.emphasis = [line_tokens[_norm(e)] for e in d.get("emphasis", []) if _norm(e) in line_tokens][:2]
    emoji = str(d.get("emoji", "")).strip()
    line.emoji = emoji if emoji and len(emoji) <= 8 and not emoji.isascii() else ""
    line.sfx = d.get("sfx") if d.get("sfx") in SFX else "none"
    line.camera = d.get("camera") if d.get("camera") in CAMERA else "none"
    line.transition = d.get("transition") if d.get("transition") in TRANSITIONS else "none"
    if "keep" in d:
        keep = d["keep"]
        line.keep = keep if isinstance(keep, bool) else str(keep).strip().lower() not in ("false", "no", "0", "cut")
    if "broll" in d:
        line.broll = str(d.get("broll") or "").strip()[:40]


FIX_SYSTEM = """You correct speech-recognition transcripts of Indian creators who speak English, Hindi,
Hinglish or Gujlish. Fix misheard or misspelled words using the context of the whole transcript so each line
says what the speaker actually meant. Rules: keep the SAME language mix and Roman script (never translate,
never convert to Devanagari/Gujarati), use common creator spellings (e.g. "hai", "nahi", "kya", "matlab",
"che", "nathi", "saru"), keep roughly the same number of words, keep numbers as digits, do not add content,
do not summarize, do not add emoji or hashtags.
Typical recognition errors to repair: dropped or softened first consonants ("olne" -> "bolne", "vi"/"bi" ->
"bhi", "aat" -> "baat"), words glued or split wrongly ("kuchvi" -> "kuch bhi"), Hindi words spelled as English
look-alikes. Never replace a Hindi/Gujarati word with an unrelated English word. If you are not sure what a
word was, leave it exactly as it is."""


def correct_transcript(sentences: list[str], model: str, log: Callable[[str], None], batch: int = 15) -> list[str]:
    """Context-aware spelling correction of each sentence. Returns the same number of sentences."""
    full = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(sentences))
    out = list(sentences)
    schema = {"type": "object", "properties": {"lines": {"type": "array", "items": {
        "type": "object", "properties": {"i": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["i", "text"]}}}, "required": ["lines"]}
    for b in range(0, len(sentences), batch):
        idx = range(b, min(len(sentences), b + batch))
        log(f"AI spell-fix: lines {idx[0] + 1}-{idx[-1] + 1} of {len(sentences)}...")
        prompt = (f"Whole transcript for context:\n{full}\n\nReturn corrected text for lines "
                  f"{idx[0] + 1}-{idx[-1] + 1} only, same numbering.")
        try:
            res = ollama_json(model, FIX_SYSTEM, prompt, schema)
        except Exception as exc:
            log(f"AI spell-fix batch failed ({exc}); keeping original text.")
            continue
        for item in res.get("lines", []):
            try:
                k = int(item["i"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            new = str(item.get("text", "")).strip()
            if not re.match(r"^\d+\.\s", sentences[k] if 0 <= k < len(sentences) else ""):
                new = re.sub(r"^\d+\.\s*", "", new)  # the model sometimes echoes the line number
            if k in idx and new:
                n_old, n_new = len(sentences[k].split()), len(new.split())
                # Reject rewrites that change length a lot (the model summarizing or inventing).
                nxt = sentences[k + 1].lower().split() if k + 1 < len(sentences) else []
                borrowed = bool(nxt) and new.lower().split()[-2:] == nxt[:2]
                if 0.75 * n_old <= n_new <= 1.25 * n_old + 1 and not borrowed:
                    out[k] = new
    return out


# --------------------------------------------------------------------------------------
# Rule-based director (fallback)
# --------------------------------------------------------------------------------------

def _norm(t: str) -> str:
    return re.sub(r"[^\wऀ-ॿ઀-૿]", "", t.lower())


def rule_line(line: LinePlan, first: bool = False) -> None:
    toks = [_norm(t) for t in line.text.split()]
    text = line.text.strip()
    if any(t in EXCITED_WORDS for t in toks) or text.endswith("!"):
        line.emotion = "excited"
    elif any(t in SERIOUS_WORDS for t in toks):
        line.emotion = "serious"
    elif any(t in EMOTIONAL_WORDS for t in toks):
        line.emotion = "emotional"
    elif text.endswith("?"):
        line.emotion = "curious"
    elif line.loudness > 0.8 and line.pitch_var > 0.8:
        line.emotion = "excited"
    else:
        line.emotion = "confident" if line.loudness > 0 else "calm"

    energy = 3 + (line.loudness > 0.7) + (line.pitch_var > 0.8) + (line.rate > 3.5) - (line.loudness < -0.7)
    line.energy = int(np.clip(energy, 1, 5))
    line.beat = "hook" if first else ("cta" if any(t in {"follow", "subscribe", "comment", "share", "like"} for t in toks)
                                      else "context")
    line.emoji = next((KEYWORD_EMOJI[t] for t in toks if t in KEYWORD_EMOJI), "")
    long_words = sorted((t for t in line.text.split() if len(_norm(t)) >= 6), key=lambda t: -len(t))
    line.emphasis = long_words[:1]
    line.camera = ("slow_push" if first or line.emotion in ("serious", "emotional")
                   else "shake" if line.energy >= 5 else "punch" if line.energy >= 4 or line.emphasis else "none")
    line.sfx = "pop" if line.emoji else ("whoosh" if first else "none")
    meaningful = [t for t in toks if t and not t.isdigit() and t not in MIC_CHECK_WORDS]
    line.keep = len(meaningful) > 0


def enforce_density(lines: list[LinePlan], rule_hints: dict[int, LinePlan]) -> None:
    """Small local LLMs tend to under-edit; top up camera moves and emoji so the edit never feels static."""
    for k, l in enumerate(lines):
        hint = rule_hints.get(l.i)
        if l.beat == "hook" and l.i != 1:
            l.beat = "context"
        if l.camera == "none" and (k % 2 == 1 or l.energy >= 4 or l.i == 1):
            l.camera = ("slow_push" if l.i == 1 or l.emotion in ("serious", "emotional", "calm")
                        else "shake" if l.energy >= 5 and l.emotion in ("surprised", "angry", "excited")
                        else "punch")
        if not l.emoji and hint and hint.emoji and not any(x.emoji for x in lines[max(0, k - 2):k]):
            l.emoji = hint.emoji
            if l.sfx == "none":
                l.sfx = "pop"


def enforce_keep_safety(lines: list[LinePlan], hook_line: int) -> None:
    """Never let the director cut the story away: at most 35% of lines, never the hook line."""
    if hook_line:
        lines[hook_line - 1].keep = True
    cut = [l for l in lines if not l.keep]
    if len(cut) > max(1, int(len(lines) * 0.35)):
        for l in cut:
            l.keep = True


def enforce_restraint(lines: list[LinePlan]) -> None:
    """Avoid repetitive or unearned effects, whatever the director said."""
    for l in lines:
        if l.camera == "shake" and l.energy < 4:
            l.camera = "punch"  # shakes are reserved for genuine peaks
    for prev, cur in zip(lines, lines[1:]):
        if cur.sfx != "none" and cur.sfx == prev.sfx:
            cur.sfx = "none"
        if cur.emoji and prev.emoji:
            cur.emoji = ""
            if cur.sfx == "pop":
                cur.sfx = "none"
        if cur.transition != "none" and prev.transition != "none":
            cur.transition = "none"
        if cur.camera == "shake" and prev.camera == "shake":
            cur.camera = "punch"


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def guess_content_type(lines: list[LinePlan]) -> tuple[str, str]:
    """Rule-based content type & pace when no LLM is available."""
    text = " ".join(l.text.lower() for l in lines)
    rate = float(np.mean([l.rate for l in lines])) if lines else 2.5
    pace = "fast" if rate > 3.2 else "calm" if rate < 2.0 else "medium"
    if any(k in text for k in ("step", "how to", "tip", "trick", "kaise", "tarika")):
        return "tips_tutorial", pace
    if any(k in text for k in ("one day", "ek din", "story", "kahani", "happened", "hua")):
        return "storytime", pace
    if any(k in text for k in ("review", "unboxing", "price", "worth")):
        return "product_review", pace
    if any(k in text for k in ("dream", "success", "never give up", "mehnat", "hard work")):
        return "motivational", pace
    return "other", pace


def build_plan(words: Sequence, audio_path: Path | None, use_ai: bool, model: str,
               log: Callable[[str], None], visual: str = "") -> EditPlan:
    groups = sentence_groups(words)
    feats = prosody_features(audio_path, groups)
    lines = [
        LinePlan(i=k + 1, start=g[0].start, end=g[-1].end, text=" ".join(w.text.strip() for w in g),
                 loudness=f["loudness"], pitch_var=f["pitch_var"], rate=f["rate"])
        for k, (g, f) in enumerate(zip(groups, feats))
    ]
    if not lines:
        return EditPlan("", "calm", "", [], "rules")

    for l in lines:  # rules first so every field has a sane default
        rule_line(l, first=(l.i == 1))
    rule_hints = {l.i: LinePlan(**asdict(l)) for l in lines}

    source = "rules"
    title, mood, summary, hook_line = "", "calm", "", 0
    content_type, pace = guess_content_type(lines)
    if use_ai:
        ok, why = ollama_available(model)
        if ok:
            try:
                title, mood, summary, hook_line, content_type, pace = ai_direct(lines, model, log, visual=visual)
                source = f"ai:{model}"
            except Exception as exc:
                log(f"AI Director failed ({exc}); falling back to rule-based director.")
        else:
            log(f"AI Director unavailable: {why} Using rule-based director.")

    if source == "rules":
        emotions = [l.emotion for l in lines]
        mood = max(set(emotions), key=emotions.count)
        first = next((l for l in lines if l.keep), lines[0])
        clean = [t for t in first.text.split() if _norm(t) and not _norm(t).isdigit()
                 and _norm(t) not in MIC_CHECK_WORDS]
        title = " ".join(clean[:5]).rstrip(",.")
        log("Rule-based director: emotions from voice energy, pitch and keywords.")

    enforce_density(lines, rule_hints)
    enforce_restraint(lines)
    enforce_keep_safety(lines, hook_line)
    if hook_line and not (1.2 <= lines[hook_line - 1].end - lines[hook_line - 1].start <= 7.0):
        hook_line = 0  # a cold open must be a short, self-contained line
    return EditPlan(title_hook=title, mood=mood, summary=summary, lines=lines, source=source, hook_line=hook_line,
                    content_type=content_type, pace=pace)


def apply_emphasis(words: Sequence, plan: EditPlan) -> None:
    """Mark director-chosen emphasis words as highlighted on the word timeline."""
    for line in plan.lines:
        targets = {_norm(e) for e in line.emphasis}
        if not targets:
            continue
        for w in words:
            if line.start - 0.01 <= w.start <= line.end + 0.01 and _norm(w.text) in targets:
                w.highlight = True


# --------------------------------------------------------------------------------------
# Long video -> viral clips (Opus Clip style)
# --------------------------------------------------------------------------------------

CLIP_SYSTEM = """You are a viral short-form video producer. From a long video transcript (podcast, vlog, talk,
stream) you pick moments that work as STANDALONE Reels/Shorts. A great clip:
- starts with a hook line that grabs attention without needing earlier context,
- contains one complete idea, story, insight, joke or hot take with a payoff,
- is 20-60 seconds long,
- does not end mid-thought.
Score virality 0-100 honestly (most moments are 40-70; 85+ only for exceptional ones)."""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "maxItems": 4, "items": {
        "type": "object",
        "properties": {
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
            "title": {"type": "string"}, "hook": {"type": "string"},
            "score": {"type": "integer", "minimum": 0, "maximum": 100}, "reason": {"type": "string"},
        },
        "required": ["start_line", "end_line", "title", "hook", "score", "reason"]}}},
    "required": ["clips"],
}


@dataclass
class ClipIdea:
    start: float
    end: float
    title: str
    hook: str
    score: int
    reason: str


def find_viral_moments(words: Sequence, model: str, log: Callable[[str], None], min_len: float = 18.0,
                       max_len: float = 65.0, window: int = 140, overlap: int = 20) -> list[ClipIdea]:
    groups = sentence_groups(words, max_words=18, pause=0.8)
    lines = [(g[0].start, g[-1].end, " ".join(w.text.strip() for w in g)) for g in groups]
    ideas: list[ClipIdea] = []
    for b in range(0, len(lines), window - overlap):
        chunk = lines[b:b + window]
        if not chunk:
            break
        log(f"Scanning for viral moments: {chunk[0][0] / 60:.1f}-{chunk[-1][1] / 60:.1f} min...")
        numbered = "\n".join(f"{b + k + 1}. [{s:.0f}s] {t}" for k, (s, _, t) in enumerate(chunk))
        try:
            res = ollama_json(model, CLIP_SYSTEM, "Transcript lines:\n" + numbered +
                              "\n\nReturn the best 1-4 clips (line numbers inclusive). Title: 3-7 words, Roman "
                              "script, same language style as the speaker.", CLIP_SCHEMA)
        except Exception as exc:
            log(f"Window failed ({exc}); skipping.")
            continue
        for c in res.get("clips", []):
            try:
                s_i, e_i = int(c["start_line"]) - 1, int(c["end_line"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= s_i <= e_i < len(lines)):
                continue
            start, end = lines[s_i][0], lines[e_i][1]
            # Extend short picks to the minimum length with following sentences.
            while end - start < min_len and e_i + 1 < len(lines):
                e_i += 1
                end = lines[e_i][1]
            if not (min_len * 0.8 <= end - start <= max_len):
                continue
            ideas.append(ClipIdea(max(0.0, start - 0.15), end + 0.3, str(c.get("title", "")).strip()[:60],
                                  str(c.get("hook", "")).strip()[:120], int(np.clip(int(c.get("score", 50)), 0, 100)),
                                  str(c.get("reason", "")).strip()[:200]))
        if b + window >= len(lines):
            break
    # Best first, no overlaps.
    picked: list[ClipIdea] = []
    for idea in sorted(ideas, key=lambda x: -x.score):
        if all(idea.end <= p.start or idea.start >= p.end for p in picked):
            picked.append(idea)
    return picked
