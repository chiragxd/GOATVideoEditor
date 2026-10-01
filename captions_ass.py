"""
Caption engine on libass (ASS subtitles burned by FFmpeg while encoding).

Why: libass renders in native code during the encode (no per-word Python image compositing), shapes
Devanagari/Gujarati correctly via HarfBuzz, and supports real animation tags (scale, blur, move, karaoke).

Styles: pop, bounce, box (karaoke box), sweep (karaoke fill), typewriter, glow, slide, oneword, neon,
clean, shake. "auto" picks per sentence from the emotion the director detected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

CAPTION_STYLES = {
    "Auto (by emotion)": "auto",
    "Bold pop": "pop",
    "Bounce": "bounce",
    "Karaoke box": "box",
    "Highlight sweep": "sweep",
    "Typewriter": "typewriter",
    "Glow": "glow",
    "Slide up": "slide",
    "One word at a time": "oneword",
    "Neon": "neon",
    "Clean minimal": "clean",
    "Shake on emphasis": "shake",
}

EMOTION_STYLE = {
    "excited": "bounce", "happy": "box", "funny": "box", "confident": "pop", "curious": "typewriter",
    "surprised": "shake", "serious": "clean", "emotional": "slide", "angry": "shake", "calm": "sweep",
}

# libass family names of the fonts in fonts/ (static instances are generated for variable fonts).
FAMILY_NAMES = {
    "Montserrat ExtraBold": "Montserrat ExtraBold",
    "Montserrat Black": "Montserrat Black",
    "Poppins Black": "Poppins Black",
    "Poppins ExtraBold": "Poppins ExtraBold",
    "Anton": "Anton",
    "Bebas Neue": "Bebas Neue",
    "Impact (Windows)": "Impact",
}


def ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def clean_text(text: str) -> str:
    return re.sub(r"[{}\\]", "", text).strip()


def build_ass(groups: Sequence[Sequence], settings, line_at: Callable[[float], object] | None,
              measure: Callable[[str, int], float], width: int = 1080, height: int = 1920,
              total: float | None = None, offset: float = 0.0, emotion_colors: dict | None = None,
              upper: Callable[[str], str] = lambda s: s) -> str:
    """Build the .ass document.

    groups: caption groups (lists of words with .text/.start/.end/.highlight) on the edited timeline.
    measure(text, size) -> pixel width with the caption font (for auto-shrink).
    offset: subtract from all times (for chunked rendering).
    """
    family = FAMILY_NAMES.get(settings.font_family, "Montserrat ExtraBold")
    base = settings.font_size
    # ASS font size = full line height (ascender+descender), not the em size the UI slider means.
    fs = lambda em: int(round(em * 1.22))  # noqa: E731
    primary = ass_color(settings.primary_color)
    outline = ass_color("#000000")
    shadow = ass_color("#000000", 0x70)
    y = int(height * settings.caption_y)
    x = width // 2
    max_w = width - 160

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{family},{fs(base)},{primary},{ass_color('#FFFFFF', 0x40)},{outline},{shadow},0,0,0,0,100,100,1,0,1,{max(4, base // 12)},4,5,40,40,40,1
Style: Box,{family},{fs(base)},{ass_color('#111111')},{primary},{ass_color('#FFE600')},{ass_color('#000000', 0x80)},0,0,0,0,100,100,1,0,3,{max(10, base // 6)},0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    n_groups = len(groups)
    for gi, group in enumerate(groups):
        if not group:
            continue
        line = line_at(group[0].start) if line_at else None
        emotion = getattr(line, "emotion", "calm")
        energy = getattr(line, "energy", 3)
        accent_hex = (emotion_colors or {}).get(emotion) if settings.emotion_colors and line else None
        accent_hex = accent_hex or settings.highlight_color
        accent = ass_color(accent_hex)
        style = settings.caption_style
        if style == "auto":
            style = EMOTION_STYLE.get(emotion, "pop") if line else "pop"
            if style in ("shake",) and energy < 4:
                style = "pop"

        words = [upper(clean_text(w.text)).rstrip(",.") or clean_text(w.text) for w in group]
        # Auto-shrink so the line fits the width.
        size = base
        while size > 44 and measure(" ".join(words), size) * 1.08 > max_w:
            size -= 4
        bord = max(4, size // 12)

        g_start = group[0].start
        nxt = groups[gi + 1][0].start if gi + 1 < n_groups else (total or group[-1].end + 0.6)
        g_end = min(nxt, group[-1].end + 0.6)
        if total:
            g_end = min(g_end, total)

        def word_span(i: int, active: int | None, hide_after: bool = False) -> str:
            w = group[i]
            txt = words[i]
            tags = []
            if hide_after and active is not None and i > active:
                tags.append("\\alpha&HFF&")
            if w.highlight:
                tags.append(f"\\1c{accent}")
            if active is not None and i == active:
                if style in ("pop", "bounce", "shake"):
                    tags.append("\\fscx112\\fscy112")
                elif style == "glow":
                    tags.append(f"\\3c{accent}\\bord{bord * 2}\\blur6")
                elif style == "neon":
                    tags.append(f"\\1c&HFFFFFF&\\3c{accent}\\bord{bord + 3}\\blur8")
                elif style == "typewriter":
                    tags.append(f"\\1c{accent}")
            # Restore only what this word changed. ("\r" would also reset the auto-shrunk font size and
            # the style's outline, making the words after a highlight jump to a bigger size.)
            changed = "".join(tags)
            restore = ""
            if "\\1c" in changed:
                restore += f"\\1c{primary}"
            if "\\fscx" in changed:
                restore += "\\fscx100\\fscy100"
            if "\\3c" in changed or "\\bord" in changed or "\\blur" in changed:
                if style == "neon":
                    restore += f"\\3c{accent}\\bord{bord}\\blur3"
                elif style == "clean":
                    restore += f"\\3c{outline}\\bord0\\blur1"
                else:
                    restore += f"\\3c{outline}\\bord{bord}\\blur0"
            if "\\alpha" in changed:
                restore += "\\alpha&H00&"
            reset = ("{" + restore + "}") if restore else ""
            return ("{" + "".join(tags) + "}" if tags else "") + txt + reset

        def entrance(first: bool) -> str:
            if not first:
                return ""
            if style == "bounce" or (style == "auto" and energy >= 4):
                return "\\fscx40\\fscy40\\t(0,140,\\fscx118\\fscy118)\\t(140,240,\\fscx100\\fscy100)"
            if style == "slide":
                return f"\\move({x},{y + 60},{x},{y},0,180)\\fad(150,0)"
            if style == "clean":
                return "\\fad(120,0)"
            return "\\fscx80\\fscy80\\t(0,110,\\fscx100\\fscy100)"

        base_tags = f"\\an5\\pos({x},{y})\\fs{fs(size)}\\bord{bord}"
        if style == "clean":
            base_tags += "\\bord0\\shad3\\blur1"
        if style == "neon":
            base_tags += f"\\3c{accent}\\blur3"

        if style == "sweep":
            # One event, karaoke fill sweeping word by word.
            parts = []
            for i, w in enumerate(group):
                k = max(1, int(round((w.end - w.start) * 100)))
                parts.append(f"{{\\kf{k}}}" + (f"{{\\1c{accent}}}" if w.highlight else "") + words[i]
                             + ("{\\1c" + primary + "}" if w.highlight else ""))
            text = "{" + base_tags + entrance(True) + f"\\2c{ass_color('#FFFFFF', 0x90)}" + "}" + " ".join(parts)
            events.append(_ev(0, g_start - offset, g_end - offset, "Cap", text))
            continue

        for wi, w in enumerate(group):
            s = g_start if wi == 0 else w.start
            e = group[wi + 1].start if wi + 1 < len(group) else g_end
            if e - s <= 0.01:
                continue
            first = wi == 0
            if style == "oneword":
                txt = "{" + base_tags + "\\fscx70\\fscy70\\t(0,90,\\fscx100\\fscy100)" + (
                    f"\\1c{accent}" if w.highlight else "") + "}" + words[wi]
                events.append(_ev(0, s - offset, e - offset, "Cap", txt))
                continue
            shake = ""
            if style == "shake" and w.highlight:
                shake = "\\t(0,60,\\frz3)\\t(60,120,\\frz-3)\\t(120,180,\\frz0)"
            hide_after = style == "typewriter"
            body = " ".join(word_span(i, wi, hide_after) for i in range(len(group)))
            events.append(_ev(0, s - offset, e - offset, "Cap", "{" + base_tags + entrance(first) + shake + "}" + body))
            if style == "box":
                # Second layer: same line, only the active word visible, drawn as an opaque box.
                spans = []
                for i in range(len(group)):
                    if i == wi:
                        spans.append(f"{{\\alpha&H00&\\3c{accent}\\1c&H111111&}}" + words[i] + "{\\alpha&HFF&}")
                    else:
                        spans.append(words[i])
                box = ("{" + f"\\an5\\pos({x},{y})\\fs{fs(size)}\\bord{max(10, size // 6)}\\shad0\\alpha&HFF&"
                       + entrance(first) + "}" + " ".join(spans))
                events.append(_ev(1, s - offset, e - offset, "Box", box))
    return head + "\n".join(events) + "\n"


def _ev(layer: int, start: float, end: float, style: str, text: str) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def filter_arg(ass_path: Path, fonts_dir: Path) -> str:
    """FFmpeg -vf value that burns the subtitles (Windows paths escaped for the filter parser)."""
    def esc(p: Path) -> str:
        return str(p.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{esc(ass_path)}':fontsdir='{esc(fonts_dir)}'"


def ensure_static_fonts(fonts_dir: Path, log: Callable[[str], None] = print) -> None:
    """libass can't pick weights inside variable fonts reliably; generate named static instances once."""
    targets = {"Montserrat-ExtraBold.ttf": ("Montserrat ExtraBold", 800), "Montserrat-Black.ttf": ("Montserrat Black", 900)}
    var = fonts_dir / "Montserrat[wght].ttf"
    if not var.exists() or all((fonts_dir / f).exists() for f in targets):
        return
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        log("fonttools not installed; Montserrat captions will fall back to the regular weight.")
        return
    for fname, (family, wght) in targets.items():
        if (fonts_dir / fname).exists():
            continue
        font = instantiateVariableFont(TTFont(str(var)), {"wght": wght})
        name = font["name"]
        for rec in list(name.names):
            if rec.nameID in (1, 4, 16):
                rec.string = family
            elif rec.nameID == 2:
                rec.string = "Regular"
            elif rec.nameID in (6,):
                rec.string = family.replace(" ", "-")
            elif rec.nameID == 17:
                rec.string = "Regular"
        font.save(str(fonts_dir / fname))
        log(f"Generated static font {fname}")
