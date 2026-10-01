# GOAT Studio

**A local AI video editor for Instagram Reels and YouTube Shorts.**

GOAT Studio turns raw footage into an edited 9:16 reel. It listens to the speech, watches the picture, works out the story and mood, decides how to edit, and then checks its own output. Everything runs on your computer with free, open-source tools. There are no cloud APIs, no subscriptions and no uploads.

It is built for English, Hindi, Hinglish (Hindi + English) and Gujlish (Gujarati + English) creators, but works with any language Whisper supports.

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: one-click installer (Windows)](#option-a-one-click-installer-windows)
  - [Option B: manual install](#option-b-manual-install)
- [Starting the app](#starting-the-app)
- [Using GOAT Studio](#using-goat-studio)
- [Outputs](#outputs)
- [Your libraries: B-roll, music, looks, presets](#your-libraries)
- [Settings reference](#settings-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Credits and licenses](#credits-and-licenses)

---

## Features

**Understanding**
- **Speech to text** with word-level timing (Whisper), automatic language detection, and Roman-script captions for Hinglish and Gujlish.
- **Transcript correction:** a local language model fixes misheard words using the context of the whole video.
- **Story analysis:** the format (tutorial, storytime, comedy, podcast…), mood, pace, story beats, best hook line and the emotion of every sentence.
- **Picture analysis:** face size and position, camera motion, existing cuts and exposure, plus scene descriptions from a local vision model.

**Editing**
- **Autopilot** chooses the editing style for each video and explains every decision.
- **Smart cuts:**
  - silences, um/uh fillers, retakes, false starts and mic checks ("1, 2, 3, testing")
  - optional **cold open**: the strongest line is moved to the very start
- **Edit by text:** delete words or lines in the transcript and the video is cut to match.
- **Caption editor:** after a render, click **Edit captions** to fix words, timing, highlights, animation, size or position. Clicking a caption jumps the preview to it. Only the captions are re-drawn, so changes are ready in seconds, and every version is kept.
- **Virtual camera:** punch-ins on key words, slow push-ins, shakes on peak moments, whip and zoom transitions, with face tracking. With two or more people in frame, it **follows whoever is speaking**.
- **Reframing** from any aspect ratio to 1080×1920, either a face-following crop or a blurred background.

**Look and sound**
- **Captions:** 11 animated styles rendered by libass (pop, bounce, karaoke box, highlight sweep, typewriter, glow, slide, one word at a time, neon, clean, shake). Highlight colors follow the emotion of each sentence, and captions are kept away from the speaker's face.
- **Graphics:** hook title card, or a title *behind* the speaker (AI person cutout), emoji pop-ins, B-roll cutaways, progress bar, logo watermark and end card.
- **Color:** automatic exposure correction, mood-based grading, 8 cinematic looks (LUTs; you can add your own `.cube` files), and stabilization for shaky footage.
- **Audio:**
  - voice cleanup: noise removal with DeepFilterNet, voice EQ and de-esser
  - mastering to -14 LUFS, the standard loudness for Instagram and YouTube
  - generated sound effects, and background music picked by mood that ducks under your voice

**Production**
- **Viral Clips:** give it a long podcast or video and it finds, scores and edits the best 20–60 second moments.
- **Batch mode** with a watch folder and resume after a restart.
- **Projects:** save an edit and reopen it later.
- **Import from links:** YouTube, Instagram, TikTok, X, Facebook and more, via yt-dlp.
- **Publishing kit:** cover image, Instagram caption and hashtags, YouTube Shorts title, SRT/VTT subtitles, English subtitles, a Premiere Pro / DaVinci Resolve XML, and 1:1, 4:5 and 16:9 versions.
- **Quality check** after every render: format, length, loudness, caption coverage and black frames.

---

## How it works

```
 video ──► Listen ──────► Watch ──────► Understand ────► Decide ─────► Edit ─────────► Check
          Whisper         OpenCV         Qwen2.5 7B       Autopilot     FFmpeg+libass    QA report
          DeepFilterNet   Moondream      (story, emotion, (style per    virtual camera,  + publishing
          transcript fix  faces/motion    hook, cuts)      content type) captions, sound  kit
```

| Job | Tool (all free and open source) |
|---|---|
| Speech to text | [openai-whisper](https://github.com/openai/whisper) |
| Story, emotion, editing decisions, transcript correction, clip scoring, post text, translation | [Qwen2.5 7B](https://ollama.com/library/qwen2.5) via [Ollama](https://ollama.com) |
| Scene descriptions | [Moondream](https://ollama.com/library/moondream) via Ollama |
| Voice cleanup | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (fallback: [noisereduce](https://github.com/timsainb/noisereduce)) |
| Person cutout | U²-Net-p ([rembg](https://github.com/danielgatis/rembg) model) with [onnxruntime](https://onnxruntime.ai) |
| Faces, active speaker, camera | [OpenCV](https://opencv.org) |
| Rendering, captions, stabilization, LUTs | [FFmpeg](https://ffmpeg.org) with libass, vid.stab and lut3d, plus [MoviePy](https://github.com/Zulko/moviepy) |
| Loudness | [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) |
| Link import | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Interface | [Gradio](https://gradio.app) |

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11 (64-bit) | Windows 11 |
| CPU | 4 cores | 8 threads or more |
| RAM | 8 GB (rule-based director only) | **16 GB** (AI director + vision) |
| Disk | 6 GB | 12 GB free |
| GPU | not required | NVIDIA with 6 GB+ VRAM (used automatically) |

GOAT Studio is developed and tested on Windows. The Python code is cross-platform, but the `.bat` installer and launcher are Windows-only. On macOS or Linux, use the [manual install](#option-b-manual-install).

---

## Installation

### Option A: one-click installer (Windows)

1. Download the repository: **Code → Download ZIP** (then extract it), or clone it:
   ```bash
   git clone https://github.com/<your-username>/goat-studio.git
   ```
2. Double-click **`Install GOAT Studio.bat`**.

The installer sets up everything in the project folder:

1. Python 3.11 (installed with `winget` if missing)
2. A virtual environment in `.venv\`
3. PyTorch: the CUDA build if an NVIDIA GPU is found, otherwise the CPU build
4. The Python libraries from `requirements.txt`
5. A separate Python 3.11 environment for DeepFilterNet voice cleanup, in `tools\dfenv\`
6. The caption fonts and the person-cutout model
7. Ollama, plus the `qwen2.5:7b` (about 4.7 GB) and `moondream` (about 1.7 GB) models

The first install downloads about 8–10 GB and takes 15–40 minutes depending on your connection.

### Option B: manual install

Run these from the project folder.

1. Create and activate a virtual environment (Python 3.11 recommended; 3.11–3.14 work):
   ```bash
   py -3.11 -m venv .venv
   ```
   ```bash
   .venv\Scripts\activate
   ```
2. Install PyTorch. For CPU only:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   Or, for an NVIDIA GPU:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install the app libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Fonts and the cutout model. Download these into `fonts\` from [Google Fonts](https://github.com/google/fonts) (all OFL-licensed): `Montserrat[wght].ttf`, `Poppins-Black.ttf`, `Poppins-ExtraBold.ttf`, `Anton-Regular.ttf`, `BebasNeue-Regular.ttf`. Then download the cutout model:
   ```bash
   curl -L -o models/u2netp.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
   ```
5. Install [Ollama](https://ollama.com/download) and pull the models:
   ```bash
   ollama pull qwen2.5:7b
   ```
   ```bash
   ollama pull moondream
   ```
6. Optional, for studio voice cleanup: DeepFilterNet needs older torch and numpy versions, so give it its own Python 3.11 environment:
   ```bash
   py -3.11 -m venv tools\dfenv
   ```
   ```bash
   tools\dfenv\Scripts\python -m pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   ```
   ```bash
   tools\dfenv\Scripts\python -m pip install deepfilternet soundfile "numpy<2"
   ```
   Without this step, the app uses `noisereduce` instead.

FFmpeg is bundled through `imageio-ffmpeg`, so there's nothing extra to install.

**What still works without the optional parts:**

| Missing | Effect |
|---|---|
| Ollama | The app uses a rule-based director (keywords, voice energy and pitch). Transcript correction, the Viral Clips tab, post copy and English subtitles are unavailable. |
| moondream | Scene understanding uses frame analysis only. |
| `tools\dfenv` | Noise removal uses noisereduce. |
| `models\u2netp.onnx` | The hook title uses a card instead of appearing behind the speaker. |

---

## Starting the app

Double-click **`Start GOAT Studio.bat`**. It starts Ollama if needed and opens http://127.0.0.1:7860.

Or start it manually:

```bash
.venv\Scripts\python app.py
```

---

## Using GOAT Studio

### Quick edit
1. **Source:** drop a video in, or use **Import link** with a YouTube, Instagram, TikTok, X or Facebook URL.
2. Leave **Mode** on *Autopilot* and **Quality** on *Final 1080p*, or choose *Draft preview* for a fast check.
3. Click **Auto edit**. Progress shows under the button, and full details are in **Activity log**.
4. The finished reel plays under **Output**, with all its files listed below it and the **Checks** results.
5. Open the **Analysis** tab to see what was understood and why each editing decision was made.

### Editing captions (Captions tab)
After a render, click **Edit captions** under the output (or open the **Captions** tab).
- **Click a caption** to jump the preview to that moment.
- **Change the words** in the *Caption* column. If the word count stays the same, the original word timing is kept.
- **Change the timing** by editing *Start* / *End*.
- **Highlight a word** by wrapping it in `*stars*`.
- **Remove a caption** by clearing its text.
- **Add a caption** with **+**.
- Change the **Animation**, **Size** or **Position** for the whole video.
- Click **Apply changes**. A new version (`reel_…_v2.mp4`, `_v3`, …) with matching subtitles is saved in a few seconds; earlier versions are kept.

The editor works by keeping a caption-free copy of each render in `outputs\.cache\`. You can delete that folder to free space, but the renders it covered can then no longer be caption-edited.

### Fine-tuning (Review tab)
- **Transcript box:**
  - fix misheard words (spelling only, timing is kept)
  - **delete words or whole lines** to cut them from the video
  - wrap a word in `*stars*` to highlight it
- **Edit plan:** every cell can be edited, for example emotion, emoji, sound effect, camera move, transition, `keep` (*yes* / *CUT*) and `broll`.
- **Hook title:** change the text shown in the first three seconds.
- Click **Render** (under *Step by step*) to apply your changes.
- **Save** and **Open** let you come back to a project later.

### Step by step
**Transcribe** → fix words → **Plan edit** → adjust the plan → **Render**.

### Viral Clips
Paste a link or upload a long video, choose how many clips you want, and click **Find clips**. You get a table of clips sorted by score, each with a reason, and every clip is fully edited.

### Batch
Add files or a folder path, then click **Start**. Tick *Watch folder* to keep processing new files until you click **Stop**. Videos that are already finished are skipped when you restart.

### Style
Caption font, animation, colors and size, plus brand settings (logo, handle, end card) that you can save as **presets**.

---

## Outputs

Everything goes to `outputs\`:

| File | What it is |
|---|---|
| `reel_*.mp4` | The 1080×1920 master, H.264/AAC, -14 LUFS |
| `*.cover.jpg` | Cover image from the best frame, with the title |
| `*.srt`, `*.vtt` | Subtitles for the final timeline |
| `*.en.srt` | English subtitles (for Hindi/Gujarati videos) |
| `*.post.txt` | Instagram caption, hashtags, YouTube Shorts title and description |
| `*.premiere.xml` | Every cut on the original footage. Import into Premiere Pro or DaVinci Resolve with File → Import → XML |
| `*_1x1.mp4`, `*_4x5.mp4`, `*_16x9.mp4` | Extra formats, if enabled |
| `*.plan.json`, `*.json` | The edit plan and word timings |
| `draft_*.mp4` | Draft previews |
| `reel_*_v2.mp4`, `_v3`… | Versions saved from the caption editor (with matching `.srt` / `.vtt`) |
| `.cache\` | Caption-free masters used by the caption editor |

---

## Your libraries

| Folder | Put here | How it's used |
|---|---|---|
| `broll\` | Clips or images named by what they show, e.g. `money cash rupees.mp4`, `gym workout.mp4` | Cut in where the director suggests a matching visual |
| `music\<mood>\` | Tracks in `energetic`, `upbeat`, `funny`, `chill`, `dramatic`, `emotional` | Picked by the detected mood and ducked under your voice |
| `luts\` | `.cube` color looks | Appear in Settings → Look |
| `presets\` | Created with *Save* in the Style tab | Your saved brand looks |
| `projects\` | Created with *Save* in the Review tab | Reopen an edit later |

Free sources:
- **B-roll:** [Pexels Videos](https://www.pexels.com/videos/), [Pixabay Videos](https://pixabay.com/videos/)
- **Music:** [YouTube Audio Library](https://studio.youtube.com), [Pixabay Music](https://pixabay.com/music/)

---

## Settings reference

In **Autopilot**, the app picks the settings for each video, and anything you switch **off** in Settings stays off. In **Manual** mode, the Settings tab is used exactly as set.

| Group | Settings |
|---|---|
| Understanding | Speech model (tiny/base/small/medium), language and caption script, director model, story analysis, scene understanding, transcript correction |
| Editing | Silence removal and threshold, filler removal, retake removal, cold open, landscape reframing |
| Picture | Look (LUT), hook title style, color correction, stabilization, camera moves, face tracking, active-speaker follow, emoji, hook title, transitions, B-roll, progress bar |
| Sound | Noise removal, voice EQ and loudness, sound effects, music library, your own music track and level |
| Export | Cover, post copy, subtitles, English subtitles, Premiere/Resolve XML, additional formats |

---

## Performance

Measured on a laptop with an Intel i7-1165G7, 16 GB RAM and no GPU, editing a 14-second clip:

| Stage | Time |
|---|---|
| Transcription (Whisper small) | ~15 s |
| Transcript correction + story + per-line direction (Qwen2.5 7B) | ~2–3 min |
| Scene descriptions (Moondream, 3 frames) | ~1 min |
| Rendering 1080×1920 | ~25 s |
| Publishing kit (subtitles, translation, post copy) | ~1–2 min |

Tips:
- **Draft preview** renders about twice as fast.
- Turning off **Scene understanding** saves about a minute per video.
- An **NVIDIA GPU** speeds up Whisper a lot and enables NVENC hardware encoding automatically.
- For long videos in **Viral Clips**, transcription takes the most time. Set the *Speech model* to `base` for about twice the speed at lower accuracy.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Top bar shows **Director** with a grey dot | Ollama isn't running or the model is missing. Start Ollama, then run `ollama pull qwen2.5:7b`. |
| **Vision** dot is grey | Run `ollama pull moondream`. |
| Captions have wrong words | Use a bigger speech model, set the language (e.g. *Gujlish (Roman script)*), and fix the words in the Review tab. |
| Instagram link fails | The post may be private or rate-limited. Export `cookies.txt` from your browser (e.g. the *Get cookies.txt LOCALLY* extension) and add it under Import link. **Keep that file private**, because it contains your login. |
| Link imports stop working | Update the downloader with `.venv\Scripts\python -m pip install -U yt-dlp` |
| "DeepFilterNet unavailable" in the log | Optional. Run the installer again, or follow step 6 of the manual install. |
| The browser tab still shows the old icon | Hard-refresh with Ctrl+F5. |
| Port 7860 is already in use | Close the other app, or change `server_port` at the bottom of `app.py`. |

---

## Project structure

```
app.py                 UI (Gradio) and pipeline: audio, transcription, cuts, render, publishing
analyzer.py            Picture understanding + Autopilot decision rules
director.py            AI director: story, emotion, per-line plan, transcript correction, viral-clip finder
effects.py             Sound effects, mixing and loudness, face/active-speaker tracking, virtual camera, frame composer
captions_ass.py        libass caption engine (11 animation styles)
finishing.py           Cinematic looks (LUT generation)
segment.py             Person cutout and "text behind subject"
publish.py             Cover, subtitles, translation, post copy, Premiere XML, aspect versions
tools/df_denoise.py    DeepFilterNet helper (runs in tools/dfenv)
assets/                Icons and favicon
broll/  music/  luts/  Your libraries
Install GOAT Studio.bat / Start GOAT Studio.bat
```

---

## Credits and licenses

GOAT Studio builds on the open-source projects listed under [How it works](#how-it-works). Each keeps its own license:
- **Fonts:** Montserrat, Poppins, Anton and Bebas Neue, all under the SIL Open Font License.
- **Models:**
  - Whisper: MIT
  - Qwen2.5: Apache-2.0
  - Moondream: Apache-2.0
  - U²-Net: Apache-2.0
  - DeepFilterNet: MIT / Apache-2.0
- **Tools:** FFmpeg is LGPL/GPL, depending on the build.

Only edit and publish footage you own or have permission to use.
