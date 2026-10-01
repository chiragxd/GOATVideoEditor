"""DeepFilterNet speech enhancement, run inside tools/dfenv (Python 3.11).

Usage: python df_denoise.py input.wav output.wav [atten_db]
"""

import sys

import numpy as np
import soundfile as sf
import torch
from df.enhance import enhance, init_df


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    atten = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    model, state, _ = init_df(log_level="ERROR")
    audio, sr = sf.read(src, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != state.sr():
        x_old = np.linspace(0, 1, audio.size, endpoint=False)
        n = int(round(audio.size * state.sr() / sr))
        audio = np.interp(np.linspace(0, 1, n, endpoint=False), x_old, audio).astype(np.float32)
    out = enhance(model, state, torch.from_numpy(audio)[None], atten_lim_db=atten).squeeze(0).numpy()
    if sr != state.sr():
        x_old = np.linspace(0, 1, out.size, endpoint=False)
        n = int(round(out.size * sr / state.sr()))
        out = np.interp(np.linspace(0, 1, n, endpoint=False), x_old, out).astype(np.float32)
    sf.write(dst, out, sr, subtype="FLOAT")


if __name__ == "__main__":
    main()
