#!/usr/bin/env python3
"""Per-video-frame beat/accent envelope for Bad Apple!! (numpy + Pillow only).

Writes src/beats.npz with: accent, energy, big (float32, len = n_frames), fps.
Also writes check plots to out/beats/.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(ROOT, "src", "badapple_audio.m4a")
OUT_NPZ = os.path.join(ROOT, "src", "beats.npz")
OUT_DIR = os.path.join(ROOT, "out", "beats")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

SR = 22050
FPS = 30
N_FFT = 2048
HOP = 256                     # 11.6 ms
N_FRAMES = 6572               # video frame count (ceil(219.0667*30))

HALF_LIFE = 4.0               # frames, accent decay
NORM_WIN_S = 4.0              # rolling percentile window (s)
BIG_MIN_GAP_S = 1.6
BIG_MIN_ENERGY = 0.35


def decode(path):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def stft_mag(y):
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    win = np.hanning(N_FFT).astype(np.float32)
    frames = sliding_window_view(yp, N_FFT)[::HOP]
    out = np.empty((frames.shape[0], N_FFT // 2 + 1), np.float32)
    for i in range(0, frames.shape[0], 1024):
        out[i:i + 1024] = np.abs(np.fft.rfft(frames[i:i + 1024] * win, axis=1))
    return out  # (T, bins); frame t centred at t*HOP/SR


def band_flux(logmag, freqs, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    d = np.diff(logmag[:, sel], axis=0, prepend=logmag[:1, sel])
    return np.maximum(d, 0).mean(axis=1)


def onset_strength(y):
    mag = stft_mag(y)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    logmag = np.log1p(1000.0 * mag / (N_FFT / 4))
    kick = band_flux(logmag, freqs, 40, 150)
    snare = band_flux(logmag, freqs, 150, 4000)
    # each band normalised by its own robust scale so both contribute
    hat = band_flux(logmag, freqs, 4000, 10000)   # hats/snare crack: clean 8th-note grid
    kick /= np.percentile(kick, 99) + 1e-9
    snare /= np.percentile(snare, 99) + 1e-9
    hat /= np.percentile(hat, 99) + 1e-9
    onset = 1.0 * kick + 0.8 * snare + 0.7 * hat
    # suppress sustained "busy" texture: keep only what pokes above the local (~0.4 s) mean
    loc = box(onset, int(0.4 * SR / HOP))
    onset = np.maximum(onset - 0.7 * loc, 0)
    return onset.astype(np.float32)


def to_frames(x, hop_s, n_frames, how="max"):
    """Pool curve x (sample times t*hop_s) into video frames [f/FPS,(f+1)/FPS)."""
    t = np.arange(len(x)) * hop_s
    idx = np.floor(t * FPS + 1e-9).astype(int)
    out = np.zeros(n_frames, np.float32)
    ok = (idx >= 0) & (idx < n_frames)
    if how == "max":
        np.maximum.at(out, idx[ok], x[ok])
    else:
        s = np.zeros(n_frames); c = np.zeros(n_frames)
        np.add.at(s, idx[ok], x[ok]); np.add.at(c, idx[ok], 1)
        out = (s / np.maximum(c, 1)).astype(np.float32)
    return out


def rolling_percentile(x, win, q):
    h = win // 2
    xp = np.pad(x, (h, win - h - 1), mode="edge")
    return np.percentile(sliding_window_view(xp, win), q, axis=1)


def decay_env(x, half_life=HALF_LIFE):
    """Instant attack, exponential release."""
    k = 0.5 ** (1.0 / half_life)
    y = np.empty_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = v if v > prev * k else prev * k
        y[i] = prev
    return y


def box(x, win):
    ker = np.ones(win) / win
    return np.convolve(np.pad(x, (win // 2, win - win // 2 - 1), mode="edge"),
                       ker, mode="valid")


def local_maxima(x, radius):
    xp = np.pad(x, radius, mode="constant", constant_values=-np.inf)
    w = sliding_window_view(xp, 2 * radius + 1)
    return (x >= w.max(axis=1)) & (x > 0)


def estimate_bpm(onset, hop_s):
    o = onset - onset.mean()
    n = len(o)
    f = np.fft.rfft(o, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    lags = np.arange(n) * hop_s
    bpm = np.full(n, np.nan); bpm[1:] = 60.0 / lags[1:]
    sel = (bpm >= 70) & (bpm <= 200)
    # mild prior favouring ~120 BPM to reduce octave errors
    prior = np.exp(-0.5 * (np.log2(bpm[sel] / 120.0) / 1.0) ** 2)
    sc = ac[sel] * prior
    i = np.argmax(sc)
    b = bpm[sel][i]
    # parabolic refinement
    j = np.where(sel)[0][i]
    a, c0, c = ac[j - 1], ac[j], ac[j + 1]
    den = a - 2 * c0 + c
    off = 0.5 * (a - c) / den if den != 0 else 0.0
    b_ref = 60.0 / ((j + off) * hop_s)
    return b_ref, b, ac, lags


def compute(n_frames=N_FRAMES):
    y = decode(AUDIO)
    dur = len(y) / SR
    onset = onset_strength(y)
    hop_s = HOP / SR

    # ---- onset -> per video frame (max-pool) ----
    on_f = to_frames(onset, hop_s, n_frames, "max")

    # ---- accent: rolling-p95 normalised, then fast attack / exp decay ----
    win = int(round(NORM_WIN_S * FPS))
    p95 = rolling_percentile(on_f, win, 95)
    p95 = box(p95, 15)
    floor = 0.15 * np.percentile(on_f, 95)      # keeps silence from being boosted
    on_norm = on_f / np.maximum(p95, floor)
    accent = np.clip(decay_env(on_norm), 0, 1).astype(np.float32)

    # ---- energy: RMS over ~0.5 s ----
    n_samp = int(round((n_frames / FPS) * SR))
    yy = np.zeros(n_samp, np.float32); m = min(n_samp, len(y)); yy[:m] = y[:m]
    spf = SR / FPS
    fe = np.minimum((np.arange(n_samp) / spf).astype(int), n_frames - 1)
    pw = np.bincount(fe, weights=yy.astype(np.float64) ** 2, minlength=n_frames) / spf
    rms = np.sqrt(box(pw, 15))                  # 15 frames = 0.5 s
    energy = (rms / rms.max()).astype(np.float32)
    energy_s = (box(energy, 15)).astype(np.float32)   # extra smoothing (~0.5 s)
    energy_s = np.clip(energy_s / energy_s.max(), 0, 1).astype(np.float32)

    # ---- big: strongest hits only ----
    # score: onset relative to a LONG (12 s) rolling 90th percentile of local peaks
    lw = 12 * FPS
    p_long = box(rolling_percentile(on_f, lw, 97), 31)
    score = on_f / np.maximum(p_long, floor)
    cand = local_maxima(score, 3)
    cand &= score >= 0.95
    cand &= energy_s >= BIG_MIN_ENERGY
    idx = np.where(cand)[0]
    # greedy: strongest first, enforce min gap
    order = idx[np.argsort(-score[idx])]
    gap = int(BIG_MIN_GAP_S * FPS)
    taken = []
    for i in order:
        if all(abs(i - j) >= gap for j in taken):
            taken.append(i)
    taken = np.array(sorted(taken), int)
    big_imp = np.zeros(n_frames, np.float32)
    if len(taken):
        big_imp[taken] = 1.0
    big = np.clip(decay_env(big_imp), 0, 1).astype(np.float32)

    return dict(accent=accent, energy=energy_s, big=big, onset=onset, on_f=on_f,
                big_idx=taken, dur=dur, hop_s=hop_s)


def plot(res, t0, t1, path, n_frames=N_FRAMES):
    W, H = 1800, 620
    L, R, T, B = 60, 20, 30, 40
    img = Image.new("RGB", (W, H), (18, 18, 24))
    d = ImageDraw.Draw(img)
    pw, ph = W - L - R, H - T - B
    f0, f1 = int(t0 * FPS), min(int(t1 * FPS), n_frames)
    def X(t): return L + (t - t0) / (t1 - t0) * pw
    def Y(v): return T + ph - v * ph
    # grid: seconds; fine tick lines each beat (138 BPM) would clutter -> 1 s ticks
    for s in range(int(np.ceil(t0)), int(t1) + 1):
        x = X(s)
        d.line([(x, T), (x, T + ph)], fill=(50, 50, 62), width=1)
        d.line([(x, T + ph), (x, T + ph + 8)], fill=(200, 200, 200), width=1)
        if (s - int(np.ceil(t0))) % (1 if t1 - t0 <= 12 else 2) == 0:
            d.text((x - 8, T + ph + 12), f"{s}s", fill=(200, 200, 200))
    for v in (0, 0.5, 1):
        d.line([(L, Y(v)), (L + pw, Y(v))], fill=(60, 60, 75))
        d.text((10, Y(v) - 5), f"{v:.1f}", fill=(200, 200, 200))
    # raw onset (per frame), scaled
    on = res["on_f"][f0:f1]; on = on / max(on.max(), 1e-9)
    ts = (np.arange(f0, f1) + 0.5) / FPS
    for t, v in zip(ts, on):
        d.line([(X(t), Y(0)), (X(t), Y(v * 0.5))], fill=(70, 70, 110), width=2)
    def curve(arr, col, w=2):
        pts = [(X(t), Y(float(v))) for t, v in zip(ts, arr[f0:f1])]
        d.line(pts, fill=col, width=w)
    curve(res["energy"], (80, 200, 120))
    curve(res["accent"], (255, 190, 60), 3)
    curve(res["big"], (255, 70, 90), 3)
    for i in res["big_idx"]:
        if f0 <= i < f1:
            x = X((i + 0.5) / FPS)
            d.polygon([(x - 6, T), (x + 6, T), (x, T + 12)], fill=(255, 70, 90))
    d.text((L, 8), f"accent (orange)  energy (green)  big (red, triangles = events)  raw onset (blue bars)   {t0}-{t1}s",
           fill=(230, 230, 230))
    img.save(path)


def main():
    res = compute()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(OUT_NPZ, accent=res["accent"], energy=res["energy"], big=res["big"],
             fps=np.float32(FPS))
    bpm_r, bpm_c, ac, lags = estimate_bpm(res["onset"], res["hop_s"])
    plot(res, 56, 66, os.path.join(OUT_DIR, "check_56s.png"))
    plot(res, 0, 30, os.path.join(OUT_DIR, "check_0s.png"))
    bi = res["big_idx"]
    print(f"audio duration {res['dur']:.3f}s, n_frames {len(res['accent'])}")
    print("shapes", res["accent"].shape, res["energy"].shape, res["big"].shape)
    print(f"BPM autocorr: {bpm_r:.2f} (raw lag {bpm_c:.2f})")
    print(f"big events: {len(bi)}  ({len(bi) / (len(res['accent']) / FPS) * 60:.1f}/min)")
    print("first 15 (s):", [round((i) / FPS, 2) for i in bi[:15]])
    print("accent mean %.3f  big mean %.3f  energy mean %.3f" % (
        res["accent"].mean(), res["big"].mean(), res["energy"].mean()))


if __name__ == "__main__":
    main()
