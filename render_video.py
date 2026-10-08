#!/usr/bin/env python3
"""Render a clip of the Bad Apple video as a candlestick chart video.

Usage:
  python3 render_video.py --start 12.0 --dur 10 --cols 60 --out out/clip.mp4 \
      [--workers 8] [--side-by-side] [--keep-frames]
"""
import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
SRC_VIDEO = os.path.join(ROOT, "src", "badapple_video.mp4")
SRC_AUDIO = os.path.join(ROOT, "src", "badapple_audio.m4a")
SRC_W, SRC_H = 480, 360
FPS = 30
WARMUP = 10
EMA = 0.5


def decode_frames(first, count):
    """Yield `count` gray 480x360 frames starting at global frame `first` (frame accurate)."""
    # seek half a frame before the target so the first decoded frame is exactly `first`
    ss = max(0.0, (first - 0.5) / FPS)
    cmd = [FFMPEG, "-v", "error", "-ss", "%.6f" % ss, "-i", SRC_VIDEO,
           "-frames:v", str(count), "-vf", "format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsize = SRC_W * SRC_H
    n = 0
    try:
        while n < count:
            buf = p.stdout.read(fsize)
            if len(buf) < fsize:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(SRC_H, SRC_W).copy()
            n += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def worker(idx, g0, g1, warm, cols, frames_dir, first, q):
    """Render global frames [g0, g1) after warming up on the `warm` frames before g0."""
    try:
        sys.path.insert(0, ROOT)
        import candlechart as cc
        w0 = g0 - warm
        scene = None
        done = 0
        for k, gray in enumerate(decode_frames(w0, warm + (g1 - g0))):
            g = w0 + k
            img, scene = cc.render(gray, n_cols=cols, prev=scene, ema=EMA, t=g, ambient=True)
            if g >= g0:
                img.save(os.path.join(frames_dir, "%06d.png" % (g - first)),
                         compress_level=1)
                done += 1
                q.put(("tick", idx, 1))
        if done != g1 - g0:
            raise RuntimeError("chunk %d decoded only %d of %d frames" % (idx, done, g1 - g0))
        q.put(("done", idx, None))
    except BaseException as e:  # noqa
        import traceback
        q.put(("error", idx, traceback.format_exc()))


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        raise SystemExit("command failed: %s" % " ".join(cmd[:3]))
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--cols", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(out))[0]
    frames_dir = os.path.join(ROOT, "out", "frames_" + name)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    first = int(round(a.start * FPS))
    n = int(round(a.dur * FPS))
    # clamp to the length of the source video
    total = int(float(run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=duration", "-of", "csv=p=0", SRC_VIDEO]).strip()) * FPS + 0.5)
    if first + n > total:
        n = max(0, total - first)
        print("clamping to %d frames (end of video)" % n)
    if n <= 0:
        raise SystemExit("nothing to render")
    start_t = first / FPS
    dur_t = n / FPS

    workers = max(1, min(a.workers, n))
    base, extra = divmod(n, workers)
    chunks, g = [], first
    for i in range(workers):
        c = base + (1 if i < extra else 0)
        chunks.append((g, g + c))
        g += c

    print("rendering %d frames (global %d..%d), %d workers, cols=%d" % (n, first, first + n - 1, workers, a.cols))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for i, (g0, g1) in enumerate(chunks):
        warm = min(WARMUP, g0)
        p = ctx.Process(target=worker, args=(i, g0, g1, warm, a.cols, frames_dir, first, q))
        p.start()
        procs.append(p)

    done_frames, finished, last_print = 0, 0, 0
    err = None
    while finished < workers and err is None:
        try:
            kind, idx, payload = q.get(timeout=5)
        except Exception:
            if all(not p.is_alive() for p in procs) and finished < workers:
                err = "workers exited unexpectedly"
            continue
        if kind == "tick":
            done_frames += payload
            if done_frames - last_print >= 30 or done_frames == n:
                last_print = done_frames
                el = time.time() - t0
                print("progress %d/%d frames  %.1fs elapsed  %.3f s/frame (wall)  eta %.0fs"
                      % (done_frames, n, el, el / done_frames, el / done_frames * (n - done_frames)), flush=True)
        elif kind == "done":
            finished += 1
        elif kind == "error":
            err = "worker %d failed:\n%s" % (idx, payload)
    if err:
        for p in procs:
            p.terminate()
        raise SystemExit(err)
    for p in procs:
        p.join()
    render_t = time.time() - t0

    # ---- encode
    t1 = time.time()
    aud = ["-ss", "%.6f" % start_t, "-t", "%.6f" % dur_t, "-i", SRC_AUDIO]
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart"]
    run([FFMPEG, "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "%06d.png")]
        + aud + ["-map", "0:v", "-map", "1:a"] + enc + [out])
    print("wrote", out)

    if a.side_by_side:
        sbs = os.path.splitext(out)[0] + "_sbs.mp4"
        src_ss = max(0.0, (first - 0.5) / FPS)
        flt = ("[0:v]setpts=PTS-STARTPTS,fps=30,scale=960:720:flags=lanczos,setsar=1[l];"   # source on the left
               "[1:v]setpts=PTS-STARTPTS,scale=960:720:flags=lanczos,setsar=1[r];"  # render on the right
               "[l][r]hstack=inputs=2[v]")
        run([FFMPEG, "-v", "error", "-y", "-ss", "%.6f" % src_ss, "-i", SRC_VIDEO, "-i", out,
             "-filter_complex", flt, "-map", "[v]", "-map", "1:a", "-frames:v", str(n),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS),
             "-c:a", "copy", "-shortest", "-movflags", "+faststart", sbs])
        print("wrote", sbs)
    enc_t = time.time() - t1

    if not a.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    print("timing: render %.1fs (%.3f s/frame wall, %.3f s/frame per-worker-cpu est.), encode %.1fs, total %.1fs"
          % (render_t, render_t / n, render_t / n * workers, enc_t, render_t + enc_t))


if __name__ == "__main__":
    main()
