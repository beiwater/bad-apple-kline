"""B站封面：三种构图，16:9 直接渲染，左侧行情区放标题。"""
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import candlechart as cc

BG = cc.PALETTE["bg"]
din = lambda s: ImageFont.truetype("/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf", s)
hei = lambda s: ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", s, index=2)   # W6


def _extract(sec):
    path = f"out/cover/src_{sec}.png"
    if not os.path.exists(path):
        os.makedirs("out/cover", exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", str(sec),
                        "-i", "src/badapple_video.mp4", "-frames:v", "1", "-vf", "format=gray", path], check=True)
    return path


def place(sec, box=None, size=None, at=(160, 0)):
    """把源帧（可裁切、缩放）放进 640x360 画布；空白按极性补白或补黑。"""
    _extract(sec)
    src = Image.open(f"out/cover/src_{sec}.png").convert("L")
    if box:
        src = src.crop(box)
    if size:
        src = src.resize(size, Image.LANCZOS)
    fill = 0 if (np.array(src) < 128).mean() > 0.5 else 255
    canvas = Image.new("L", (640, 360), fill)
    canvas.paste(src, at)
    return np.array(canvas)


def chart(gray, beat_sec):
    cc.W, cc.H, cc.UI = 1920, 1080, False
    cc._BG_CACHE.clear()
    return cc.render(gray, 96, t=int(beat_sec * 30))[0].convert("RGB")


def title(img, y=190):
    d = ImageDraw.Draw(img)
    x0 = 70
    sw = dict(stroke_fill=BG)
    d.text((x0, 66), "BADAPPLE / 1D", font=din(44), fill=(250, 250, 250), stroke_width=10, **sw)
    d.text((x0 + 350, 58), "138.00", font=din(60), fill=(250, 250, 250), stroke_width=10, **sw)
    tx, ty, r = x0 + 590, 90, 16
    d.polygon([(tx - r, ty + r * 0.6), (tx + r, ty + r * 0.6), (tx, ty - r)], fill=cc.PALETTE["up"])
    d.text((x0 + 618, 66), "+999.99%", font=din(44), fill=cc.PALETTE["up"], stroke_width=10, **sw)
    d.text((x0 - 6, y), "Bad Apple!!", font=din(200), fill=(250, 250, 250), stroke_width=18, **sw)
    d.text((x0, y + 245), "但它是一张", font=hei(112), fill=(250, 250, 250), stroke_width=16, **sw)
    d.text((x0, y + 385), "K线图", font=hei(190), fill=(232, 232, 72), stroke_width=18, **sw)
    return img


# 女巫的包围盒
g = np.array(Image.open(_extract("20")).convert("L"))
ys, xs = np.nonzero(g > 128)
m = (xs > 50) & (ys < 300)
bx = (xs[m].min() - 8, ys[m].min() - 8, xs[m].max() + 8, ys[m].max() + 8)
bw, bh = bx[2] - bx[0], bx[3] - bx[1]
s = 400 / bw
variants = {
    "A_witch": (place("20", bx, (int(bw * s), int(bh * s)), (232, 330 - int(bh * s))), 56.97),
    "B_sword": (place("61.5"), 60.87),
    "C_flandre": (place("56.6"), 56.97),
}
thumbs = []
for name, (gray, bs) in variants.items():
    img = title(chart(gray, bs))
    img.save(f"out/cover/cover_{name}.png")
    thumbs.append(img)
sheet = Image.new("RGB", (480 * 3 + 20, 270 + 135 + 30), (14, 22, 34))
for k, im in enumerate(thumbs):
    sheet.paste(im.resize((480, 270), Image.LANCZOS), (k * 490, 0))
    sheet.paste(im.resize((240, 135), Image.LANCZOS), (k * 490, 285))   # B站信息流里的实际大小
sheet.save("out/cover/compare.png")
print("ok", bx)
