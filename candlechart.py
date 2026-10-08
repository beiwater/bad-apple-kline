"""把一帧黑白图像渲染成 K 线图（v2）。

核心规则：图像只决定每根蜡烛的「最高价 / 最低价」，「开盘 / 收盘」交给背景行情。
  - 固定 N 列等宽等距网格，每列就是一个交易日。
  - 人物（前景）在每列里的连续区间 = 一根蜡烛的 high..low，实体只占其中一段，
    位置和涨跌由行情决定，其余是影线。轮廓由影线尖端暗示，而不是硬边。
  - 留白处放同一尺度的背景行情蜡烛，和人物蜡烛是同一种东西。
  - 前景极性自动判断（带回差）：画面大面积变黑时，白色人物变成蜡烛。
  - 被前景包围的小块留白（眼睛之类）画成白色短横，借用纽约客的五官手法。
  - 白线 MA5、橙线 MA20 跟随背景行情，不描人物；黄线是画面顶部的独立指标。
"""
import numpy as np
from PIL import Image, ImageDraw

W, H = 1440, 1080          # 输出尺寸（4:3，与 Bad Apple 原片一致，满幅出血）
SS = 2                     # 超采样倍数，用于抗锯齿

PALETTE = {
    "bg":     (30, 90, 166),
    "grid":   (46, 104, 178),
    "up":     (46, 178, 122),
    "down":   (226, 69, 124),
    "vol":    (25, 179, 230),
    "ma5":    (250, 250, 250),
    "ma20":   (222, 104, 58),
    "band":   (232, 232, 72),
    "eye":    (250, 250, 250),
}

BODY_T = 0.5               # 列覆盖率阈值
FIG_BODY = (0.55, 0.92)    # 人物蜡烛实体占区间的比例范围（越大人物越实，越小越像行情）
WICK_W = 0.2               # 影线宽度 / 列距
HALO = 0.07                # 人物周围给背景行情留出的空白（占画面高度）
WICK_T = 0.15              # 部分覆盖（尖端、斜边）-> 影线，形成「插针」
BULL = 0.82                # 正常画面里人物蜡烛的上涨比例；反色画面翻转成熊市（1 - BULL）
LINE_GAP = 0.012           # 均线在人物上下沿外再让出的距离（占画面高度）
MIN_HIDE = 0.05            # 高于此值（占画面高度）的人物蜡烛才挖线
MIN_RUN = 3                # 可见线段至少几列，否则丢掉
UI = True                  # 左上角行情抬头
BEATS_PATH = "src/beats.npz"
INV_ON, INV_OFF = 0.58, 0.42   # 极性回差：黑占比超过 INV_ON 反转，低于 INV_OFF 恢复


def _runs(mask):
    """返回布尔数组中连续 True 区间 [(start, end_exclusive), ...]。"""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


class Market:
    """背景行情：均值回归的随机游走。每 step 帧向左滚动一根，像实时行情。"""

    def __init__(self, length=6000, seed=7):
        rng = np.random.default_rng(seed)

        def walk(sigma, pull, start=0.5):
            p = np.empty(length + 1)
            p[0] = start
            for k in range(length):
                p[k + 1] = p[k] + rng.normal(0, sigma) + pull * (0.5 - p[k])
            return p

        p = walk(0.045, 0.03)
        self.o, self.c = p[:-1], p[1:]
        span = np.abs(self.c - self.o)
        self.hi = np.maximum(self.o, self.c) + rng.uniform(0.01, 0.06, length) + 0.3 * span
        self.lo = np.minimum(self.o, self.c) - rng.uniform(0.01, 0.06, length) - 0.3 * span
        self.vol = rng.uniform(0.04, 0.14, length)
        # 人物蜡烛的实体：中心在区间内的位置（平滑游走，相邻列连成一条走势）、长度占比
        self.body_c = np.clip(walk(0.06, 0.08)[:-1], 0.15, 0.85)
        self.body_u = rng.uniform(0, 1, length)
        self.coin = rng.uniform(0, 1, length)
        self.osc = np.clip(walk(0.05, 0.05)[:-1], 0, 1)     # 黄线指标
        self.up = self.c > self.o

    def window(self, n, t=0, step=8):
        return (t // step) % (len(self.o) - n - 64)


MARKET = Market()

_BEATS = None


def beat(t):
    """返回第 t 帧的 (accent, big, energy)，都在 0..1；没有节拍文件时返回 0。"""
    global _BEATS
    if _BEATS is None:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BEATS_PATH)
        _BEATS = dict(np.load(path)) if os.path.exists(path) else {}
    if not _BEATS or t >= len(_BEATS["accent"]):
        return 0.0, 0.0, 0.0
    return float(_BEATS["accent"][t]), float(_BEATS["big"][t]), float(_BEATS["energy"][t])


def _price_y(a):
    """价格 0..1 -> 屏幕 y（价格高 = y 小）。"""
    return (0.88 - 0.76 * np.clip(a, 0, 1)) * H


def _holes(fg):
    """被前景包围的小块背景（眼睛、纽扣之类）。fg: bool 数组。返回 [(x0,y0,x1,y1), ...]，源坐标。"""
    h, w = fg.shape
    im = Image.fromarray(np.pad(~fg, 1, constant_values=True).astype(np.uint8) * 255)
    ImageDraw.floodfill(im, (0, 0), 128)
    a = np.array(im)[1:-1, 1:-1]
    out = []
    max_area = 0.004 * h * w
    for _ in range(12):
        ys, xs = np.nonzero(a == 255)
        if len(ys) == 0:
            break
        seed = (int(xs[0]) + 1, int(ys[0]) + 1)
        tmp = Image.fromarray(np.pad(a, 1, constant_values=128))
        ImageDraw.floodfill(tmp, seed, 64)
        comp = np.array(tmp)[1:-1, 1:-1] == 64
        a[comp] = 0
        cys, cxs = np.nonzero(comp)
        if 4 <= len(cys) <= max_area:
            out.append((cxs.min(), cys.min(), cxs.max() + 1, cys.max() + 1))
    return out


def analyze(gray, inv_prev=None):
    """gray: HxW uint8，0=黑。返回 (前景覆盖率图, 是否反转)。"""
    dark = 1.0 - gray.astype(np.float32) / 255.0
    ratio = dark.mean()
    if inv_prev is None:
        inv = ratio > 0.5
    else:
        inv = ratio > INV_ON if not inv_prev else ratio > INV_OFF
    fg = 1.0 - dark if inv else dark
    return fg, inv


def _column_profiles(fg, n_cols):
    h, w = fg.shape
    edges = np.linspace(0, w, n_cols + 1)
    cols = np.stack([fg[:, int(round(a)):max(int(round(b)), int(round(a)) + 1)].mean(axis=1)
                     for a, b in zip(edges[:-1], edges[1:])])          # n_cols x h
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    return np.stack([np.interp(ys, np.arange(h), c) for c in cols])    # n_cols x H


def build(fg, inv, n_cols, prev=None, ema=0.0, t=0, ambient=True):
    profile = _column_profiles(fg, n_cols)
    n = n_cols
    pitch = W / n
    M = MARKET
    k0 = M.window(n, t)
    min_body = 0.06 * H
    wick_ext = 0.22 * pitch

    candles = []      # (col, wick_top, wick_bot, body_top, body_bot, up, 是否人物)
    occupied = []     # 每列被人物占用的区间（含留白边距），给背景行情避让
    for i in range(n):
        p = profile[i]
        runs = [(a, b) for a, b in _runs(p >= BODY_T) if b - a >= 3]
        pins = _runs(p >= WICK_T)
        bull = BULL if not inv else 1 - BULL
        occ = []
        # 没有实体的细线（刀刃、发丝）：只画插针 + 一个小十字星
        for ws, we in pins:
            if we - ws >= 3 and not any(a < we and ws < b for a, b in runs):
                c = ws + int(np.argmax(p[ws:we]))
                candles.append((i, ws, we, c - 3, c + 3, bool(M.coin[k0 + i + 7] < bull), True))
                occ.append((ws - HALO * H, we + HALO * H))
        for k, (a, b) in enumerate(runs):
            j = k0 + i + 23 * k
            L = b - a
            up = bool(M.coin[j] < bull)
            ws, we = next(((x, y) for x, y in pins if x <= a and b <= y), (a, b))
            if L < min_body:            # 短区间：整段是实体，上下补一点影线
                bt, bb = a, b
                wt, wb = min(ws, max(0, a - wick_ext)), max(we, min(H, b + wick_ext))
            else:                       # 长区间：high/low 来自图像，实体位置来自行情
                bl = max(min_body, (FIG_BODY[0] + (FIG_BODY[1] - FIG_BODY[0]) * M.body_u[j]) * L)
                c = a + M.body_c[j] * L
                bt = int(np.clip(c - bl / 2, a, b - bl))
                bb = int(bt + bl)
                wt, wb = ws, we
            candles.append((i, wt, wb, bt, bb, up, True))
            occ.append((wt - HALO * H, wb + HALO * H))
        occupied.append(occ)

    # 背景行情：同一尺度，只放在留白里
    mo = _price_y(M.o[k0:k0 + n])
    mc = _price_y(M.c[k0:k0 + n])
    mh = _price_y(M.hi[k0:k0 + n])
    ml = _price_y(M.lo[k0:k0 + n])
    for i in range(n):
        bt, bb = sorted((mo[i], mc[i]))
        bb = max(bb, bt + 6)
        wt, wb = mh[i], ml[i]
        if ambient and not any(a < wb and wt < b for a, b in occupied[i]):
            candles.append((i, wt, wb, bt, bb, bool(M.up[k0 + i]), False))
    price = (mo + mc) / 2

    # 眼睛：被前景包围的小块留白 -> 白色短横
    eyes = []
    h, w = fg.shape
    for x0, y0, x1, y1 in _holes(fg >= BODY_T):
        cx, cy = (x0 + x1) / 2 * W / w, (y0 + y1) / 2 * H / h
        hw = max((x1 - x0) * W / w, 0.7 * pitch) / 2
        hh = max((y1 - y0) * H / h, 6) / 2
        hh = min(hh, 0.5 * pitch)
        eyes.append((cx - hw, cy - hh, cx + hw, cy + hh))

    ma5 = _smooth(price, 5)
    ma20 = _smooth(price, 17)
    osc = (0.05 + 0.16 * (1 - M.osc[k0:k0 + n])) * H
    vol = np.maximum(profile.mean(axis=1), M.vol[k0:k0 + n]) if ambient else profile.mean(axis=1)
    lines = np.stack([ma5, ma20, _smooth(osc, 3)])
    if prev is not None and ema > 0 and prev["n"] == n:
        lines = ema * prev["lines"] + (1 - ema) * lines
        vol = ema * prev["vol"] + (1 - ema) * vol
    accent, big, energy = beat(t)
    gain = 0.45 + 0.9 * accent + 1.4 * big
    px = lambda a: 60 + 90 * float(a)
    last, first = px(M.c[k0 + n - 1]) + 8 * energy, px(M.o[k0])
    tick = {"price": last, "chg": 100 * (last / first - 1), "big": big}
    return {"n": n, "candles": candles, "lines": lines, "vol": vol, "gain": gain,
            "eyes": eyes, "inv": inv, "tick": tick}


_BG_CACHE = {}


def _background(n):
    if n not in _BG_CACHE:
        im = Image.new("RGB", (W * SS, H * SS), PALETTE["bg"])
        d = ImageDraw.Draw(im)
        pitch = W / n
        r = 1.3 * SS
        for gx in np.arange(pitch * 0.5, W, pitch * 3):
            for gy in np.arange(12, H, 14):
                cx, cy = gx * SS, gy * SS
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PALETTE["grid"])
        _BG_CACHE[n] = im
    return _BG_CACHE[n].copy()


def draw(scene):
    n = scene["n"]
    pitch = W / n
    s = SS
    im = _background(n)
    d = ImageDraw.Draw(im)

    vw = pitch * 0.74
    gain = scene.get("gain", 1.0)
    for i, v in enumerate(scene["vol"]):
        h = min(v * H * 0.18 * gain, 0.42 * H)
        if h < 1:
            continue
        cx = (i + 0.5) * pitch
        d.rectangle([(cx - vw / 2) * s, (H - h) * s, (cx + vw / 2) * s, H * s], fill=PALETTE["vol"])

    # 均线：独立图层，人物所在的整列区域（含列间缝隙）挖空，线碰到人物就隐入、从另一侧浮出。
    # 小插针（高度 < MIN_HIDE）不挖线；挖完后可见长度不足 MIN_RUN 列的碎线头整段丢掉，避免闪烁。
    pad = LINE_GAP * H
    spans = [[] for _ in range(n)]
    for i, wt, wb, bt, bb, up, fig in scene["candles"]:
        if fig and wb - wt >= MIN_HIDE * H:
            spans[i].append((wt - pad, wb + pad))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    xs = (np.arange(n) + 0.5) * pitch
    for row, key, lw in [(1, "ma20", 5.0), (2, "band", 4.5), (0, "ma5", 6.0)]:
        ys = scene["lines"][row]
        vis = np.array([not any(a <= y <= b for a, b in spans[i]) for i, y in enumerate(ys)])
        for a, b in _runs(vis):
            if b - a < MIN_RUN and not (a == 0 or b == n):
                continue
            px = np.concatenate([[xs[a] - pitch], xs[a:b], [xs[b - 1] + pitch]])
            py = np.concatenate([[ys[max(a - 1, 0)]], ys[a:b], [ys[min(b, n - 1)]]])
            pts = [(float(x * s), float(y * s)) for x, y in zip(px, py)]
            ld.line(pts, fill=PALETTE[key] + (255,), width=int(lw * s), joint="curve")
    hole = Image.new("L", im.size, 0)
    hd = ImageDraw.Draw(hole)
    for i, sp in enumerate(spans):
        for a, b in sp:
            hd.rectangle([i * pitch * s - 1, a * s, (i + 1) * pitch * s + 1, b * s], fill=255)
    alpha = np.array(layer.getchannel("A"), dtype=np.uint16) * (255 - np.array(hole, dtype=np.uint16)) // 255
    layer.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    im.paste(layer, (0, 0), layer)

    bw = pitch * 0.58
    ww = max(2.5, pitch * WICK_W)
    for i, wt, wb, bt, bb, up, _ in scene["candles"]:
        col = PALETTE["up"] if up else PALETTE["down"]
        cx = (i + 0.5) * pitch
        d.rectangle([(cx - ww / 2) * s, wt * s, (cx + ww / 2) * s, wb * s], fill=col)
        d.rectangle([(cx - bw / 2) * s, bt * s, (cx + bw / 2) * s, bb * s], fill=col)

    for x0, y0, x1, y1 in scene["eyes"]:
        d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=PALETTE["eye"])

    if UI and "tick" in scene:
        _header(d, scene["tick"])

    return im.resize((W, H), Image.LANCZOS)


_FONTS = {}


def _font(size):
    if size not in _FONTS:
        from PIL import ImageFont
        _FONTS[size] = ImageFont.truetype("/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf", size)
    return _FONTS[size]


def _header(d, tick):
    """左上角行情抬头：代码、最新价、涨跌幅。"""
    s = SS
    x, y = 28 * s, 22 * s
    chg = tick["chg"]
    up = chg >= 0
    col = PALETTE["up"] if up else PALETTE["down"]
    f1, f2, f3 = _font(26 * s), _font(40 * s), _font(26 * s)
    items = [("BADAPPLE / 1D", f1, (250, 250, 250)),
             (f"{tick['price']:.2f}", f2, (250, 250, 250)),
             (f"     {chg:+.2f}%", f3, col)]
    widths = [d.textlength(t, font=f) for t, f, _ in items]
    pad = 14 * s
    total = sum(widths) + pad * 2 * len(items)
    d.rectangle([x - pad, y - 6 * s, x + total - pad, y + 52 * s], fill=PALETTE["bg"])
    cx = x
    for k, ((t, f, c), w) in enumerate(zip(items, widths)):
        d.text((cx, y + (6 * s if f is not f2 else 0)), t, font=f, fill=c)
        if k == 2:   # 涨跌三角（字体里没有 ▲▼，手画）
            tx, ty, r = cx + 10 * s, y + 20 * s, 9 * s
            tri = [(tx - r, ty + r * 0.6), (tx + r, ty + r * 0.6), (tx, ty - r)] if up else \
                  [(tx - r, ty - r * 0.6), (tx + r, ty - r * 0.6), (tx, ty + r)]
            d.polygon(tri, fill=c)
        cx += w + pad * 2


def render(gray, n_cols=60, prev=None, ema=0.0, t=0, ambient=True):
    fg, inv = analyze(gray, prev["inv"] if prev else None)
    scene = build(fg, inv, n_cols, prev, ema, t, ambient)
    return draw(scene), scene


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    g = np.array(Image.open(src).convert("L"))
    render(g, n)[0].save(dst)
