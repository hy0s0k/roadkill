"""
고라니 개체 기반 시뮬레이션: 생태통로 + 유도 울타리 배치별 도로 횡단 비교 (시각화 포함)

준비:  py -m pip install numpy matplotlib
실행:  py roadkill_sim_vis.py              -> ① 움직임 애니메이션 창, 창을 닫으면 ② 결과 그래프
       py roadkill_sim_vis.py --mode anim   -> 애니메이션만
       py roadkill_sim_vis.py --mode summary-> 결과 그래프만
       py roadkill_sim_vis.py --gif 로드킬.gif -> 애니메이션을 GIF 파일로 저장(발표용)

단위: m, 시간 간격 10분
도로: y = 0 (폭 20 m), 울타리: y = ±15 m, 생태통로: x = -15 ~ 15 m (폭 30 m)
"""
import argparse
import numpy as np

ROAD_HALF = 10.0
FENCE_Y = 15.0
ECO_HALF = 15.0


# ---------------------------------------------------------------------------
# 1. 울타리 배치
# ---------------------------------------------------------------------------
def build_fences(L_side, r=0.0, theta_deg=90.0, walls=True):
    """L_side: 도로 한쪽 면에 쓰는 울타리 총 길이(생태통로 좌우 합).
    r: 그중 끝단을 꺾는 데 쓰는 비율, theta: 꺾는 각도(도로와 나란한 바깥 방향 기준)."""
    segs = []
    if walls:  # 생태통로 양쪽 벽(통로 구조물의 일부)
        for q in (1, -1):
            segs.append(((q * ECO_HALF, -FENCE_Y), (q * ECO_HALF, FENCE_Y)))
    if L_side <= 0:
        return np.array(segs, dtype=float)
    half = L_side / 2.0
    Ls = (1 - r) * half
    Lb = r * half
    th = np.deg2rad(theta_deg)
    for s in (1, -1):          # 도로 위/아래 면
        for q in (1, -1):      # 통로 오른쪽/왼쪽
            a = (q * ECO_HALF, s * FENCE_Y)
            b = (q * (ECO_HALF + Ls), s * FENCE_Y)
            segs.append((a, b))
            if Lb > 0:
                c = (b[0] + q * Lb * np.cos(th), b[1] + s * Lb * np.sin(th))
                segs.append((b, c))
    return np.array(segs, dtype=float)


def cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def first_hit(p0, d, segs, exclude=None):
    """각 개체의 이동(p0 -> p0+d)이 처음 만나는 울타리 번호(-1: 없음)와 그 비율 t."""
    n = p0.shape[0]
    best_t = np.full(n, np.inf)
    best_k = np.full(n, -1, dtype=int)
    for k, (A, B) in enumerate(segs):
        sx, sy = B[0] - A[0], B[1] - A[1]
        denom = cross2(d[:, 0], d[:, 1], sx, sy)
        apx, apy = A[0] - p0[:, 0], A[1] - p0[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross2(apx, apy, sx, sy) / denom
            u = cross2(apx, apy, d[:, 0], d[:, 1]) / denom
        hit = (np.abs(denom) > 1e-12) & (t > 1e-9) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        if exclude is not None:
            hit &= exclude != k
        upd = hit & (t < best_t)
        best_t[upd] = t[upd]
        best_k[upd] = k
    return best_k, best_t


# ---------------------------------------------------------------------------
# 2. 시뮬레이션 (한 걸음씩 진행할 수 있게 클래스로 구성)
# ---------------------------------------------------------------------------
class RoadkillSim:
    """follow_mean: 울타리를 만났을 때 울타리를 따라 걷는 거리의 평균(m, 지수분포)."""

    def __init__(self, segs, n=1000, seed=0, step_mean=30.0, turn_sd=0.6, beta=0.063,
                 follow_mean=300.0, x_range=4500.0, y_min=50.0, y_max=1200.0,
                 window=2000.0, bin_w=50.0):
        self.segs = segs
        self.n = n
        self.step_mean, self.turn_sd, self.beta = step_mean, turn_sd, beta
        self.follow_mean = follow_mean
        self.window = window
        self.rng = rng = np.random.default_rng(seed)
        self.cx = rng.uniform(-x_range, x_range, n)
        side = rng.choice([-1.0, 1.0], n)
        self.cy = side * rng.uniform(y_min, y_max, n)
        pos = np.stack([self.cx, self.cy], axis=1) + rng.normal(0, 50, (n, 2))
        pos[:, 1] = np.where(np.abs(pos[:, 1]) < FENCE_Y + 5, side * (FENCE_Y + 5), pos[:, 1])
        self.pos = pos
        self.heading = rng.uniform(-np.pi, np.pi, n)
        self.sigma_scale = 400.0
        if len(segs):
            self.A = segs[:, 0]; self.B = segs[:, 1]
            self.seglen = np.linalg.norm(self.B - self.A, axis=1)
            self.F = (self.B - self.A) / self.seglen[:, None]
            self.NRM = np.stack([-self.F[:, 1], self.F[:, 0]], 1)
        self.following = np.zeros(n, bool)
        self.fseg = np.full(n, -1)
        self.fdir = np.zeros((n, 2))
        self.remain = np.zeros(n)
        self.edges = np.arange(-window, window + bin_w, bin_w)
        self.hist_unsafe = np.zeros(len(self.edges) - 1)
        self.unsafe = 0
        self.safe = 0
        self.t = 0

    def step(self):
        """10분 진행. 이번 걸음에 생긴 (위험 횡단 x좌표 배열, 생태통로 횡단 x좌표 배열)을 돌려준다."""
        n, rng, segs = self.n, self.rng, self.segs
        pos, heading, cx, cy = self.pos, self.heading, self.cx, self.cy
        following, fseg, fdir, remain = self.following, self.fseg, self.fdir, self.remain
        L = rng.gamma(2.0, self.step_mean / 2.0, n)
        d = np.zeros((n, 2))
        free = ~following
        # --- 자유 이동: 상관 무작위 보행 + 행동권 중심 쪽 약한 끌림
        heading[free] = heading[free] + rng.normal(0, self.turn_sd, free.sum())
        ux = np.cos(heading[free]) + self.beta * (cx[free] - pos[free, 0]) / self.sigma_scale
        uy = np.sin(heading[free]) + self.beta * (cy[free] - pos[free, 1]) / self.sigma_scale
        heading[free] = np.arctan2(uy, ux)
        d[free, 0] = np.cos(heading[free]) * L[free]
        d[free, 1] = np.sin(heading[free]) * L[free]
        newpos = pos + d

        if len(segs):
            A, B, F, NRM, seglen = self.A, self.B, self.F, self.NRM, self.seglen
            # (1) 자유 이동 중 울타리에 부딪힘 -> 따라 걷기 시작
            idx = np.where(free)[0]
            k, tt = first_hit(pos[idx], d[idx], segs)
            h = k >= 0
            if h.any():
                ii = idx[h]; kk = k[h]; th = tt[h]
                P = pos[ii] + d[ii] * th[:, None]
                sgn = np.sign(np.sum((pos[ii] - A[kk]) * NRM[kk], axis=1))
                sgn[sgn == 0] = 1
                newpos[ii] = P + NRM[kk] * (0.5 * sgn)[:, None]
                dot = np.sum(d[ii] * F[kk], axis=1)
                s = np.where(dot >= 0, 1.0, -1.0)
                fdir[ii] = F[kk] * s[:, None]
                fseg[ii] = kk
                if self.follow_mean > 0:
                    remain[ii] = rng.exponential(self.follow_mean, len(ii))
                    following[ii] = True
                else:
                    heading[ii] = heading[ii] + np.pi + rng.normal(0, 0.5, len(ii))
            # (2) 울타리를 따라 걷는 중
            idx = np.where(following & ~np.isin(np.arange(n), idx[h] if h.any() else []))[0]
            if len(idx):
                step = np.minimum(L[idx], remain[idx])
                dd = fdir[idx] * step[:, None]
                k, tt = first_hit(pos[idx], dd, segs, exclude=fseg[idx])
                h2 = k >= 0
                # 다른 울타리(꺾인 끝단 등)와 만나면 그 울타리로 갈아타 계속 따라감
                if h2.any():
                    ii = idx[h2]; kk = k[h2]; th = tt[h2]
                    P = pos[ii] + dd[h2] * th[:, None]
                    sgn = np.sign(np.sum((pos[ii] - A[kk]) * NRM[kk], axis=1))
                    sgn[sgn == 0] = 1
                    newpos[ii] = P - fdir[ii] * 0.5 + NRM[kk] * (0.5 * sgn)[:, None]
                    toA = np.linalg.norm(A[kk] - P, axis=1); toB = np.linalg.norm(B[kk] - P, axis=1)
                    fdir[ii] = np.where((toB >= toA)[:, None], F[kk], -F[kk])
                    fseg[ii] = kk
                    remain[ii] -= step[h2] * th
                jj = idx[~h2]
                if len(jj):
                    cand = pos[jj] + dd[~h2]
                    ks = fseg[jj]
                    u = np.sum((cand - A[ks]) * F[ks], axis=1) / seglen[ks]
                    newpos[jj] = cand
                    remain[jj] -= step[~h2]
                    passed = (u > 1.0) | (u < 0.0)       # 울타리 끝을 지나침
                    rel = jj[passed | (remain[jj] <= 0)]  # 따라 걷기 끝
                    following[rel] = False
                    heading[rel] = np.arctan2(fdir[rel, 1], fdir[rel, 0])
                d = newpos - pos
                # 따라 걷는 이동이 다시 다른 울타리를 넘지 않도록 확인
                k3, _ = first_hit(pos[idx], d[idx], segs, exclude=fseg[idx])
                bad = idx[k3 >= 0]
                newpos[bad] = pos[bad]
            d = newpos - pos

        # --- 도로 중앙선(y = 0)을 넘은 개체 = 도로 횡단
        unsafe_x = np.empty(0); safe_x = np.empty(0)
        crossed = np.sign(pos[:, 1]) != np.sign(newpos[:, 1])
        if crossed.any():
            y0, y1 = pos[crossed, 1], newpos[crossed, 1]
            xc = pos[crossed, 0] + d[crossed, 0] * (y0 / (y0 - y1))
            inwin = np.abs(xc) <= self.window
            is_safe = np.abs(xc) <= ECO_HALF
            safe_x = xc[is_safe & inwin]
            unsafe_x = xc[(~is_safe) & inwin]
            self.safe += len(safe_x)
            self.unsafe += len(unsafe_x)
            self.hist_unsafe += np.histogram(unsafe_x, bins=self.edges)[0]
        self.pos = newpos
        self.t += 1
        return unsafe_x, safe_x


def simulate(segs, n=1000, days=30, seed=0, **kw):
    sim = RoadkillSim(segs, n=n, seed=seed, **kw)
    for _ in range(int(days * 24 * 6)):
        sim.step()
    return dict(unsafe=sim.unsafe, safe=sim.safe, hist=sim.hist_unsafe, edges=sim.edges)


# ---------------------------------------------------------------------------
# 3. 행동권 크기 확인(scipy 없이)
# ---------------------------------------------------------------------------
def hull_area(pts):
    """점들의 볼록 다각형(convex hull) 넓이(m^2). Andrew 단조 사슬 + 신발끈 공식."""
    P = sorted(map(tuple, pts))
    if len(P) < 3:
        return 0.0

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in P:
        while len(lower) >= 2 and cr(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(P):
        while len(upper) >= 2 and cr(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    h = np.array(lower[:-1] + upper[:-1])
    x, y = h[:, 0], h[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def home_range_check(n=200, days=30, seed=1, step_mean=30.0, turn_sd=0.6, beta=0.063):
    """울타리·도로 영향 없이 30일 95% MCP 면적(km^2) 추정."""
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 6)
    pos = np.zeros((n, 2))
    heading = rng.uniform(-np.pi, np.pi, n)
    track = np.zeros((steps, n, 2))
    for t in range(steps):
        heading = heading + rng.normal(0, turn_sd, n)
        ux = np.cos(heading) + beta * (0 - pos[:, 0]) / 400.0
        uy = np.sin(heading) + beta * (0 - pos[:, 1]) / 400.0
        heading = np.arctan2(uy, ux)
        L = rng.gamma(2.0, step_mean / 2.0, n)
        pos = pos + np.stack([np.cos(heading) * L, np.sin(heading) * L], 1)
        track[t] = pos
    areas = []
    for i in range(n):
        p = track[::6, i]  # 1시간 간격 위치(원격추적 측정 간격 가정)
        dist = np.linalg.norm(p - p.mean(0), axis=1)
        keep = p[dist <= np.quantile(dist, 0.95)]
        areas.append(hull_area(keep) / 1e6)
    return float(np.mean(areas))


# ---------------------------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------------------------
COL = dict(road="#cfcfcf", lane="#ffffff", eco="#7fb77e", fence="#1f5fa8", deer="#8a5a2b",
           follow="#f39c12", unsafe="#d62728", safe="#2ca02c", base="#6b6b6b")

LAYOUTS = {
    "울타리 없음": dict(L=0, r=0.0, th=90),
    "직선형 1 km": dict(L=1000, r=0.0, th=90),
    "끝단 회귀형 1 km (끝 10%, 135°)": dict(L=1000, r=0.1, th=135),
}


def set_korean_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False


def draw_scene(ax, segs, title, xlim=900, ylim=260):
    """도로, 생태통로, 울타리를 그린다(세로 방향은 확대해서 보여줌)."""
    from matplotlib.patches import Rectangle
    ax.axhspan(-ROAD_HALF, ROAD_HALF, color=COL["road"], zorder=0)
    ax.axhline(0, color=COL["lane"], lw=0.8, ls=(0, (6, 4)), zorder=1)
    ax.add_patch(Rectangle((-ECO_HALF, -22), 2 * ECO_HALF, 44, color=COL["eco"], zorder=2))
    for (a, b) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COL["fence"], lw=2.6, zorder=3,
                solid_capstyle="butt")
    ax.set_xlim(-xlim, xlim); ax.set_ylim(-ylim, ylim)
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(-xlim + 15, 14, "도로", fontsize=9, color="#555555", zorder=4)
    ax.text(22, 28, "생태통로", fontsize=9, color="#2f6b2f", zorder=4)
    ax.tick_params(labelbottom=False)


def animate(days=10, steps_per_frame=3, n=700, seed=0, gif=None):
    """직선형과 끝단 회귀형을 같은 난수 조건에서 나란히 움직여 보여준다.
    위: 고라니 위치(세로 확대), 아래: 지금까지의 위험 횡단을 도로 위치별로 센 막대그래프."""
    import matplotlib
    if gif:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    set_korean_font()

    XL = 900
    names = ["직선형 1 km", "끝단 회귀형 1 km (끝 10%, 135°)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex="col",
                             gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.06, wspace=0.07))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.14)
    sims, arts = [], []
    for col, name in enumerate(names):
        ax, hx = axes[0, col], axes[1, col]
        p = LAYOUTS[name]
        segs = build_fences(p["L"], p["r"], p["th"])
        # 화면 근처에 행동권 중심을 둔 고라니만 계산(보기 쉽게)
        sim = RoadkillSim(segs, n=n, seed=seed, x_range=1500, y_max=700, window=1500)
        sims.append(sim)
        draw_scene(ax, segs, name, xlim=XL)
        dots = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], s=10, c=COL["deer"], zorder=5, linewidths=0)
        bad = ax.scatter([], [], s=34, marker="x", c=COL["unsafe"], zorder=6, linewidths=1.4)
        good = ax.scatter([], [], s=60, marker="o", facecolors="none", edgecolors=COL["safe"],
                          zorder=6, linewidths=1.6)
        info = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb", alpha=0.92), zorder=7)
        # 아래: 위험 횡단 위치별 누적 막대
        e = sim.edges; xc = (e[:-1] + e[1:]) / 2
        m = np.abs(xc) <= XL
        bars = hx.bar(xc[m], np.zeros(m.sum()), width=(e[1] - e[0]) * 0.9, color=COL["unsafe"], alpha=0.8)
        L = p["L"] / 2 * (1 - p["r"]) + ECO_HALF
        hx.axvspan(-L, L, color=COL["fence"], alpha=0.08, zorder=0)
        hx.set_xlabel("생태통로로부터의 거리 (m)")
        if col == 0:
            ax.set_ylabel("도로로부터의 거리 (m)\n(세로 확대)")
            hx.set_ylabel("위험 횡단 누적\n(50 m 구간)")
        else:
            ax.tick_params(labelleft=False); hx.tick_params(labelleft=False)
        for sp in ("top", "right"):
            hx.spines[sp].set_visible(False)
        arts.append(dict(dots=dots, bad=bad, good=good, info=info, bars=bars, mask=m, hx=hx,
                         recent_b=[], recent_g=[]))
    clock = fig.text(0.5, 0.95, "", ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.915, f"고라니 {n}마리, 두 화면은 같은 난수 조건 · 빨간 ×와 초록 ○는 최근 12시간 동안의 도로 횡단",
             ha="center", fontsize=10, color="#555555")
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=5, color=COL["deer"], label="고라니"),
        Line2D([], [], ls="", marker="o", ms=5, color=COL["follow"], label="울타리를 따라 걷는 고라니"),
        Line2D([], [], ls="", marker="x", ms=7, color=COL["unsafe"], label="위험 횡단(통로 밖에서 도로를 건넘)"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="none", color=COL["safe"], label="생태통로 이용"),
        Line2D([], [], color=COL["fence"], lw=2.6, label="유도 울타리"),
    ], loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    frames = int(days * 24 * 6 / steps_per_frame)
    KEEP = 72  # 최근 12시간(72걸음) 동안의 횡단만 지도에 표시
    jitter = np.random.default_rng(1)
    deer_rgb = np.array([[0.541, 0.353, 0.169]]); follow_rgb = np.array([[0.953, 0.612, 0.071]])

    def update(f):
        for sim, a in zip(sims, arts):
            for _ in range(steps_per_frame):
                ux, sx = sim.step()
                for x in ux:
                    a["recent_b"].append((x, jitter.uniform(-7, 7), sim.t))
                for x in sx:
                    a["recent_g"].append((x, 0.0, sim.t))
            a["recent_b"] = [r for r in a["recent_b"] if sim.t - r[2] < KEEP]
            a["recent_g"] = [r for r in a["recent_g"] if sim.t - r[2] < KEEP]
            a["dots"].set_offsets(sim.pos)
            a["dots"].set_color(np.where(sim.following[:, None], follow_rgb, deer_rgb))
            a["bad"].set_offsets(np.array([r[:2] for r in a["recent_b"]]).reshape(-1, 2))
            a["good"].set_offsets(np.array([r[:2] for r in a["recent_g"]]).reshape(-1, 2))
            for b, hgt in zip(a["bars"], sim.hist_unsafe[a["mask"]]):
                b.set_height(hgt)
        top = max(sim.hist_unsafe[a["mask"]].max() for sim, a in zip(sims, arts))
        for sim, a in zip(sims, arts):
            a["hx"].set_ylim(0, max(5, top * 1.15))
        u0 = sims[0].unsafe
        for i, (sim, a) in enumerate(zip(sims, arts)):
            txt = f"위험 횡단  {sim.unsafe:5d}회\n생태통로 이용 {sim.safe:4d}회"
            if i == 1 and u0 > 0:
                txt += f"\n직선형 대비 위험 횡단 {100 * (sim.unsafe / u0 - 1):+.0f}%"
            a["info"].set_text(txt)
        t = sims[0].t * 10  # 분
        clock.set_text(f"{t // 1440 + 1}일째  {t % 1440 // 60:02d}:{t % 60:02d}")
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=False, repeat=False)
    if gif:
        print(f"GIF 저장 중... ({frames}장면, 몇 분 걸릴 수 있음)")
        ani.save(gif, writer=PillowWriter(fps=20), dpi=72)
        print("저장 완료:", gif)
        plt.close(fig)
    else:
        plt.show()
    return ani


def summary(reps=3, n=1000, days=30):
    """세 배치를 고라니 1000마리 x 30일로 돌려 결과 그래프를 그린다."""
    import matplotlib.pyplot as plt
    set_korean_font()
    res = {}
    for name, p in LAYOUTS.items():
        segs = build_fences(p["L"], p["r"], p["th"])
        u, s, h = [], [], []
        for seed in range(reps):
            print(f"  계산 중: {name}  ({seed + 1}/{reps})", flush=True)
            o = simulate(segs, n=n, days=days, seed=seed)
            u.append(o["unsafe"]); s.append(o["safe"]); h.append(o["hist"])
            edges = o["edges"]
        res[name] = dict(u=np.array(u, float), s=np.array(s, float), h=np.mean(h, 0))
    base = res["울타리 없음"]
    print("\n배치별 결과 (울타리 없음 대비)")
    for name, v in res.items():
        print(f"  {name:28s} 위험 횡단 감소율 {100 * np.mean(1 - v['u'] / base['u']):5.1f}%   "
              f"생태통로 이용 {np.mean(v['s'] / base['s']):.2f}배")

    xc = (edges[:-1] + edges[1:]) / 2
    pos = xc > 0
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    colors = [COL["base"], COL["fence"], COL["unsafe"]]
    for (name, v), c in zip(res.items(), colors):
        prof = v["h"][pos] + v["h"][~pos][::-1]   # 생태통로 양쪽을 합침
        a1.plot(xc[pos], np.convolve(prof, np.ones(2) / 2, mode="same"), color=c, lw=1.8, label=name)
    a1.axvspan(15, 515, color=COL["fence"], alpha=0.08)
    a1.text(265, 0.95, "직선형 울타리 구간", ha="center", fontsize=9, color=COL["fence"],
            transform=a1.get_xaxis_transform())
    a1.set_xlim(0, 1500); a1.set_ylim(bottom=0)
    a1.set_xlabel("생태통로 중심에서의 거리 (m)")
    a1.set_ylabel(f"위험 횡단 횟수 (50 m 구간, {reps}회 평균)")
    a1.set_title("도로를 따라 본 위험 횡단 분포", fontsize=12)
    a1.legend(frameon=False, fontsize=9)

    labels = ["울타리\n없음", "직선형\n1 km", "끝단 회귀형\n1 km"]
    eff = [100 * np.mean(1 - v["u"] / base["u"]) for v in res.values()]
    use = [np.mean(v["s"] / base["s"]) for v in res.values()]
    for ax, vals, title, fmt in [(a2, eff, "위험 횡단 감소율 (%)", "{:.1f}%"),
                                 (a3, use, "생태통로 이용 (울타리 없음 = 1)", "{:.2f}배")]:
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(vals) * 1.2 + 0.01)
    for ax in (a1, a2, a3):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="고라니 로드킬 유도 울타리 시뮬레이션")
    ap.add_argument("--mode", choices=["both", "anim", "summary"], default="both")
    ap.add_argument("--days", type=float, default=10, help="애니메이션으로 보여줄 기간(일)")
    ap.add_argument("--gif", default=None, help="애니메이션을 GIF로 저장할 파일 이름")
    ap.add_argument("--reps", type=int, default=3, help="결과 그래프의 반복 횟수")
    args = ap.parse_args()
    if args.gif:
        animate(days=args.days, gif=args.gif)
    else:
        if args.mode in ("both", "anim"):
            print("애니메이션 창이 열립니다. 창을 닫으면 다음 단계로 넘어갑니다.")
            animate(days=args.days)
        if args.mode in ("both", "summary"):
            print("결과 그래프 계산 중... (컴퓨터에 따라 1~3분)")
            summary(reps=args.reps)
