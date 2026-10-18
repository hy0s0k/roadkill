"""
고라니 개체 기반 시뮬레이션: 생태통로 + 유도 울타리 배치별 도로 횡단 비교
실행: python roadkill_sim.py  (numpy, scipy 필요)
단위: m, 시간 간격 10분
도로: y = 0 (폭 20 m), 울타리: y = ±15 m, 생태통로: x = -15 ~ 15 m (폭 30 m)
"""
import numpy as np

ROAD_HALF = 10.0
FENCE_Y = 15.0
ECO_HALF = 15.0


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


def simulate(segs, n=1000, days=30, seed=0, step_mean=30.0, turn_sd=0.6, beta=0.063,
             follow_mean=300.0, x_range=4500.0, y_min=50.0, y_max=1200.0, window=2000.0,
             record=None, bin_w=50.0):
    """follow_mean: 울타리를 만났을 때 울타리를 따라 걷는 거리의 평균(m, 지수분포)."""
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 6)
    cx = rng.uniform(-x_range, x_range, n)
    side = rng.choice([-1.0, 1.0], n)
    cy = side * rng.uniform(y_min, y_max, n)
    pos = np.stack([cx, cy], axis=1) + rng.normal(0, 50, (n, 2))
    pos[:, 1] = np.where(np.abs(pos[:, 1]) < FENCE_Y + 5, side * (FENCE_Y + 5), pos[:, 1])
    heading = rng.uniform(-np.pi, np.pi, n)
    sigma_scale = 400.0
    nseg = len(segs)
    if nseg:
        A = segs[:, 0]; B = segs[:, 1]
        seglen = np.linalg.norm(B - A, axis=1)
        F = (B - A) / seglen[:, None]
        NRM = np.stack([-F[:, 1], F[:, 0]], 1)

    following = np.zeros(n, bool)
    fseg = np.full(n, -1)
    fdir = np.zeros((n, 2))
    remain = np.zeros(n)

    edges = np.arange(-window, window + bin_w, bin_w)
    hist_unsafe = np.zeros(len(edges) - 1)
    unsafe = 0
    safe = 0
    traj = [] if record is not None else None

    for t in range(steps):
        L = rng.gamma(2.0, step_mean / 2.0, n)
        d = np.zeros((n, 2))
        free = ~following
        # --- 자유 이동: 상관 무작위 보행 + 행동권 중심 쪽 약한 끌림
        heading[free] = heading[free] + rng.normal(0, turn_sd, free.sum())
        ux = np.cos(heading[free]) + beta * (cx[free] - pos[free, 0]) / sigma_scale
        uy = np.sin(heading[free]) + beta * (cy[free] - pos[free, 1]) / sigma_scale
        heading[free] = np.arctan2(uy, ux)
        d[free, 0] = np.cos(heading[free]) * L[free]
        d[free, 1] = np.sin(heading[free]) * L[free]
        newpos = pos + d

        if nseg:
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
                if follow_mean > 0:
                    remain[ii] = rng.exponential(follow_mean, len(ii))
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
                    nd = np.where((toB >= toA)[:, None], F[kk], -F[kk])
                    fdir[ii] = nd
                    fseg[ii] = kk
                    remain[ii] -= step[h2] * th
                jj = idx[~h2]
                if len(jj):
                    cand = pos[jj] + dd[~h2]
                    ks = fseg[jj]
                    u = np.sum((cand - A[ks]) * F[ks], axis=1) / seglen[ks]
                    newpos[jj] = cand
                    remain[jj] -= step[~h2]
                    passed = (u > 1.0) | (u < 0.0)
                    rel = jj[passed | (remain[jj] <= 0)]
                    following[rel] = False
                    heading[rel] = np.arctan2(fdir[rel, 1], fdir[rel, 0])
                    d_rel = newpos[rel] - pos[rel]
                d = newpos - pos
                # 따라 걷는 이동이 다시 다른 울타리를 넘지 않도록 확인
                k3, _ = first_hit(pos[idx], d[idx], segs, exclude=fseg[idx])
                bad = idx[k3 >= 0]
                newpos[bad] = pos[bad]
            d = newpos - pos

        crossed = np.sign(pos[:, 1]) != np.sign(newpos[:, 1])
        if crossed.any():
            y0, y1 = pos[crossed, 1], newpos[crossed, 1]
            xc = pos[crossed, 0] + d[crossed, 0] * (y0 / (y0 - y1))
            inwin = np.abs(xc) <= window
            is_safe = np.abs(xc) <= ECO_HALF
            safe += int(np.sum(is_safe & inwin))
            us = (~is_safe) & inwin
            unsafe += int(np.sum(us))
            hist_unsafe += np.histogram(xc[us], bins=edges)[0]
        pos = newpos
        if record is not None:
            traj.append(pos[record].copy())

    out = dict(unsafe=unsafe, safe=safe, hist=hist_unsafe, edges=edges)
    if record is not None:
        out["traj"] = np.array(traj)
        out["centers"] = np.stack([cx, cy], 1)[record]
    return out


def home_range_check(n=200, days=30, seed=1, **kw):
    """울타리·도로 영향 없이 30일 95% MCP 면적(km^2) 추정."""
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    steps = int(days * 24 * 6)
    step_mean = kw.get("step_mean", 30.0)
    turn_sd = kw.get("turn_sd", 0.6)
    beta = kw.get("beta", 0.08)
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
    daily = []
    for i in range(n):
        p = track[::6, i]  # 1시간 간격 위치(텔레메트리 측정 간격 가정)
        c = p.mean(0)
        dist = np.linalg.norm(p - c, axis=1)
        keep = p[dist <= np.quantile(dist, 0.95)]
        areas.append(ConvexHull(keep).volume / 1e6)
    full = track
    step_len = np.linalg.norm(np.diff(full, axis=0), axis=2)
    daily_km = step_len.sum(0) / days / 1000
    return float(np.mean(areas)), float(np.std(areas)), float(np.mean(daily_km))


if __name__ == "__main__":
    # 예시: 울타리 없음 / 직선형 1 km / 끝단 회귀형(r=10%, θ=135°)을 난수 조건 3가지로 비교
    # (계획서 결과는 같은 모델을 배치마다 10회 반복한 값)
    print("행동권 크기 확인(30일 95% MCP, km^2):", round(home_range_check(beta=0.063, n=200)[0], 2))
    layouts = {
        "울타리 없음": build_fences(0),
        "직선형 1 km": build_fences(1000),
        "끝단 회귀형 1 km (r=10%, θ=135°)": build_fences(1000, 0.1, 135),
    }
    base = None
    for name, segs in layouts.items():
        u, s = [], []
        for seed in range(3):
            o = simulate(segs, seed=seed)
            u.append(o["unsafe"]); s.append(o["safe"])
        u = np.array(u, float); s = np.array(s, float)
        if base is None:
            base = (u, s)
        print(f"{name:32s} 위험 횡단 {u.mean():8.0f}  감소율 {100*np.mean(1-u/base[0]):5.1f}%  "
              f"생태통로 이용 {s.mean():5.0f} (울타리 없음 대비 {np.mean(s/base[1]):.2f}배)")
