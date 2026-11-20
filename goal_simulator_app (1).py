import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="월 목표 수익 시뮬레이터", page_icon="🎲", layout="wide")

# ──────────────────────────────────────────────
# 선택 가능한 대상 (분류 → 이름 → 야후 파이낸스 티커)
# 추가하고 싶은 종목은 여기에 한 줄만 넣으면 선택 목록에 나타납니다.
# ──────────────────────────────────────────────
ASSETS = {
    "한국 지수": {
        "KOSPI (코스피)": "^KS11",
        "KOSDAQ (코스닥)": "^KQ11",
    },
    "해외 지수": {
        "NASDAQ (나스닥)": "^IXIC",
        "S&P 500": "^GSPC",
        "다우존스": "^DJI",
        "러셀 2000": "^RUT",
        "필라델피아 반도체": "^SOX",
        "닛케이 225": "^N225",
        "항셍": "^HSI",
        "상해종합": "000001.SS",
        "대만 가권": "^TWII",
        "영국 FTSE 100": "^FTSE",
        "독일 DAX": "^GDAXI",
        "프랑스 CAC 40": "^FCHI",
    },
    "한국 대표 종목": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "LG에너지솔루션": "373220.KS",
        "현대차": "005380.KS",
        "기아": "000270.KS",
        "셀트리온": "068270.KS",
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
        "에코프로": "086520.KQ",
        "에코프로비엠": "247540.KQ",
        "알테오젠": "196170.KQ",
    },
    "미국 대표 종목": {
        "애플 (AAPL)": "AAPL",
        "엔비디아 (NVDA)": "NVDA",
        "마이크로소프트 (MSFT)": "MSFT",
        "구글 (GOOGL)": "GOOGL",
        "아마존 (AMZN)": "AMZN",
        "메타 (META)": "META",
        "테슬라 (TSLA)": "TSLA",
        "팔란티어 (PLTR)": "PLTR",
    },
    "ETF": {
        "SPY (S&P 500)": "SPY",
        "QQQ (나스닥 100)": "QQQ",
        "SOXX (반도체)": "SOXX",
        "TQQQ (나스닥 3배)": "TQQQ",
    },
}

# ① 언제 살까? → 내부 값 (None = 조건 없이 바로, 0 = 상승장, 1 = 횡보, 2 = 하락장)
ENTRY_OPTIONS = {
    "바로 매수 (시작하자마자)": None,
    "상승장일 때 매수": 0,
    "횡보일 때 매수": 1,
    "하락장일 때 매수 (쌀 때 사기)": 2,
}

HORIZON = 21  # 한 달 = 21거래일
STATES = ["상승장", "횡보", "하락장"]
ICON = {"상승장": "🟢", "횡보": "🟡", "하락장": "🔴"}
STATE_COLOR = ["rgba(22,163,74,0.12)", "rgba(202,138,4,0.12)", "rgba(220,38,38,0.12)"]


# ──────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load(ticker: str, period: str = "10y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["Close"])


# ===== CORE START (numpy/pandas만 사용) =====
# ──────────────────────────────────────────────
# 추세 판단 (기존 앱과 같은 6점 기준)
# ──────────────────────────────────────────────
def trend_score_series(close: pd.Series) -> pd.Series:
    m20, m60, m120 = (close.rolling(n).mean() for n in (20, 60, 120))
    m60_prev = m60.shift(20)
    score = (
        (close > m20).astype(int) + (close > m60).astype(int) + (close > m120).astype(int)
        + (m20 > m60).astype(int) + (m60 > m120).astype(int) + (m60 > m60_prev).astype(int)
    )
    return score.where(m120.notna() & m60_prev.notna())


def state_index(score: float) -> int:
    """0 = 상승장(4~6점), 1 = 횡보(3점), 2 = 하락장(0~2점)"""
    if score >= 4:
        return 0
    if score == 3:
        return 1
    return 2


def build_market_model(close: pd.Series):
    """추세별 일간 수익률 풀 + 추세 전환 확률표 + 현재 추세를 만든다."""
    score = trend_score_series(close).dropna()
    states = score.apply(state_index).astype(int)
    next_ret = close.pct_change().shift(-1).reindex(states.index)

    all_pool = close.pct_change().dropna().values
    pools, sizes = [], []
    for s in range(3):
        p = next_ret[states == s].dropna().values
        sizes.append(len(p))
        pools.append(p if len(p) >= 100 else all_pool)  # 표본이 적으면 전체 사용

    arr = states.values
    counts = np.ones((3, 3))  # 라플라스 보정
    np.add.at(counts, (arr[:-1], arr[1:]), 1)
    trans = counts / counts.sum(axis=1, keepdims=True)

    return {
        "pools": pools,
        "all_pool": all_pool,
        "sizes": sizes,
        "trans": trans,
        "current": int(arr[-1]),
        "current_score": int(score.iloc[-1]),
    }


# ──────────────────────────────────────────────
# 시장 시뮬레이션 (일간 수익률 + 그날의 추세)
# ──────────────────────────────────────────────
def simulate_market(pools, trans, start, fixed, n, days, rng, vol_mult=1.0):
    """일간 수익률 행렬(n×days)과 그날 시작 시점의 추세 행렬(n×days)을 만든다."""
    state = np.full(n, start)
    cum = np.cumsum(trans, axis=1)
    rets = np.empty((n, days), dtype=np.float32)
    states = np.empty((n, days), dtype=np.int8)
    for d in range(days):
        states[:, d] = state
        for s in range(3):
            m = state == s
            k = int(m.sum())
            if k:
                pool = pools[s]
                r = pool[rng.integers(0, len(pool), k)]
                if vol_mult != 1.0:
                    mu = pool.mean()
                    r = mu + (r - mu) * vol_mult
                rets[m, d] = r
        if not fixed:
            u = rng.random(n)
            state = np.minimum((u[:, None] > cum[state]).sum(axis=1), 2)
    return np.clip(rets, -0.99, None), states


# ──────────────────────────────────────────────
# 매매 전략 (언제 살까 / 몇 % 오르면 팔까 / 몇 % 떨어지면 팔까)
# ──────────────────────────────────────────────
def run_strategy(rets, states, capital, entry, tp, sl, repeat):
    """
    entry : None(바로) / 0 상승장 / 1 횡보 / 2 하락장 일 때 전액 매수
    tp    : 매수가 대비 +tp(비율) 이상이면 매도 (None이면 사용 안 함)
    sl    : 매수가 대비 -sl(비율) 이하이면 매도 (None이면 사용 안 함)
    repeat: 매도 후에도 조건이 되면 다시 매수할지
    매수·매도는 하루 종가 기준이며, 현금으로 있는 동안 수익률은 0%로 봅니다.
    """
    n, days = rets.shape
    V = np.full(n, float(capital))
    ref = np.ones(n)
    in_pos = np.zeros(n, dtype=bool)
    can = np.ones(n, dtype=bool)
    ever = np.zeros(n, dtype=bool)
    n_tp = np.zeros(n)
    n_sl = np.zeros(n)
    days_in = np.zeros(n)
    paths = np.empty((n, days + 1), dtype=np.float32)
    paths[:, 0] = capital

    for d in range(days):
        want = (~in_pos) & can
        if entry is not None:
            want &= states[:, d] == entry
        in_pos |= want
        ever |= want
        ref = np.where(want, V, ref)
        days_in += in_pos
        V = V * (1 + rets[:, d] * in_pos)

        if tp is not None or sl is not None:
            g = V / ref - 1
            hit_tp = in_pos & (g >= tp) if tp is not None else np.zeros(n, dtype=bool)
            hit_sl = in_pos & (g <= -sl) & ~hit_tp if sl is not None else np.zeros(n, dtype=bool)
            sell = hit_tp | hit_sl
            n_tp += hit_tp
            n_sl += hit_sl
            in_pos &= ~sell
            if not repeat:
                can &= ~sell
        paths[:, d + 1] = V

    trades = n_tp + n_sl
    info = {
        "entered": float(ever.mean() * 100),
        "avg_trades": float(trades.mean()),
        "tp_share": float(n_tp.sum() / trades.sum() * 100) if trades.sum() > 0 else float("nan"),
        "invested_ratio": float(days_in.mean() / days * 100),
    }
    return paths, info


def trace_path(rets_row, states_row, entry, tp, sl, repeat):
    """경로 하나의 매수·매도 시점을 기록한다 (예시 차트용)."""
    price = 100 * np.cumprod(1 + rets_row.astype(np.float64))
    price = np.concatenate([[100.0], price])
    in_pos, can, ref, V = False, True, 1.0, 1.0
    buys, tps, sls = [], [], []
    for d in range(len(rets_row)):
        if (not in_pos) and can and (entry is None or states_row[d] == entry):
            in_pos, ref = True, V
            buys.append(d)
        if in_pos:
            V *= 1 + float(rets_row[d])
            g = V / ref - 1
            if tp is not None and g >= tp:
                tps.append(d + 1)
                in_pos = False
                can = repeat
            elif sl is not None and g <= -sl:
                sls.append(d + 1)
                in_pos = False
                can = repeat
    return price, buys, tps, sls


def to_paths(rets: np.ndarray, capital: float) -> np.ndarray:
    """그냥 사서 계속 들고 있을 때의 자산 경로 (비교 기준)."""
    n = rets.shape[0]
    growth = np.cumprod(1 + rets, axis=1, dtype=np.float64)
    return (np.concatenate([np.ones((n, 1)), growth], axis=1) * capital).astype(np.float32)


def evaluate(paths: np.ndarray, capital: float, target: float, months: int) -> dict:
    """경로 묶음에서 성공률·위험 지표를 뽑는다."""
    p = paths.astype(np.float64)
    month_end = p[:, ::HORIZON][:, : months + 1]
    profit = np.diff(month_end, axis=1)  # 월별 손익 (원)
    final = p[:, -1]
    cummax = np.maximum.accumulate(p, axis=1)
    mdd = (p / cummax - 1).min(axis=1)

    hit = profit >= target
    return {
        "monthly_success": float(hit.mean() * 100),
        "all_months": float(hit.all(axis=1).mean() * 100),
        "any_month": float(hit.any(axis=1).mean() * 100),
        "monthly_loss": float((profit < 0).mean() * 100),
        "cum_target": float((final >= capital + target * months).mean() * 100),
        "principal_loss": float((final < capital).mean() * 100),
        "final_p5": float(np.percentile(final, 5)),
        "final_p50": float(np.percentile(final, 50)),
        "final_p95": float(np.percentile(final, 95)),
        "mdd_worst5": float(np.percentile(mdd, 5) * 100),
        "mdd20": float((mdd <= -0.20).mean() * 100),
        "m1_p30": float(np.percentile(profit[:, 0], 30)),
        "m1_p50": float(np.percentile(profit[:, 0], 50)),
    }


def risk_grade(success: float, loss: float):
    if success >= 60 and loss <= 35:
        return "🟢 낮음", "#16a34a"
    if success >= 40:
        return "🟡 보통", "#ca8a04"
    if success >= 20:
        return "🟠 높음", "#ea580c"
    return "🔴 매우 높음", "#dc2626"


def run_all_scenarios(model, capital, target, months, n, seed, strat):
    rng = np.random.default_rng(seed)
    days = HORIZON * months
    cur = model["current"]
    pools, trans, allp = model["pools"], model["trans"], model["all_pool"]
    cur_name = STATES[cur]

    scenarios = [
        ("현재 추세 + 추세 전환 반영 (기본)", dict(pools=pools, start=cur, fixed=False)),
        (f"현재 추세({cur_name}) 계속 유지", dict(pools=pools, start=cur, fixed=True)),
        ("상승장 계속", dict(pools=pools, start=0, fixed=True)),
        ("횡보장 계속", dict(pools=pools, start=1, fixed=True)),
        ("하락장 계속", dict(pools=pools, start=2, fixed=True)),
        ("과거 전체 평균 (추세와 무관한 수익률)", dict(pools=[allp] * 3, start=cur, fixed=False)),
        ("변동성 1.5배 충격", dict(pools=pools, start=cur, fixed=False, vol_mult=1.5)),
    ]
    out, base_market = {}, None
    for i, (name, kw) in enumerate(scenarios):
        rets, states = simulate_market(
            kw["pools"], trans, kw["start"], kw["fixed"], n, days, rng, kw.get("vol_mult", 1.0)
        )
        paths, info = run_strategy(rets, states, capital, **strat)
        bh = to_paths(rets, capital)
        out[name] = {
            "paths": paths,
            "stat": evaluate(paths, capital, target, months),
            "info": info,
            "bh_stat": evaluate(bh, capital, target, months),
        }
        if i == 0:
            base_market = (rets, states)
    return out, base_market


def run_grid(rets, states, capital, target, months, entry, repeat, tps, sls):
    """익절 % × 손절 % 조합별 결과 (같은 시장 시뮬레이션을 재사용)."""
    res = {k: np.zeros((len(tps), len(sls))) for k in ("success", "final_p50", "loss", "mdd")}
    for i, t in enumerate(tps):
        for j, s in enumerate(sls):
            paths, _ = run_strategy(rets, states, capital, entry, t, s, repeat)
            e = evaluate(paths, capital, target, months)
            res["success"][i, j] = e["monthly_success"]
            res["final_p50"][i, j] = e["final_p50"]
            res["loss"][i, j] = e["principal_loss"]
            res["mdd"][i, j] = e["mdd_worst5"]
    return res


def run_entry_compare(rets, states, capital, target, months, tp, sl, repeat):
    rows = {}
    for label, entry in ENTRY_OPTIONS.items():
        paths, info = run_strategy(rets, states, capital, entry, tp, sl, repeat)
        rows[label] = {**evaluate(paths, capital, target, months), **info}
    return rows
# ===== CORE END =====


def won(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}{abs(v):,.0f}원"


def strat_text(entry_label, tp, sl, repeat):
    parts = [entry_label]
    parts.append(f"+{tp * 100:g}% 오르면 익절" if tp is not None else "익절 없음")
    parts.append(f"-{sl * 100:g}% 떨어지면 손절" if sl is not None else "손절 없음")
    parts.append("매도 후 재매수" if repeat else "1회만 매매")
    return " · ".join(parts)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("🎲 월 목표 수익 시뮬레이터")
st.caption(
    "과거 차트의 추세(상승장/횡보/하락장)별 움직임으로 수천 번 가상 투자를 돌려, "
    "내가 정한 매매 규칙(언제 사고 · 몇 %에 팔지)이 여러 상황에서 어떤 결과를 내는지 보여줍니다. "
    "참고용이며 투자 권유가 아닙니다."
)

with st.sidebar:
    st.header("① 대상 · 자금")
    category = st.selectbox("분류", list(ASSETS.keys()))
    asset_name = st.selectbox("대상 선택", list(ASSETS[category].keys()))
    ticker = ASSETS[category][asset_name]
    capital = st.number_input("초기 투자금 (원)", min_value=10_000, value=10_000_000, step=1_000_000)
    target = st.number_input("한 달 목표 수익 (원)", min_value=1_000, value=500_000, step=100_000)

    st.header("② 매매 전략")
    entry_label = st.selectbox("언제 살까?", list(ENTRY_OPTIONS.keys()), index=1)
    use_tp = st.checkbox("몇 % 오르면 팔까? (익절)", value=True)
    tp_pct = st.slider("익절 기준 (매수가 대비 +%)", 1, 50, 5, disabled=not use_tp)
    use_sl = st.checkbox("몇 % 떨어지면 팔까? (손절)", value=True)
    sl_pct = st.slider("손절 기준 (매수가 대비 -%)", 1, 50, 5, disabled=not use_sl)
    repeat = st.checkbox("팔고 나서 조건이 되면 다시 사기", value=True)
    if use_tp:
        st.caption(f"전액 투자 후 익절 1회 ≈ {won(capital * tp_pct / 100)} 수익")

    st.header("③ 시뮬레이션")
    months = st.slider("기간 (개월)", 1, 12, 6)
    n_sims = st.select_slider("시뮬레이션 횟수", options=[1000, 3000, 5000, 10000], value=5000)
    lookback = st.selectbox("참고할 과거 기간", ["10년", "5년", "3년"])
    fix_seed = st.checkbox("같은 결과 재현 (시드 고정)", value=False)
    run = st.button("🎲 시뮬레이션 돌리기", type="primary", use_container_width=True)

if run:
    years = {"10년": 10, "5년": 5, "3년": 3}[lookback]
    with st.spinner("과거 차트 불러오는 중..."):
        df = load(ticker, "10y")
    if df.empty or len(df) < 300:
        st.error("데이터를 불러오지 못했거나 분석에 필요한 기간(약 1.5년)이 부족합니다.")
        st.stop()
    close = df["Close"].tail(years * 252 + 140)
    model = build_market_model(close)
    seed = 42 if fix_seed else int(time.time()) % 1_000_000
    entry = ENTRY_OPTIONS[entry_label]
    tp = tp_pct / 100 if use_tp else None
    sl = sl_pct / 100 if use_sl else None
    strat = dict(entry=entry, tp=tp, sl=sl, repeat=repeat)

    with st.spinner(f"{n_sims:,}번 × 7가지 상황 시뮬레이션 중..."):
        res, (b_rets, b_states) = run_all_scenarios(
            model, float(capital), float(target), months, n_sims, seed, strat
        )
    with st.spinner("익절·손절·매수 시점 조합 비교 중..."):
        ng = min(n_sims, 3000)
        tps_grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
        sls_grid = [0.02, 0.03, 0.05, 0.10, 0.20, None]
        grid = run_grid(b_rets[:ng], b_states[:ng], float(capital), float(target), months,
                        entry, repeat, tps_grid, sls_grid)
        entry_cmp = run_entry_compare(b_rets[:ng], b_states[:ng], float(capital), float(target),
                                      months, tp, sl, repeat)
    st.session_state["sim"] = {
        "res": res, "model": model, "name": asset_name, "capital": float(capital),
        "target": float(target), "months": months, "n": n_sims, "lookback": lookback,
        "strat": strat, "entry_label": entry_label, "b_rets": b_rets[:30], "b_states": b_states[:30],
        "grid": grid, "tps_grid": tps_grid, "sls_grid": sls_grid, "entry_cmp": entry_cmp, "ng": ng,
    }

if "sim" not in st.session_state:
    st.info("왼쪽에서 대상·투자금·매매 전략을 정하고 **시뮬레이션 돌리기**를 눌러 주세요.")
    st.stop()

S = st.session_state["sim"]
res, model, strat = S["res"], S["model"], S["strat"]
capital, target, months, n = S["capital"], S["target"], S["months"], S["n"]
names = list(res.keys())
base_name = names[0]
base, base_info, base_bh = res[base_name]["stat"], res[base_name]["info"], res[base_name]["bh_stat"]
req_pct = target / capital * 100
cur = model["current"]
s_text = strat_text(S["entry_label"], strat["tp"], strat["sl"], strat["repeat"])

grade, color = risk_grade(base["monthly_success"], base["monthly_loss"])

st.subheader(f"{S['name']} · 현재 추세 {ICON[STATES[cur]]} {STATES[cur]} (추세 점수 {model['current_score']}/6)")
st.markdown(
    f"<div style='padding:22px;border-radius:14px;background:{color}20;"
    f"border:2px solid {color};text-align:center'>"
    f"<div style='font-size:14px;color:#6b7280'>전략: {s_text}</div>"
    f"<div style='font-size:14px;color:#6b7280'>투자금 {won(capital)} · 월 목표 {won(target)} "
    f"(월 {req_pct:.2f}%) · {months}개월 · {n:,}회 시뮬레이션</div>"
    f"<div style='font-size:40px;font-weight:800;color:{color}'>위험도 {grade}</div>"
    f"<div style='font-size:20px;color:{color}'>한 달 목표 달성 확률 약 {base['monthly_success']:.0f}% · "
    f"월 손실 확률 약 {base['monthly_loss']:.0f}%</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.write("")

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{months}개월 누적 목표 달성", f"{base['cum_target']:.0f}%")
k2.metric("원금 손실 확률", f"{base['principal_loss']:.0f}%")
k3.metric("최악 5% 최종 자산", won(base["final_p5"]), f"{(base['final_p5'] / capital - 1) * 100:+.1f}%", delta_color="off")
k4.metric("최악 5% 최대 낙폭", f"{base['mdd_worst5']:.1f}%")

tab_sim, tab_cmp, tab_var, tab_ana = st.tabs(
    ["🎲 시뮬레이션 결과", "⚖️ 상황별 비교", "🔬 전략 변인 비교", "🧠 분석 결과"]
)

# ── 시뮬레이션 결과 ───────────────────────────
with tab_sim:
    pick = st.selectbox("보고 싶은 상황", names, index=0)
    paths = res[pick]["paths"]
    st_ = res[pick]["stat"]
    inf = res[pick]["info"]
    x = np.arange(paths.shape[1]) / HORIZON

    rng_vis = np.random.default_rng(0)
    idx = rng_vis.choice(paths.shape[0], size=min(60, paths.shape[0]), replace=False)
    q = {k: np.percentile(paths, k, axis=0) for k in (5, 25, 50, 75, 95)}

    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=x, y=paths[i], mode="lines", line=dict(width=0.6, color="rgba(100,116,139,0.35)"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[95], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[5], fill="tonexty", fillcolor="rgba(37,99,235,0.12)",
                             line=dict(width=0), name="상·하위 5% 범위"))
    fig.add_trace(go.Scatter(x=x, y=q[75], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=q[25], fill="tonexty", fillcolor="rgba(37,99,235,0.25)",
                             line=dict(width=0), name="25~75% 범위"))
    fig.add_trace(go.Scatter(x=x, y=q[50], name="중앙값", line=dict(color="#2563eb", width=3)))
    fig.add_trace(go.Scatter(x=x, y=capital + target * x, name="목표 경로", line=dict(color="#dc2626", dash="dash", width=2)))
    fig.add_hline(y=capital, line_color="#111827", line_width=1)
    fig.update_layout(title=f"{pick} — 자산 변화 ({n:,}번 중 60번 표시)", xaxis_title="개월",
                      yaxis_title="자산 (원)", height=480, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=70, b=10))
    st.plotly_chart(fig, use_container_width=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("월 목표 달성률", f"{st_['monthly_success']:.1f}%")
    d2.metric(f"{months}개월 내내 매달 달성", f"{st_['all_months']:.1f}%")
    d3.metric("한 번이라도 달성", f"{st_['any_month']:.1f}%")
    d4.metric("최종 자산 중앙값", won(st_["final_p50"]))

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("매수가 한 번이라도 발생", f"{inf['entered']:.0f}%")
    e2.metric("평균 매도 횟수", f"{inf['avg_trades']:.1f}회")
    e3.metric("익절 비율 (익절/전체 매도)", "-" if np.isnan(inf["tp_share"]) else f"{inf['tp_share']:.0f}%")
    e4.metric("자금이 투자돼 있던 비중", f"{inf['invested_ratio']:.0f}%")
    if inf["entered"] < 5:
        st.warning("이 상황에서는 매수 조건이 거의 충족되지 않아 투자가 일어나지 않았습니다. "
                   "예: '하락장 매수' 전략을 '상승장 계속' 상황에 적용하면 사지 않고 현금으로 남습니다.")

    c_left, c_right = st.columns(2)
    with c_left:
        m1 = (paths[:, HORIZON] - capital) / capital * 100
        h1 = go.Figure(go.Histogram(x=m1, nbinsx=50, marker_color="#6366f1"))
        h1.add_vline(x=req_pct, line_dash="dash", line_color="#dc2626", annotation_text=f"목표 {req_pct:.1f}%")
        h1.add_vline(x=0, line_color="#111827", line_width=1)
        h1.update_layout(title="1개월 뒤 수익률 분포", xaxis_title="수익률 (%)", height=340,
                         margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(h1, use_container_width=True)
    with c_right:
        hf = go.Figure(go.Histogram(x=paths[:, -1], nbinsx=50, marker_color="#0ea5e9"))
        hf.add_vline(x=capital + target * months, line_dash="dash", line_color="#dc2626",
                     annotation_text="누적 목표")
        hf.add_vline(x=capital, line_color="#111827", line_width=1)
        hf.update_layout(title=f"{months}개월 뒤 최종 자산 분포", xaxis_title="자산 (원)", height=340,
                         margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(hf, use_container_width=True)

    # 예시 한 번의 투자 (기본 상황의 가상 경로 하나를 골라 매수·매도 시점 표시)
    st.markdown("#### 🔎 가상 투자 한 번 따라가 보기 (기본 상황)")
    nb = S["b_rets"].shape[0]
    path_no = st.slider("가상 경로 번호", 1, nb, 1)
    rr, ss = S["b_rets"][path_no - 1], S["b_states"][path_no - 1]
    price, buys, tps_, sls_ = trace_path(rr, ss, strat["entry"], strat["tp"], strat["sl"], strat["repeat"])
    eq = res[base_name]["paths"][path_no - 1]
    dd_ = np.arange(len(price))

    tf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45],
                       subplot_titles=("가격 (시작=100)과 매수·매도 시점", "내 자산"))
    seg_start = 0
    for d in range(1, len(ss) + 1):
        if d == len(ss) or ss[d] != ss[seg_start]:
            tf.add_vrect(x0=seg_start, x1=d, fillcolor=STATE_COLOR[int(ss[seg_start])], line_width=0,
                         layer="below", row="all", col=1)
            seg_start = d
    tf.add_trace(go.Scatter(x=dd_, y=price, name="가격", line=dict(color="#111827", width=1.5)), row=1, col=1)
    if buys:
        tf.add_trace(go.Scatter(x=buys, y=price[buys], mode="markers", name="매수",
                                marker=dict(symbol="triangle-up", size=12, color="#16a34a")), row=1, col=1)
    if tps_:
        tf.add_trace(go.Scatter(x=tps_, y=price[tps_], mode="markers", name="익절 매도",
                                marker=dict(symbol="circle", size=11, color="#dc2626")), row=1, col=1)
    if sls_:
        tf.add_trace(go.Scatter(x=sls_, y=price[sls_], mode="markers", name="손절 매도",
                                marker=dict(symbol="x", size=11, color="#2563eb")), row=1, col=1)
    tf.add_trace(go.Scatter(x=dd_, y=eq, name="자산", line=dict(color="#7c3aed", width=2)), row=2, col=1)
    tf.add_hline(y=capital, line_color="#111827", line_width=1, row=2, col=1)
    tf.update_layout(height=560, legend=dict(orientation="h", y=1.1, x=0), margin=dict(l=10, r=10, t=60, b=10))
    tf.update_xaxes(title_text="거래일", row=2, col=1)
    st.plotly_chart(tf, use_container_width=True)
    st.caption(f"이 경로: 매수 {len(buys)}회 · 익절 {len(tps_)}회 · 손절 {len(sls_)}회 · "
               f"최종 {won(float(eq[-1]))} ({(eq[-1] / capital - 1) * 100:+.1f}%) · "
               "배경색 🟢 상승장 🟡 횡보 🔴 하락장")

# ── 상황별 비교 ───────────────────────────────
with tab_cmp:
    rows = []
    for nm in names:
        s, f, b = res[nm]["stat"], res[nm]["info"], res[nm]["bh_stat"]
        rows.append({
            "상황": nm,
            "월 목표 달성률": f"{s['monthly_success']:.1f}%",
            "월 손실 확률": f"{s['monthly_loss']:.1f}%",
            f"{months}개월 누적 목표 달성": f"{s['cum_target']:.1f}%",
            "원금 손실 확률": f"{s['principal_loss']:.1f}%",
            "최종 자산 중앙값": won(s["final_p50"]),
            "최악 5% 최종 자산": won(s["final_p5"]),
            "최대 낙폭 -20% 이하": f"{s['mdd20']:.1f}%",
            "매수 발생 확률": f"{f['entered']:.0f}%",
            "그냥 보유 시 최종 중앙값": won(b["final_p50"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    bar = go.Figure()
    bar.add_trace(go.Bar(x=names, y=[res[k]["stat"]["monthly_success"] for k in names], name="월 목표 달성률", marker_color="#16a34a"))
    bar.add_trace(go.Bar(x=names, y=[res[k]["stat"]["principal_loss"] for k in names], name="원금 손실 확률", marker_color="#dc2626"))
    bar.update_layout(barmode="group", yaxis_title="%", height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(bar, use_container_width=True)
    st.caption("'그냥 보유'는 첫날 전액 매수 후 끝까지 팔지 않았을 때의 결과로, 내 전략이 도움이 되는지 비교하는 기준입니다.")

# ── 전략 변인 비교 ────────────────────────────
with tab_var:
    st.markdown(f"기본 상황(**{base_name}**)의 같은 시장 시뮬레이션 {S['ng']:,}개에 전략만 바꿔 적용한 결과입니다.")

    st.markdown("#### ① 언제 살까? — 매수 시점별 비교")
    st.caption(f"익절/손절 설정은 현재 선택 그대로: "
               f"{strat_text('', strat['tp'], strat['sl'], strat['repeat']).lstrip(' · ')}")
    erows = []
    for lb, v in S["entry_cmp"].items():
        erows.append({
            "매수 시점": lb,
            "월 목표 달성률": f"{v['monthly_success']:.1f}%",
            "원금 손실 확률": f"{v['principal_loss']:.1f}%",
            "최종 자산 중앙값": won(v["final_p50"]),
            "최악 5% 최대 낙폭": f"{v['mdd_worst5']:.1f}%",
            "평균 매도 횟수": f"{v['avg_trades']:.1f}회",
            "투자 비중": f"{v['invested_ratio']:.0f}%",
        })
    st.dataframe(pd.DataFrame(erows), hide_index=True, use_container_width=True)

    st.markdown("#### ②·③ 몇 % 오르면 / 몇 % 떨어지면 팔까? — 익절 × 손절 조합")
    metric_label = st.radio("보고 싶은 지표", ["월 목표 달성률", "최종 자산 중앙값", "원금 손실 확률", "최악 5% 최대 낙폭"],
                            horizontal=True)
    key = {"월 목표 달성률": "success", "최종 자산 중앙값": "final_p50",
           "원금 손실 확률": "loss", "최악 5% 최대 낙폭": "mdd"}[metric_label]
    z = S["grid"][key]
    if key == "final_p50":
        text = [[f"{v / 10000:,.0f}만" for v in row] for row in z]
    else:
        text = [[f"{v:.0f}%" for v in row] for row in z]
    good_high = key in ("success", "final_p50", "mdd")  # mdd는 0에 가까울수록(큰 값) 좋음
    hm = go.Figure(go.Heatmap(
        z=z, x=["손절 없음" if s is None else f"-{s * 100:g}%" for s in S["sls_grid"]],
        y=[f"+{t * 100:g}%" for t in S["tps_grid"]], text=text, texttemplate="%{text}",
        colorscale="RdYlGn" if good_high else "RdYlGn_r", showscale=False,
    ))
    hm.update_layout(xaxis_title="손절 기준 (떨어지면 팔기)", yaxis_title="익절 기준 (오르면 팔기)",
                     height=440, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(hm, use_container_width=True)

    gs = S["grid"]["success"]
    bi, bj = np.unravel_index(np.argmax(gs), gs.shape)
    sl_txt = "손절 없음" if S["sls_grid"][bj] is None else f"-{S['sls_grid'][bj] * 100:g}% 손절"
    st.success(f"이 격자에서 월 목표 달성률이 가장 높은 조합: **+{S['tps_grid'][bi] * 100:g}% 익절 / {sl_txt}** "
               f"({gs[bi, bj]:.1f}%). 다만 같은 시뮬레이션 안에서 고른 값이라 실제로는 이보다 낮을 수 있습니다.")

# ── 분석 결과 ─────────────────────────────────
with tab_ana:
    best = max(names, key=lambda k: res[k]["stat"]["monthly_success"])
    worst = min(names, key=lambda k: res[k]["stat"]["monthly_success"])
    bs, ws = res[best]["stat"], res[worst]["stat"]

    st.markdown("### 📌 한눈에 보기")
    st.markdown(
        f"- 전략: **{s_text}**\n"
        f"- 월 {won(target)}을 벌려면 투자금 {won(capital)} 기준 **한 달에 {req_pct:.2f}%** 수익이 필요하고, "
        f"매달 달성하면 연 환산 약 **{((1 + req_pct / 100) ** 12 - 1) * 100:,.0f}%** 입니다.\n"
        f"- 기본 상황에서 한 달 목표 달성률은 **{base['monthly_success']:.0f}%**, "
        f"{months}개월 내내 매달 달성할 확률은 **{base['all_months']:.1f}%** 입니다.\n"
        f"- {months}개월 뒤 원금보다 줄어 있을 확률은 **{base['principal_loss']:.0f}%**, "
        f"최악 5% 경우 자산은 **{won(base['final_p5'])}**, 최대 낙폭은 **{base['mdd_worst5']:.1f}%** 입니다."
    )

    st.markdown("### 🧭 매매 규칙은 어떻게 작동했나")
    lines = [
        f"- 기본 상황에서 매수가 한 번이라도 일어난 비율은 **{base_info['entered']:.0f}%**, "
        f"평균 매도 횟수는 **{base_info['avg_trades']:.1f}회**, 자금이 투자돼 있던 기간은 **{base_info['invested_ratio']:.0f}%** 입니다."
    ]
    if strat["tp"] is not None and strat["sl"] is not None and not np.isnan(base_info["tp_share"]):
        be = strat["sl"] / (strat["tp"] + strat["sl"]) * 100
        lines.append(
            f"- 매도 중 익절 비율은 **{base_info['tp_share']:.0f}%** 입니다. +{strat['tp'] * 100:g}% 익절 / -{strat['sl'] * 100:g}% 손절 구조에서 "
            f"본전이 되려면 익절 비율이 약 **{be:.0f}%** 이상이어야 합니다."
            + (" 현재는 이 기준을 넘습니다." if base_info["tp_share"] >= be else " 현재는 이 기준에 못 미칩니다.")
        )
    if strat["tp"] is not None:
        need = target / (capital * strat["tp"])
        lines.append(f"- 익절 1회 수익은 약 {won(capital * strat['tp'])}라서, 월 목표를 채우려면 한 달에 익절이 **약 {need:.1f}회** 필요합니다.")
    if strat["sl"] is None:
        lines.append("- 손절이 없어서 하락이 길어지면 손실이 커질 수 있습니다. 최악 5% 최대 낙폭을 확인하세요.")
    st.markdown("\n".join(lines))

    st.markdown("### 🆚 그냥 보유했다면?")
    diff = base["final_p50"] - base_bh["final_p50"]
    st.markdown(
        f"- 최종 자산 중앙값: 전략 **{won(base['final_p50'])}** vs 그냥 보유 **{won(base_bh['final_p50'])}** "
        f"→ 전략이 **{won(abs(diff))} {'더 높음' if diff >= 0 else '더 낮음'}**\n"
        f"- 원금 손실 확률: 전략 **{base['principal_loss']:.0f}%** vs 그냥 보유 **{base_bh['principal_loss']:.0f}%**\n"
        f"- 최악 5% 최대 낙폭: 전략 **{base['mdd_worst5']:.1f}%** vs 그냥 보유 **{base_bh['mdd_worst5']:.1f}%**"
    )

    st.markdown("### ⚖️ 상황에 따른 차이")
    st.markdown(
        f"- 가장 유리한 상황은 **{best}** (월 목표 달성률 {bs['monthly_success']:.0f}%), "
        f"가장 불리한 상황은 **{worst}** ({ws['monthly_success']:.0f}%) 입니다. "
        f"시장 상황만으로 성공률이 **{bs['monthly_success'] - ws['monthly_success']:.0f}%p** 달라집니다.\n"
        f"- 하락장이 이어지면 원금 손실 확률은 **{res['하락장 계속']['stat']['principal_loss']:.0f}%**, "
        f"상승장이 이어지면 **{res['상승장 계속']['stat']['principal_loss']:.0f}%** 입니다.\n"
        f"- 변동성이 1.5배로 커지면 최악 5% 최대 낙폭이 **{res['변동성 1.5배 충격']['stat']['mdd_worst5']:.1f}%** 까지 깊어집니다."
    )

    st.markdown("### 🎯 현실적인 목표는?")
    p50, p30 = base["m1_p50"], base["m1_p30"]
    lines = []
    if p50 > 0:
        lines.append(f"- 이 전략으로 한 달 목표 달성 확률 **약 50%** 수준: 월 **{won(p50)}** ({p50 / capital * 100:.2f}%)")
    else:
        lines.append("- 이 전략의 한 달 수익 중앙값이 0 이하여서, 달성 확률 50%를 만들 수 있는 양의 목표가 없습니다. 매수 시점이나 익절·손절 값을 바꿔 보세요.")
    if p30 > 0:
        lines.append(f"- 한 달 목표 달성 확률 **약 70%** 수준: 월 **{won(p30)}** ({p30 / capital * 100:.2f}%)")
    lines.append(f"- 현재 목표({won(target)})는 이 기준을 {'웃돌아' if target > max(p50, 0) else '밑돌아'} "
                 f"{'다소 공격적' if target > max(p50, 0) else '비교적 보수적'}인 목표입니다.")
    st.markdown("\n".join(lines))

    st.markdown("### ⚠️ 해석할 때 주의")
    notes = []
    if cur == 2:
        notes.append("- 현재 **하락장** 구간입니다. 하락장이 이어질 때의 시뮬레이션 결과를 특히 눈여겨보세요.")
    if req_pct > 10:
        notes.append(f"- 한 달 {req_pct:.1f}% 목표는 매우 높습니다. 높은 목표일수록 큰 손실 위험과 함께 가야 합니다.")
    if base["mdd20"] >= 20:
        notes.append(f"- {months}개월 동안 -20% 이상 떨어지는 구간을 겪을 확률이 **{base['mdd20']:.0f}%** 입니다. 중간에 버틸 수 있는지 생각해 보세요.")
    notes.append("- 매수·매도는 하루 종가 기준으로 체결된다고 가정했고, 수수료·세금·슬리피지·환율은 반영하지 않았습니다. 거래가 잦은 전략일수록 실제 결과는 더 나빠집니다.")
    notes.append("- 시뮬레이션은 과거 일간 수익률을 무작위로 다시 뽑는 방식이라, 과거에 없던 충격은 반영하지 못합니다.")
    st.markdown("\n".join(notes))

with st.expander("시뮬레이션 방법"):
    st.markdown(
        f"""
1. 과거 {S['lookback']} 차트에서 매일의 추세를 6점 기준(종가 vs 20/60/120일선, 정배열, 60일선 기울기)으로 판정해 상승장(4~6점)·횡보(3점)·하락장(0~2점)으로 나눕니다.
2. 추세별 **다음 날 수익률**(상승장 {model['sizes'][0]:,}일 · 횡보 {model['sizes'][1]:,}일 · 하락장 {model['sizes'][2]:,}일)과 추세 **전환 확률**을 계산합니다.
3. 현재 추세에서 출발해 매일 그 추세의 수익률을 무작위로 뽑고, 전환 확률에 따라 추세를 바꿔 가며 {HORIZON * months}거래일 경로를 {n:,}개 만듭니다. (그날의 추세도 함께 기록)
4. 각 경로 위에서 **내 매매 규칙**을 적용합니다: 선택한 추세가 되면 전액 매수 → 매수가 대비 익절·손절 기준에 닿으면 종가에 전량 매도 → (선택 시) 조건이 되면 다시 매수. 매도 후 현금으로 있는 동안 수익률은 0%입니다.
5. 같은 방식으로 '상승장 계속', '하락장 계속', '변동성 1.5배' 등 7가지 상황을 돌리고, 한 달(21거래일)마다 손익이 목표 금액 이상인지 셉니다.
6. '전략 변인 비교' 탭은 같은 시장 시뮬레이션에 매수 시점·익절·손절만 바꿔 적용한 결과입니다.
        """
    )

st.caption("⚠️ 본 결과는 과거 가격 기반 시뮬레이션이며 미래 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")
