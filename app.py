import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="목표 수익 위험도 분석", page_icon="🎯", layout="wide")

# ──────────────────────────────────────────────
# 종목 사전 (원본 앱과 동일 방식, 필요한 만큼 추가하세요)
# ──────────────────────────────────────────────
STOCKS = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "현대차": "005380.KS",
    "네이버": "035420.KS", "카카오": "035720.KS", "에코프로": "086520.KQ",
    "에코프로비엠": "247540.KQ", "알테오젠": "196170.KQ", "셀트리온": "068270.KS",
    "애플": "AAPL", "엔비디아": "NVDA", "마이크로소프트": "MSFT", "테슬라": "TSLA",
    "구글": "GOOGL", "아마존": "AMZN", "메타": "META", "팔란티어": "PLTR",
    "토요타": "7203.T", "소니": "6758.T", "텐센트": "0700.HK", "TSMC": "2330.TW",
    "코스피": "^KS11", "KOSPI": "^KS11", "코스닥": "^KQ11", "KOSDAQ": "^KQ11",
    "나스닥": "^IXIC", "NASDAQ": "^IXIC", "S&P500": "^GSPC", "다우존스": "^DJI",
    "닛케이": "^N225", "SPY": "SPY", "QQQ": "QQQ", "TQQQ": "TQQQ",
}
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-=]{0,11}$|^\^[A-Z0-9.\-]{1,12}$")

HORIZON = 21  # 한 달 = 21거래일
REGIME_ICON = {"상승장": "🟢", "횡보": "🟡", "하락장": "🔴"}


# ──────────────────────────────────────────────
# 데이터 / 종목 검색
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


def resolve_ticker(query: str):
    q = query.strip()
    if not q:
        return None, None
    key = q.replace(" ", "").upper()
    for name, tk in STOCKS.items():
        if name.replace(" ", "").upper() == key:
            return name, tk
    if key.isdigit() and len(key) == 6:
        for suf in (".KS", ".KQ"):
            if not load(key + suf, "6mo").empty:
                return key, key + suf
    if not key.isdigit() and TICKER_RE.match(key):
        if not load(key, "6mo").empty:
            return key, key
    matches = [(n, t) for n, t in STOCKS.items() if key in n.replace(" ", "").upper()]
    if matches:
        return matches[0]
    try:
        for r in yf.Search(q, max_results=10).quotes:
            sym = r.get("symbol", "")
            if sym and str(r.get("quoteType", "")).upper() in ("EQUITY", "ETF", "INDEX", ""):
                return r.get("shortname") or r.get("longname") or sym, sym
    except Exception:
        pass
    return None, None


# ──────────────────────────────────────────────
# 추세 판단 (원본 앱의 6점 기준 그대로)
# ──────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for n in (20, 60, 120):
        df[f"MA{n}"] = df["Close"].rolling(n).mean()
    return df


def trend_score_series(df: pd.DataFrame) -> pd.Series:
    c, m20, m60, m120 = df["Close"], df["MA20"], df["MA60"], df["MA120"]
    score = (
        (c > m20).astype(int) + (c > m60).astype(int) + (c > m120).astype(int)
        + (m20 > m60).astype(int) + (m60 > m120).astype(int)
        + (m60 > m60.shift(20)).astype(int)
    )
    return score.where(m120.notna() & m60.shift(20).notna())


def regime_of(score: float) -> str:
    if score >= 4:
        return "상승장"
    if score == 3:
        return "횡보"
    return "하락장"


# ──────────────────────────────────────────────
# 월간 수익률 분포 계산
# ──────────────────────────────────────────────
def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """매 거래일을 시작점으로 한 '이후 21거래일' 결과 + 시작 시점의 추세."""
    c = df["Close"]
    score = trend_score_series(df)
    fwd_ret = c.shift(-HORIZON) / c - 1
    # 이후 21거래일 중 최저 종가 (월 중 최대 낙폭용)
    fwd_min = c.shift(-1)[::-1].rolling(HORIZON).min()[::-1]
    fwd_dd = fwd_min / c - 1
    out = pd.DataFrame({"score": score, "ret": fwd_ret, "dd": fwd_dd}).dropna()
    out["regime"] = out["score"].apply(regime_of)
    return out


def monte_carlo(df: pd.DataFrame, regime: str, n_sims: int = 10000, seed: int = 42):
    """같은 추세였던 날들의 일간 수익률을 무작위로 뽑아 21일 경로를 만든다."""
    c = df["Close"]
    score = trend_score_series(df)
    daily = c.pct_change().shift(-1)  # 그날 추세 → 다음날 수익률
    pool = daily[score.apply(lambda s: regime_of(s) if pd.notna(s) else None) == regime].dropna()
    used_all = False
    if len(pool) < 200:
        pool = c.pct_change().dropna()
        used_all = True
    rng = np.random.default_rng(seed)
    draws = rng.choice(pool.values, size=(n_sims, HORIZON), replace=True)
    paths = np.cumprod(1 + draws, axis=1)
    return paths[:, -1] - 1, paths.min(axis=1) - 1, used_all, len(pool)


def summarize(rets: np.ndarray, dds: np.ndarray, req: float) -> dict:
    return {
        "n": len(rets),
        "success": float((rets >= req).mean() * 100),
        "loss": float((rets < 0).mean() * 100),
        "big_loss": float((rets <= -0.10).mean() * 100),
        "mean": float(rets.mean() * 100),
        "median": float(np.median(rets) * 100),
        "var95": float(np.percentile(rets, 5) * 100),
        "best5": float(np.percentile(rets, 95) * 100),
        "dd_avg": float(dds.mean() * 100),
        "dd_worst": float(np.percentile(dds, 5) * 100),
    }


def risk_grade(success: float, loss: float, req_pct: float):
    if req_pct <= 0:
        return "-", "#6b7280"
    if success >= 60 and loss <= 35:
        return "🟢 낮음", "#16a34a"
    if success >= 40:
        return "🟡 보통", "#ca8a04"
    if success >= 20:
        return "🟠 높음", "#ea580c"
    return "🔴 매우 높음", "#dc2626"


def won(v: float) -> str:
    return f"{v:,.0f}원"


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("🎯 목표 수익 위험도 · 성공률 분석")
st.caption(
    "과거 차트의 추세(상승장/횡보/하락장)와 월간 수익률 분포로 계산한 참고용 통계이며 "
    "투자 권유가 아닙니다. 과거 확률이 미래를 보장하지 않습니다."
)

with st.sidebar:
    st.header("입력")
    query = st.text_input("종목 / 지수", value="코스피", help="종목명, 6자리 코드, 티커(AAPL, ^IXIC 등)")
    capital = st.number_input("초기 투자금 (원)", min_value=10_000, value=10_000_000, step=1_000_000)
    target = st.number_input("한 달 목표 수익 (원)", min_value=1_000, value=500_000, step=100_000)
    ratio = st.slider("투자금 중 이 종목에 넣는 비율 (%)", 10, 100, 100, 10)
    lookback = st.selectbox("분석에 쓸 과거 기간", ["10년", "5년", "3년"], index=0)
    run = st.button("분석하기", type="primary", use_container_width=True)

if not run and "ran" not in st.session_state:
    st.info("왼쪽에서 종목, 초기 투자금, 한 달 목표 수익을 입력하고 **분석하기**를 눌러 주세요.")
    st.stop()
st.session_state["ran"] = True

with st.spinner("데이터 분석 중..."):
    name, ticker = resolve_ticker(query)
    if not ticker:
        st.warning(f"'{query}' 종목을 찾지 못했습니다. 티커나 6자리 종목코드로 검색해 보세요.")
        st.stop()
    df = load(ticker, "10y")
    if df.empty or len(df) < 300:
        st.error("분석에 필요한 데이터가 부족합니다. (최소 약 1.5년 이상 필요)")
        st.stop()
    df = add_indicators(df)
    years = {"10년": 10, "5년": 5, "3년": 3}[lookback]
    df = df.tail(int(years * 252) + 140)  # 이동평균 계산용 여유분 포함

    score_s = trend_score_series(df)
    cur_score = int(score_s.dropna().iloc[-1])
    cur_regime = regime_of(cur_score)

    windows = build_windows(df)
    windows = windows.tail(years * 252)

    invested = capital * ratio / 100
    req = target / invested  # 투자한 돈 기준 필요 수익률
    req_pct = req * 100
    annual = ((1 + req) ** 12 - 1) * 100

    all_stat = summarize(windows["ret"].values, windows["dd"].values, req)
    same = windows[windows["regime"] == cur_regime]
    same_stat = summarize(same["ret"].values, same["dd"].values, req) if len(same) >= 30 else None
    mc_r, mc_d, mc_all, mc_n = monte_carlo(df, cur_regime)
    mc_stat = summarize(mc_r, mc_d, req)

# 종합 확률: 같은 추세 과거 통계 + 시뮬레이션 평균 (같은 추세 표본이 적으면 전체 사용)
base = same_stat if same_stat else all_stat
final_success = (base["success"] + mc_stat["success"]) / 2
final_loss = (base["loss"] + mc_stat["loss"]) / 2
grade, color = risk_grade(final_success, final_loss, req_pct)

st.subheader(f"{name} ({ticker})")
st.markdown(
    f"현재 추세: **{REGIME_ICON[cur_regime]} {cur_regime}** (추세 점수 {cur_score}/6) · "
    f"분석 기간 {lookback}"
)

st.markdown(
    f"<div style='padding:22px;border-radius:14px;background:{color}20;"
    f"border:2px solid {color};text-align:center'>"
    f"<div style='font-size:15px;color:#6b7280'>목표 {won(target)} / 투자금 {won(invested)} "
    f"→ 한 달 필요 수익률 {req_pct:.2f}%</div>"
    f"<div style='font-size:40px;font-weight:800;color:{color}'>위험도 {grade}</div>"
    f"<div style='font-size:20px;color:{color}'>목표 달성 확률 약 {final_success:.0f}% · "
    f"손실 확률 약 {final_loss:.0f}%</div>"
    f"</div>",
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("한 달 필요 수익률", f"{req_pct:.2f}%")
c2.metric("연환산 (매달 달성 시)", f"{annual:,.0f}%")
c3.metric("최악 5% 시나리오 손실", f"{base['var95']:.1f}%",
          f"약 {won(invested * base['var95'] / 100)}", delta_color="off")
c4.metric("월 중 평균 최대 낙폭", f"{base['dd_avg']:.1f}%")

if req_pct > 10:
    st.warning(f"한 달 {req_pct:.1f}% 수익은 대부분의 종목에서 과거에도 드물게 달성된 수준입니다. 목표를 낮추거나 변동성 큰 종목/레버리지 위험을 반드시 감안하세요.")
if cur_regime == "하락장":
    st.warning("현재 하락장 구간입니다. 같은 추세 구간의 과거 통계를 중심으로 해석하세요.")

# 상세 비교 표
rows = [("전체 기간 (모든 시점)", all_stat)]
if same_stat:
    rows.append((f"현재와 같은 추세({cur_regime})였던 시점", same_stat))
rows.append((f"시뮬레이션 {len(mc_r):,}회" + (" (전체 일간수익률)" if mc_all else f" ({cur_regime} 일간수익률)"), mc_stat))
table = pd.DataFrame(
    [
        {
            "기준": lbl,
            "표본 수": f"{s['n']:,}",
            "목표 달성 확률": f"{s['success']:.1f}%",
            "손실 확률": f"{s['loss']:.1f}%",
            "-10% 이하 확률": f"{s['big_loss']:.1f}%",
            "평균 월수익률": f"{s['mean']:+.2f}%",
            "중앙값": f"{s['median']:+.2f}%",
            "하위 5%": f"{s['var95']:+.1f}%",
            "상위 5%": f"{s['best5']:+.1f}%",
        }
        for lbl, s in rows
    ]
)
st.markdown("**기준별 비교**")
st.dataframe(table, hide_index=True, use_container_width=True)

# 분포 차트
fig = go.Figure()
fig.add_trace(go.Histogram(x=windows["ret"] * 100, nbinsx=60, name="전체 기간",
                           marker_color="#9ca3af", opacity=0.6))
if same_stat:
    fig.add_trace(go.Histogram(x=same["ret"] * 100, nbinsx=60, name=f"{cur_regime}였던 시점",
                               marker_color="#2563eb", opacity=0.6))
fig.add_vline(x=req_pct, line_dash="dash", line_color="#dc2626",
              annotation_text=f"목표 {req_pct:.1f}%", annotation_position="top")
fig.add_vline(x=0, line_color="#111827", line_width=1)
fig.update_layout(barmode="overlay", title="21거래일(약 1개월) 수익률 분포",
                  xaxis_title="월 수익률 (%)", yaxis_title="빈도", height=420,
                  legend=dict(orientation="h", y=1.08, x=0), margin=dict(l=10, r=10, t=70, b=10))
st.plotly_chart(fig, use_container_width=True)

# 목표별 성공률 표
st.markdown("**목표 금액별 달성 확률 (참고)**")
steps = sorted({target * m for m in (0.25, 0.5, 1, 2, 4)})
ref = same["ret"].values if same_stat else windows["ret"].values
ref_rows = []
for t in steps:
    r = t / invested
    ref_rows.append({
        "한 달 목표": won(t),
        "필요 수익률": f"{r * 100:.2f}%",
        "과거 통계 달성 확률": f"{(ref >= r).mean() * 100:.1f}%",
        "시뮬레이션 달성 확률": f"{(mc_r >= r).mean() * 100:.1f}%",
    })
st.dataframe(pd.DataFrame(ref_rows), hide_index=True, use_container_width=True)

with st.expander("계산 방법 / 한계"):
    st.markdown(
        f"""
- **추세 판단**: 원본 앱과 같은 6점 기준(종가 vs 20/60/120일선, 정배열, 60일선 기울기)으로 매일 상승장(4~6) / 횡보(3) / 하락장(0~2)을 구분합니다.
- **과거 통계**: 모든 거래일을 시작점으로 21거래일 뒤 수익률을 구하고, 현재와 같은 추세였던 시점만 따로 모았습니다. (표본이 30개 미만이면 전체 기간 사용)
- **시뮬레이션**: 같은 추세였던 날들의 일간 수익률을 무작위로 뽑아 21일 경로 10,000개를 만들었습니다. (같은 추세 표본 {mc_n:,}일)
- **종합 확률**: 과거 통계와 시뮬레이션 달성 확률의 평균입니다.
- **위험도 등급**: 달성 확률 60%↑ & 손실 확률 35%↓ = 낮음 / 40%↑ = 보통 / 20%↑ = 높음 / 그 미만 = 매우 높음.
- **한계**: 수수료·세금·환율은 반영하지 않았고, 겹치는 구간이라 표본이 독립적이지 않으며, 과거 분포가 미래에도 유지된다는 가정입니다.
        """
    )

st.caption("⚠️ 본 결과는 과거 가격 기반 통계이며 미래 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")
