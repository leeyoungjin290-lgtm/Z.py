import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="한국 주식 추세 분석", page_icon="📈", layout="wide")

# ──────────────────────────────────────────────
# 종목 사전 (이름 → 야후파이낸스 티커)
# 목록에 없는 종목은 6자리 종목코드(예: 005930)로도 검색할 수 있습니다.
# ──────────────────────────────────────────────
STOCKS = {
    "삼성전자": "005930.KS", "삼성전자우": "005935.KS", "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS", "삼성바이오로직스": "207940.KS", "현대차": "005380.KS",
    "기아": "000270.KS", "셀트리온": "068270.KS", "KB금융": "105560.KS",
    "신한지주": "055550.KS", "하나금융지주": "086790.KS", "우리금융지주": "316140.KS",
    "POSCO홀딩스": "005490.KS", "포스코홀딩스": "005490.KS", "NAVER": "035420.KS",
    "네이버": "035420.KS", "카카오": "035720.KS", "카카오뱅크": "323410.KS",
    "LG화학": "051910.KS", "삼성SDI": "006400.KS", "현대모비스": "012330.KS",
    "삼성물산": "028260.KS", "한화에어로스페이스": "012450.KS", "HD현대중공업": "329180.KS",
    "두산에너빌리티": "034020.KS", "크래프톤": "259960.KS", "삼성생명": "032830.KS",
    "삼성화재": "000810.KS", "한국전력": "015760.KS", "SK이노베이션": "096770.KS",
    "SK텔레콤": "017670.KS", "KT": "030200.KS", "LG전자": "066570.KS",
    "삼성전기": "009150.KS", "하이브": "352820.KS", "엔씨소프트": "036570.KS",
    "포스코퓨처엠": "003670.KS", "현대건설": "000720.KS", "대한항공": "003490.KS",
    "한미반도체": "042700.KS", "두산로보틱스": "454910.KS", "LG": "003550.KS",
    "SK": "034730.KS", "롯데케미칼": "011170.KS", "고려아연": "010130.KS",
    "현대글로비스": "086280.KS", "이마트": "139480.KS", "아모레퍼시픽": "090430.KS",
    "한화오션": "042660.KS", "에이피알": "278470.KS",
    # 코스닥
    "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ", "알테오젠": "196170.KQ",
    "레인보우로보틱스": "277810.KQ", "HLB": "028300.KQ", "펄어비스": "263750.KQ",
    "리노공업": "058470.KQ", "셀트리온제약": "068760.KQ",
}

INDICES = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11"}


# ──────────────────────────────────────────────
# 데이터 / 지표 계산
# ──────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    for n in (20, 60, 120):
        df[f"MA{n}"] = c.rolling(n).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI"] = 100 - 100 / (1 + rs.astype(float))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def trend_score(df: pd.DataFrame):
    """이동평균 기반 6점 만점 추세 점수와 근거."""
    last = df.iloc[-1]
    reasons = []
    score = 0

    checks = [
        (last["Close"] > last["MA20"], "종가가 20일선 위", "종가가 20일선 아래"),
        (last["Close"] > last["MA60"], "종가가 60일선 위", "종가가 60일선 아래"),
        (last["Close"] > last["MA120"], "종가가 120일선 위", "종가가 120일선 아래"),
        (last["MA20"] > last["MA60"], "20일선 > 60일선 (정배열)", "20일선 < 60일선"),
        (last["MA60"] > last["MA120"], "60일선 > 120일선 (정배열)", "60일선 < 120일선"),
    ]
    for ok, good, bad in checks:
        score += int(bool(ok))
        reasons.append(("✅ " + good) if ok else ("❌ " + bad))

    ma60_prev = df["MA60"].iloc[-21] if len(df) > 21 else float("nan")
    rising = last["MA60"] > ma60_prev
    score += int(bool(rising))
    reasons.append("✅ 60일선 상승 중" if rising else "❌ 60일선 하락 중")
    return score, reasons


def label_from_score(score: float):
    if score >= 5:
        return "상승장", "🟢", "#16a34a"
    if score >= 4:
        return "약한 상승", "🟢", "#65a30d"
    if score >= 3:
        return "횡보 / 혼조", "🟡", "#ca8a04"
    if score >= 2:
        return "약한 하락", "🔴", "#ea580c"
    return "하락장", "🔴", "#dc2626"


def pct_change(df: pd.DataFrame, days: int):
    if len(df) <= days:
        return None
    return (df["Close"].iloc[-1] / df["Close"].iloc[-1 - days] - 1) * 100


def fmt_pct(v):
    return "-" if v is None else f"{v:+.2f}%"


# ──────────────────────────────────────────────
# 차트
# ──────────────────────────────────────────────
def price_chart(df: pd.DataFrame, title: str, candle: bool = True, show_sub: bool = True):
    view = df.tail(250)
    rows = 3 if show_sub else 1
    heights = [0.6, 0.2, 0.2] if show_sub else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=heights)

    if candle and {"Open", "High", "Low"}.issubset(view.columns):
        fig.add_trace(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], name="가격",
            increasing_line_color="#dc2626", decreasing_line_color="#2563eb",  # 한국식: 상승=빨강
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="종가",
                                 line=dict(color="#111827")), row=1, col=1)

    for n, color in ((20, "#f59e0b"), (60, "#10b981"), (120, "#8b5cf6")):
        fig.add_trace(go.Scatter(x=view.index, y=view[f"MA{n}"], name=f"MA{n}",
                                 line=dict(width=1.5, color=color)), row=1, col=1)

    if show_sub:
        fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], name="RSI(14)",
                                 line=dict(color="#0ea5e9")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#2563eb", row=2, col=1)

        hist = view["MACD"] - view["MACD_SIGNAL"]
        fig.add_trace(go.Bar(x=view.index, y=hist, name="MACD 히스토그램",
                             marker_color=["#dc2626" if v >= 0 else "#2563eb" for v in hist]),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], name="MACD",
                                 line=dict(color="#111827", width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=view["MACD_SIGNAL"], name="Signal",
                                 line=dict(color="#f97316", width=1)), row=3, col=1)

    fig.update_layout(title=title, height=720 if show_sub else 420,
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0),
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ──────────────────────────────────────────────
# 종목 검색
# ──────────────────────────────────────────────
def resolve_ticker(query: str):
    """입력(종목명 / 6자리 코드 / 티커) → (표시이름, 티커). 못 찾으면 (None, None)."""
    q = query.strip()
    if not q:
        return None, None
    key = q.replace(" ", "").upper()

    # 1) 사전 정확 일치 (대소문자/공백 무시)
    for name, tk in STOCKS.items():
        if name.replace(" ", "").upper() == key:
            return name, tk

    # 2) 6자리 종목코드 → .KS / .KQ 순서로 시도
    if key.isdigit() and len(key) == 6:
        for suffix in (".KS", ".KQ"):
            if not load(key + suffix, "6mo").empty:
                return key, key + suffix

    # 3) 티커 직접 입력 (예: 005930.KS)
    if key.endswith((".KS", ".KQ")):
        return key, key

    # 4) 사전 부분 일치
    matches = [(n, t) for n, t in STOCKS.items() if key in n.replace(" ", "").upper()]
    if matches:
        return matches[0]

    # 5) yfinance 검색 (한국 거래소만)
    try:
        res = yf.Search(q, max_results=10).quotes
        for r in res:
            sym = r.get("symbol", "")
            if sym.endswith((".KS", ".KQ")):
                return r.get("shortname") or r.get("longname") or sym, sym
    except Exception:
        pass
    return None, None


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("📈 한국 주식 시장 추세 분석")
st.caption("야후 파이낸스 데이터 기반 · 이동평균선으로 판단한 참고용 지표이며 투자 권유가 아닙니다.")

tab_market, tab_stock = st.tabs(["🏛️ 시장 상태 (KOSPI / KOSDAQ)", "🔍 종목 추세 검색"])

# ── 시장 탭 ───────────────────────────────────
with tab_market:
    with st.spinner("지수 데이터 불러오는 중..."):
        idx_data = {}
        for name, tk in INDICES.items():
            d = load(tk)
            if not d.empty and len(d) > 130:
                idx_data[name] = add_indicators(d)

    if not idx_data:
        st.error("지수 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
    else:
        scores = {n: trend_score(d) for n, d in idx_data.items()}
        avg = sum(s for s, _ in scores.values()) / len(scores)
        label, icon, color = label_from_score(round(avg))

        st.markdown(
            f"<div style='padding:22px;border-radius:14px;background:{color}20;"
            f"border:2px solid {color};text-align:center'>"
            f"<div style='font-size:15px;color:#6b7280'>현재 한국 시장 종합 판단</div>"
            f"<div style='font-size:42px;font-weight:800;color:{color}'>{icon} {label}</div>"
            f"<div style='font-size:14px;color:#6b7280'>추세 점수 평균 {avg:.1f} / 6</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        cols = st.columns(len(idx_data))
        for col, (name, d) in zip(cols, idx_data.items()):
            sc, reasons = scores[name]
            lb, ic, _ = label_from_score(sc)
            last = d["Close"].iloc[-1]
            chg1d = pct_change(d, 1)
            with col:
                st.subheader(f"{name}  {ic} {lb}")
                st.metric("현재 지수", f"{last:,.2f}", fmt_pct(chg1d))
                m1, m2, m3 = st.columns(3)
                m1.metric("1개월", fmt_pct(pct_change(d, 21)))
                m2.metric("3개월", fmt_pct(pct_change(d, 63)))
                m3.metric("6개월", fmt_pct(pct_change(d, 126)))
                hi, lo = d["Close"].tail(252).max(), d["Close"].tail(252).min()
                st.caption(f"52주 고점 대비 {(last / hi - 1) * 100:+.1f}% · 52주 저점 대비 {(last / lo - 1) * 100:+.1f}%")
                with st.expander(f"판단 근거 ({sc}/6점)"):
                    for r in reasons:
                        st.write(r)

        for name, d in idx_data.items():
            st.plotly_chart(price_chart(d, f"{name} 지수", candle=False, show_sub=False),
                            use_container_width=True)

        st.info("**판단 기준** (6점 만점): 종가 > 20/60/120일선 (각 1점), 20일선 > 60일선, "
                "60일선 > 120일선, 60일선 상승 중 (각 1점).  \n"
                "5~6점 상승장 · 4점 약한 상승 · 3점 횡보 · 2점 약한 하락 · 0~1점 하락장")

# ── 종목 탭 ───────────────────────────────────
with tab_stock:
    query = st.text_input("종목명 또는 종목코드를 입력하세요",
                          placeholder="예: 삼성전자, SK하이닉스, 에코프로, 005930")
    st.caption("대표 종목은 이름으로, 그 외 종목은 6자리 종목코드로 검색하면 가장 정확합니다.")

    if query:
        with st.spinner("검색 중..."):
            name, ticker = resolve_ticker(query)

        if not ticker:
            st.warning(f"'{query}' 종목을 찾지 못했습니다. 종목명을 다시 확인하거나 6자리 종목코드로 검색해 보세요.")
        else:
            df = load(ticker)
            if df.empty or len(df) < 130:
                st.error("데이터가 부족해 분석할 수 없습니다. (상장 6개월 미만이거나 데이터 조회 실패)")
            else:
                df = add_indicators(df)
                score, reasons = trend_score(df)
                lb, ic, color = label_from_score(score)
                last = df.iloc[-1]

                st.markdown(
                    f"<div style='padding:18px;border-radius:14px;background:{color}20;"
                    f"border:2px solid {color};text-align:center'>"
                    f"<div style='font-size:15px;color:#6b7280'>{name} ({ticker})</div>"
                    f"<div style='font-size:36px;font-weight:800;color:{color}'>{ic} {lb}</div>"
                    f"<div style='font-size:14px;color:#6b7280'>추세 점수 {score} / 6</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.write("")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("현재가", f"{last['Close']:,.0f}원", fmt_pct(pct_change(df, 1)))
                c2.metric("1개월", fmt_pct(pct_change(df, 21)))
                c3.metric("3개월", fmt_pct(pct_change(df, 63)))
                c4.metric("6개월", fmt_pct(pct_change(df, 126)))

                hi, lo = df["Close"].tail(252).max(), df["Close"].tail(252).min()
                rsi = last["RSI"]
                macd_up = last["MACD"] > last["MACD_SIGNAL"]

                d1, d2, d3 = st.columns(3)
                d1.metric("52주 고점 대비", f"{(last['Close'] / hi - 1) * 100:+.1f}%")
                d2.metric("52주 저점 대비", f"{(last['Close'] / lo - 1) * 100:+.1f}%")
                rsi_txt = "과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립"
                d3.metric("RSI(14)", f"{rsi:.1f}", rsi_txt, delta_color="off")

                left, right = st.columns(2)
                with left:
                    st.markdown("**추세 판단 근거**")
                    for r in reasons:
                        st.write(r)
                with right:
                    st.markdown("**보조 지표**")
                    st.write("✅ MACD가 시그널선 위 (상승 모멘텀)" if macd_up
                             else "❌ MACD가 시그널선 아래 (하락 모멘텀)")
                    st.write(f"{'⚠️' if rsi >= 70 or rsi <= 30 else 'ℹ️'} RSI {rsi:.1f} → {rsi_txt}")
                    vol = df["Volume"]
                    if vol.tail(20).mean() > 0:
                        ratio = vol.iloc[-1] / vol.tail(20).mean()
                        st.write(f"ℹ️ 오늘 거래량은 20일 평균의 {ratio:.1f}배")

                st.plotly_chart(price_chart(df, f"{name} 주가 추세"), use_container_width=True)

st.divider()
st.caption("⚠️ 본 앱의 결과는 과거 가격 기반의 기술적 분석이며 미래 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")
