import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="주식 추세 분석기", page_icon="📈", layout="wide")

st.title("📈 주식 차트 · 상승/하락 추세 분석")
st.caption("Yahoo Finance 데이터를 가져와 이동평균선과 기간 수익률로 추세를 요약합니다. 투자 판단을 대신하지 않습니다.")

with st.sidebar:
    st.header("분석 설정")
    ticker = st.text_input("종목 코드", value="AAPL",
        help="미국 주식: AAPL, TSLA / 한국 주식: 삼성전자 005930.KS, 코스닥 예시 035720.KQ")
    period_label = st.selectbox("조회 기간", ["1개월", "3개월", "6개월", "1년", "2년", "5년"], index=3)
    interval_label = st.selectbox("차트 간격", ["일봉", "주봉"], index=0)
    show_volume = st.checkbox("거래량 표시", value=True)
    run = st.button("분석하기", type="primary", use_container_width=True)

period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y", "5년": "5y"}
interval_map = {"일봉": "1d", "주봉": "1wk"}

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = True
if run:
    st.session_state.analysis_run = True

if st.session_state.analysis_run:
    symbol = ticker.strip().upper()
    if not symbol:
        st.error("종목 코드를 입력해 주세요.")
        st.stop()

    with st.spinner(f"{symbol} 데이터를 불러오는 중..."):
        try:
            data = yf.download(
                symbol,
                period=period_map[period_label],
                interval=interval_map[interval_label],
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:
            st.error(f"데이터를 가져오지 못했습니다: {e}")
            st.stop()

    if data is None or data.empty:
        st.error("데이터가 없습니다. 종목 코드가 맞는지 확인해 주세요. 한국 종목은 보통 .KS 또는 .KQ가 필요합니다.")
        st.stop()

    # yfinance 버전에 따라 컬럼이 MultiIndex로 올 수 있어 평탄화
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Close"}
    if not required.issubset(set(data.columns)):
        st.error("가격 데이터 형식이 예상과 다릅니다. 잠시 후 다시 시도해 주세요.")
        st.stop()

    data = data.dropna(subset=["Close"]).copy()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["DailyReturn"] = data["Close"].pct_change()
    data["RSI"] = None
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    data["RSI"] = 100 - (100 / (1 + rs))

    close = float(data["Close"].iloc[-1])
    first_close = float(data["Close"].iloc[0])
    total_return = (close / first_close - 1) * 100
    ma20 = data["MA20"].iloc[-1]
    ma50 = data["MA50"].iloc[-1]
    ma200 = data["MA200"].iloc[-1]
    rsi = data["RSI"].iloc[-1]

    # 단순 규칙 기반 분류: 수익률 + 이동평균선 배열 + 단기/중기 이동평균선 방향
    score = 0
    reasons = []
    if total_return > 2:
        score += 1
        reasons.append(f"조회 기간 수익률이 {total_return:.2f}%로 플러스입니다.")
    elif total_return < -2:
        score -= 1
        reasons.append(f"조회 기간 수익률이 {total_return:.2f}%로 마이너스입니다.")
    else:
        reasons.append(f"조회 기간 수익률이 {total_return:.2f}%로 큰 방향성이 뚜렷하지 않습니다.")

    if pd.notna(ma20) and pd.notna(ma50):
        if ma20 > ma50:
            score += 1
            reasons.append("20일 이동평균선이 50일 이동평균선 위에 있습니다.")
        else:
            score -= 1
            reasons.append("20일 이동평균선이 50일 이동평균선 아래에 있습니다.")

    if pd.notna(ma50) and pd.notna(ma200):
        if ma50 > ma200:
            score += 1
            reasons.append("50일 이동평균선이 200일 이동평균선 위에 있습니다.")
        else:
            score -= 1
            reasons.append("50일 이동평균선이 200일 이동평균선 아래에 있습니다.")

    if score >= 2:
        trend, color, summary = "상승 추세 우세", "🟢", "가격 흐름과 이동평균선이 전반적으로 긍정적인 쪽입니다."
    elif score <= -2:
        trend, color, summary = "하락 추세 우세", "🔴", "가격 흐름과 이동평균선이 전반적으로 부정적인 쪽입니다."
    else:
        trend, color, summary = "혼조 · 방향 불명확", "🟡", "상승과 하락 신호가 섞여 있어 한 방향으로 단정하기 어렵습니다."

    c1, c2, c3 = st.columns(3)
    c1.metric("최근 종가", f"{close:,.2f}", f"{total_return:+.2f}% (조회 기간)")
    c2.metric("규칙 기반 추세", f"{color} {trend}")
    c3.metric("RSI(14)", f"{float(rsi):.1f}" if pd.notna(rsi) else "데이터 부족",
              "70 이상은 과열 가능성, 30 이하는 과매도 가능성 참고")

    st.subheader("가격 차트")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="종가", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50", line=dict(width=1.4)))
    if data["MA200"].notna().any():
        fig.add_trace(go.Scatter(x=data.index, y=data["MA200"], name="MA200", line=dict(width=1.4)))
    fig.update_layout(
        height=520, xaxis_title="날짜", yaxis_title="조정 종가",
        hovermode="x unified", legend_title="선",
        margin=dict(l=10, r=10, t=25, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_volume and "Volume" in data.columns:
        st.subheader("거래량")
        vol_fig = go.Figure(go.Bar(x=data.index, y=data["Volume"], name="거래량"))
        vol_fig.update_layout(height=220, xaxis_title="날짜", yaxis_title="거래량",
                              margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(vol_fig, use_container_width=True)

    st.subheader("왜 이렇게 판단했을까?")
    for reason in reasons:
        st.write(f"- {reason}")

    st.info("이 분류는 조회 기간 수익률과 이동평균선만 사용하는 간단한 참고 지표입니다. 미래 가격을 예측하거나 매수·매도 신호를 보장하지 않습니다. 데이터는 Yahoo Finance를 통해 제공되며 지연·누락될 수 있습니다.")

    with st.expander("최근 데이터 보기"):
        display = data[["Close", "MA20", "MA50", "MA200", "Volume"]].tail(20).copy()
        display.index = display.index.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True)
