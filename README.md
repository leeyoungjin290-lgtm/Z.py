# 주식 추세 분석기 (Streamlit)

Yahoo Finance 데이터를 가져와 주가 차트, 이동평균선(MA20/MA50/MA200), 거래량, RSI(14), 규칙 기반 상승/하락 추세 요약을 보여주는 간단한 앱입니다.

## 1. GitHub에 올리기
1. GitHub에서 새 저장소(repository)를 만듭니다.
2. 이 폴더의 `app.py`, `requirements.txt`, `README.md`를 저장소에 업로드합니다.

## 2. Streamlit Community Cloud 배포
1. https://share.streamlit.io/ 에 접속해 GitHub 계정으로 로그인합니다.
2. **Create app**을 누릅니다.
3. 저장소, 브랜치(`main`), 메인 파일(`app.py`)을 선택합니다.
4. Deploy를 누릅니다. `requirements.txt`에 적힌 라이브러리가 자동 설치됩니다.

## 3. 내 컴퓨터에서 실행
```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## 종목 코드 예시
- 미국 주식: `AAPL`, `MSFT`, `TSLA`
- 한국 거래소(KOSPI): `005930.KS` (삼성전자)
- 코스닥(KOSDAQ): 종목에 따라 `035720.KQ`처럼 `.KQ` 접미사를 사용합니다.

## 분석 방식과 주의점
- 조회 기간 수익률, 20일/50일/200일 이동평균선의 상대 위치를 합쳐 상승 우세/하락 우세/혼조로 표시합니다.
- 이는 설명용 규칙 기반 요약일 뿐, 검증된 투자전략이나 미래 예측이 아닙니다.
- `yfinance`는 Yahoo Finance 데이터를 이용하는 비공식 라이브러리이며 데이터 이용 조건, 지연 또는 누락 가능성을 확인하세요.
