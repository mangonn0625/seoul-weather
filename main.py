import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------
# 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="서울 100년 기온 변화",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
MIN_DAYS_PER_YEAR = 300  # 이만큼의 관측일이 없는 해는 '반쪽 데이터'로 보고 제외


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def make_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            최저기온=("최저기온", "mean"),
            최고기온=("최고기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )
    # 관측일이 너무 적은 해(첫 해, 마지막 해 등)는 제외해 왜곡을 막음
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS_PER_YEAR].reset_index(drop=True)
    return yearly


def linear_trend(x: np.ndarray, y: np.ndarray):
    """단순 선형회귀로 추세선과 10년당 상승폭을 계산"""
    coef = np.polyfit(x, y, 1)
    trend_y = coef[0] * x + coef[1]
    slope_per_decade = coef[0] * 10
    return trend_y, slope_per_decade


# ---------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------
with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
    raw_df = load_data(DATA_URL)
    yearly_df = make_yearly(raw_df)

st.title("🌡️ 서울, 100년의 기온 변화")
st.caption(
    f"기상청 관측 데이터 기준 · {int(yearly_df['연도'].min())}년 ~ "
    f"{int(yearly_df['연도'].max())}년 · 관측지점: 서울(108)"
)

# ---------------------------------------------------
# 사이드바 - 옵션
# ---------------------------------------------------
st.sidebar.header("⚙️ 옵션")

year_min = int(yearly_df["연도"].min())
year_max = int(yearly_df["연도"].max())

year_range = st.sidebar.slider(
    "살펴볼 연도 범위",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)

show_trend = st.sidebar.checkbox("선형 추세선 표시", value=True)
show_ma = st.sidebar.checkbox("10년 이동평균선 표시", value=True)
show_range = st.sidebar.checkbox("최저·최고 기온 범위 표시", value=False)

filtered = yearly_df[
    (yearly_df["연도"] >= year_range[0]) & (yearly_df["연도"] <= year_range[1])
].copy()

# ---------------------------------------------------
# 핵심 지표 카드
# ---------------------------------------------------
first_decade = filtered[filtered["연도"] < filtered["연도"].min() + 10]["평균기온"].mean()
last_decade = filtered[filtered["연도"] > filtered["연도"].max() - 10]["평균기온"].mean()
change = last_decade - first_decade

col1, col2, col3 = st.columns(3)
col1.metric(
    f"처음 10년 평균 ({filtered['연도'].min()}년대)",
    f"{first_decade:.1f} °C",
)
col2.metric(
    f"최근 10년 평균 ({filtered['연도'].max() - 9}~{filtered['연도'].max()}년)",
    f"{last_decade:.1f} °C",
)
col3.metric(
    "변화량",
    f"{change:+.1f} °C",
    delta=f"{change:+.1f} °C",
)

st.divider()

# ---------------------------------------------------
# 메인 그래프 - 연평균 기온 변화
# ---------------------------------------------------
st.subheader("연평균 기온 변화 추이")

fig = go.Figure()

if show_range:
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["최고기온"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["최저기온"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,159,64,0.15)",
            name="최저~최고 기온 범위",
            hoverinfo="skip",
        )
    )

fig.add_trace(
    go.Scatter(
        x=filtered["연도"],
        y=filtered["평균기온"],
        mode="lines+markers",
        name="연평균 기온",
        line=dict(color="#ff6b35", width=2),
        marker=dict(size=4),
    )
)

if show_ma:
    filtered["이동평균"] = filtered["평균기온"].rolling(10, center=True, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["이동평균"],
            mode="lines",
            name="10년 이동평균",
            line=dict(color="#3d5a80", width=3, dash="dash"),
        )
    )

if show_trend and len(filtered) > 1:
    trend_y, slope = linear_trend(filtered["연도"].values, filtered["평균기온"].values)
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=trend_y,
            mode="lines",
            name="선형 추세선",
            line=dict(color="#8d99ae", width=2, dash="dot"),
        )
    )
    st.info(f"📈 선택한 기간 동안 서울의 연평균 기온은 **10년마다 약 {slope:+.2f}°C** 변해왔어요.")

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# 원본 데이터 확인
# ---------------------------------------------------
with st.expander("📄 연도별 데이터 표 보기"):
    st.dataframe(
        filtered[["연도", "평균기온", "최저기온", "최고기온", "관측일수"]]
        .sort_values("연도", ascending=False)
        .style.format({"평균기온": "{:.1f}", "최저기온": "{:.1f}", "최고기온": "{:.1f}"}),
        use_container_width=True,
    )

with st.expander("ℹ️ 데이터 안내"):
    st.markdown(
        f"""
        - 출처: 기상청 서울(지점번호 108) 일별 기온 관측 데이터
        - 원본 컬럼: 날짜, 지점, 평균기온, 최저기온, 최고기온
        - 연평균 기온은 하루 단위 평균기온을 연도별로 평균 낸 값이에요.
        - 한 해의 관측일수가 {MIN_DAYS_PER_YEAR}일 미만인 해(자료가 시작·종료되는 해)는
          왜곡을 막기 위해 그래프에서 제외했어요.
        """
    )
