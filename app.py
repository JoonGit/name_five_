import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
# CSV 파일을 로드하고 ds 컬럼을 datetime 형식으로 변환합니다.
df = pd.read_csv('./data/sunspots_for_prophet.csv')
df['ds'] = pd.to_datetime(df['ds'])

st.subheader("📄 데이터 미리보기")
st.dataframe(df.head())

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
# 11년 주기의 태양 활동 seasonality를 추가하여 학습합니다.
model = Prophet()
model.add_seasonality(name='sunspot_cycle', period=365.25 * 11, fourier_order=5)
model.fit(df)

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
# 향후 30년(periods=30)간 연 단위(freq='Y') 예측을 수행합니다.
future = model.make_future_dataframe(periods=30, freq='Y')
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1, use_container_width=True)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2, use_container_width=True)

# ----------------------------------
# [5] 커스텀 시각화: 실제값 vs 예측값 + 신뢰구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))

# 실제값: 파란색 실선 + 마커
ax.plot(df["ds"], df["y"], label="Actual", color="blue", marker='o', markersize=4, alpha=0.7)

# 예측값: 빨간색 점선 (마커 제거)
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="red", linestyle='--')

# 신뢰구간 채우기
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="pink", alpha=0.3, label="Confidence Interval")

ax.set_title("Sunspots: Actual vs. Predicted")
ax.set_xlabel("Year")
ax.set_ylabel("Sun Activity")
ax.legend(loc='upper left')
ax.grid(True)

st.pyplot(fig3, use_container_width=True)

# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

# df와 forecast를 'ds' 기준으로 병합하여 오차(residual) 계산
merged = pd.merge(df, forecast[['ds', 'yhat']], on='ds')
merged['residual'] = merged['y'] - merged['yhat']

fig4, ax2 = plt.subplots(figsize=(14, 4))

# 잔차 시계열 시각화
ax2.plot(merged["ds"], merged["residual"], color="purple", label="Residual", marker='o', markersize=4, alpha=0.7)
ax2.axhline(0, color='black', linestyle='--', linewidth=1) # 기준선 0

ax2.set_title("Residuals (Actual - Predicted)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Error")
ax2.legend()
ax2.grid(True)

st.pyplot(fig4, use_container_width=True)

# ----------------------------------
# [7] 잔차 통계 요약 출력
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
st.write(merged["residual"].describe())