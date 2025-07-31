import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 타이틀
st.title("전력 수요량 예측 모델")
st.write("예상 기상 데이터를 통해 당일 전력 수요를 예상해 봅시다.")

temp = st.slider("기온(섭씨)", min_value = -30.0, max_value=45.0, value=25.0)
rain = st.slider("강수량(mm)", min_value = 0.0, max_value=500.0, value=0.0)
wind = st.slider("평균풍속 (m/s)", min_value = 0.0, max_value=100.0, value=10.0)
humid = st.slider("습도 (%)", min_value=0.0, max_value=100.0, value=50.0)
sun = st.slider("일사합 (MJ/m^2)", min_value=0.0, max_value=50.0, value=10.0)
year = st.slider("예상 연도", min_value=2018, max_value=2050, value=2025)

modelfile = "model_elec_demand.pkl"
if st.button("수요 예측"):
    try:
        model = joblib.load("model_elec_demand.pkl")
        input_data = np.array([[temp, sun, rain, wind, humid, year-2018]])
        prediction = model.predict(input_data)[0]
        
        st.write(f"당일 전력 수요량 예측 결과: {prediction:.2f} MW")
        
    except FileNotFoundError :
        st.error("오류: 'model_elec_demand.pkl' 파일을 찾을 수 없습니다. 모델 파일이 앱과 같은 디렉터리에 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다.: {e}")