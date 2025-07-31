import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

def main():
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

    df = pd.read_csv('dataset/df_total.csv', encoding='utf8')
    new_data = []

    if st.button("수요 예측"):
        try:
            model = joblib.load(modelfile)
            input_data = np.array([[temp, sun, rain, wind, humid, year-2018]])
            prediction = model.predict(input_data)[0]
            new_data.append((temp, prediction))
            st.write(f"당일 전력 수요량 예측 결과: {prediction:.2f} MW")
            
        except FileNotFoundError :
            st.error(f"오류: '{modelfile}' 파일을 찾을 수 없습니다. 모델 파일이 앱과 같은 디렉터리에 있는지 확인해주세요.")
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다.: {e}")

    st.title("평균기온 vs. 전력수요총합")
    st.title("예측데이터 표기")
    fig = plt.figure()

    df_new = pd.DataFrame({
        "평균기온": [climate for climate, demand in new_data],
        "전력수요총합": [demand for climate, demand in new_data]
    })
    sns.scatterplot(df, x="평균기온", y="전력수요총합")
    sns.scatterplot(df_new, x="평균기온", y="전력수요총합")
    st.pyplot(fig)

    st.title("데이터 샘플")
    st.dataframe( df.head() )

    st.title("상관계수 히트맵")
    corr_heatmap = plt.figure()
    sns.heatmap(df.iloc[:, 1:].corr(method='spearman'), annot=True)
    st.pyplot(corr_heatmap)

if __name__ == "__main__" : 
    main()