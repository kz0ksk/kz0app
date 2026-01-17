import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. モデルの準備（アプリ起動時に1回だけ実行）
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# 2. Web画面の構成
st.title("アヤメの種類 判定アプリ")
st.write("スライダーを動かして、アヤメのサイズを入力してください。")

# サイドバーに入力スライダーを作成
sepal_l = st.sidebar.slider("がく片の長さ (cm)", 4.0, 8.0, 5.0)
sepal_w = st.sidebar.slider("がく片の幅 (cm)", 2.0, 5.0, 3.0)
petal_l = st.sidebar.slider("花弁の長さ (cm)", 1.0, 7.0, 1.5)
petal_w = st.sidebar.slider("花弁の幅 (cm)", 0.1, 3.0, 0.2)

# 3. ボタンが押されたら予測を実行
if st.button("判定する"):
    prediction = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    species = iris.target_names[prediction][0]
    
    st.success(f"結果: これは「{species}」です！")