import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# 1. モデルの準備
@st.cache_resource # アプリが再実行されてもモデル学習をスキップして高速化
def load_model():
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    return iris, model

iris, model = load_model()

# 2. Web画面の構成
st.title("アヤメの種類 判定アプリ 🌸")
st.write("スライダーを動かして、サイズを入力してください。下の図形がリアルタイムで変化します。")

# サイドバーに入力スライダーを作成
st.sidebar.header("計測データの入力")
sepal_l = st.sidebar.slider("がく片の長さ (cm)", 4.0, 8.0, 5.8)
sepal_w = st.sidebar.slider("がく片の幅 (cm)", 2.0, 5.0, 3.0)
petal_l = st.sidebar.slider("花弁の長さ (cm)", 1.0, 7.0, 4.3)
petal_w = st.sidebar.slider("花弁の幅 (cm)", 0.1, 3.0, 1.3)

# --- リアルタイム楕円描画セクション ---
st.subheader("入力サイズの視覚化 (楕円)")

# 角度を作成
theta = np.linspace(0, 2 * np.pi, 100)

# グラフの作成
fig, ax = plt.subplots(figsize=(5, 5))

# 花弁(Petal)の楕円（内側・濃い色）
px = (petal_l / 2) * np.cos(theta)
py = (petal_w / 2) * np.sin(theta)
ax.fill(px, py, color="magenta", alpha=0.5, label="Petal (花弁)")

# がく片(Sepal)の楕円（外側・薄い色）
sx = (sepal_l / 2) * np.cos(theta)
sy = (sepal_w / 2) * np.sin(theta)
ax.fill(sx, sy, color="green", alpha=0.2, label="Sepal (がく片)")

# グラフ設定
ax.set_aspect('equal')
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.legend(loc='upper right')
ax.axis('off') # 枠線を消して図形を目立たせる

st.pyplot(fig)
# -----------------------------------

# 3. 判定ボタン
if st.button("このサイズで種類を判定する"):
    prediction = model.predict([[sepal_l, sepal_w, petal_l, petal_w]])
    species = iris.target_names[prediction][0]
    
    st.balloons() # お祝いのアニメーション
    st.success(f"結果: これは「{species}」です！")