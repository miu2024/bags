import streamlit as st
import os
from fastai.vision.all import *
import pathlib
import sys

# 根据不同的操作系统设置正确的pathlib.Path
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path,"bags.pkl")

# 加载模型
learn_inf = load_learner(model_path)

# 恢复pathlib.Path的原始值
if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp

st.title("包包识别分类")
st.write("上传一张图片，应用将预测对应的标签。")

# 允许用户上传图片
uploaded_file = st.file_uploader("选择一张图片...",
                                 type=["jpg","jpeg","png"
                                 ])

# 如果用户已上传图片
if uploaded_file is not None:
    # 显示上传的图片
    image = PILImage.create(uploaded_file)
    st.image(image,caption="您上传的图片",
             use_column_width=true)

    
    # 获取预测的标签
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"识别包包类型: {pred}; 概率: {probs[pred_idx]:.04f}")
import streamlit as st
import os
from fastai.vision.all import *
import pathlib
import sys
import pandas as pd
import random


if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath


path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "1.pkl")


learn_inf = load_learner(model_path)


if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp


def load_bags(pkl_file):
    with open(pkl_file, 'rb') as file:
         bags= pickle.load(file)
    return bags
st.title("包包推荐系统")

import pandas as pd
import random
import streamlit as st


def load_bags_from_excel(filename):
    df = pd.read_excel(filename)
    bags = df['bag'].tolist()
    return bags

bags = load_bags_from_excel("e.xlsx")
initial_bags = random.sample(bags, 3)
ratings = {}

# 遍历并显示评分组件
for i, bag in enumerate(initial_bags):
    st.write(f"{i+1}. {bag}")
    rating = st.slider(f"Rate this bag ({i+1})", 1, 5)
    ratings[bag] = rating

# 创建一个提交评分的按钮
if st.button("提交评分"):
    
    rated_bags = set(initial_bags)
    remaining_bags = [bag for bag in bags if bag not in rated_bags]
    recommended_bags = random.sample(remaining_bags,1)
    

    recommended_ratings = [st.slider(bag, 1, 5) for bag in recommended_bags]
    satisfaction = sum(recommended_ratings) / len(recommended_ratings)
 



if st.button("提交推荐评分"):
    avg_recommended_score = sum(ratings.values()) / len(ratings)
    percentage_score = (avg_recommended_score / 5) * 100

    # 显示结果
    st.write(f"You rated the recommended bags {percentage_score:.2f}% of the total possible score.")