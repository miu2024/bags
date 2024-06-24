
import streamlit as st
import os
from fastai.vision.all import *
import pathlib
import sys
import pandas as pd
import random
import pickle

def load_bags(pkl_file):
    with open(pkl_file, 'rb') as file:
        bags = pickle.load(file)
    return bags

def load_bags_from_excel(filename):
    df = pd.read_excel(filename)
    bags = df['bag'].tolist()
    return bags

# Set the correct pathlib.Path based on the operating system
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# Get the directory of the current file
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "bags.pkl")
learn_inf = load_learner(model_path)

# Restore the original pathlib.Path
if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp

# Streamlit app for image classification
st.title("包包识别分类")
st.write("上传一张图片，应用将预测对应的标签。")

# Allow user to upload an image
uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

# If user has uploaded an image
if uploaded_file is not None:
    # Display the uploaded image
    image = PILImage.create(uploaded_file)
    st.image(image, caption="您上传的图片", use_column_width=True)
    
    # Get the predicted label
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"识别包包类型: {pred}; 概率: {probs[pred_idx]:.04f}")

# Streamlit app for bag recommendation
st.title("包包推荐系统")
bags = load_bags_from_excel("e.xlsx")
initial_bags = random.sample(bags, 3)
ratings = {}

# Loop through and display rating components
for i, bag in enumerate(initial_bags):
    st.write(f"{i+1}. {bag}")
    rating = st.slider(f"Rate this bag ({i+1})", 1, 5)
    ratings[bag] = rating

# Create a button to submit ratings
if st.button("提交评分"):
    rated_bags = set(initial_bags)
    remaining_bags = [bag for bag in bags if bag not in rated_bags]
    recommended_bags = random.sample(remaining_bags, 1)

    recommended_ratings = [st.slider(bag, 1, 5) for bag in recommended_bags]
    satisfaction = sum(recommended_ratings) / len(recommended_ratings)

if st.button("提交推荐评分"):
    avg_recommended_score = sum(ratings.values()) / len(ratings)
    percentage_score = (avg_recommended_score / 5) * 100

    # Display the result
    st.write(f"You rated the recommended bags {percentage_score:.2f}% of the total possible score.")
