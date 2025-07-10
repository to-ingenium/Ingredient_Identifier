# 🥘 Ingredient Identifier with Recipe Recommender

This project is a hands-on exploration of neural networks, specifically focused on **image classification** using deep learning. The goal is to detect **raw ingredients** from images and recommend relevant **recipes** using a local database.

## 🚀 Project Summary

- 🔍 **Model**: A custom-trained [MobileNetV2](https://arxiv.org/abs/1801.04381) image classifier.
- 🥦 **Input**: Photos of raw ingredients (e.g., tomatoes, eggs, garlic).
- 🍽️ **Output**: The top 5 recipe suggestions pulled from a local SQLite database based on the predicted ingredient.
- 📚 **Purpose**: Built to deepen understanding of image classification and neural network pipelines, from model training to app logic integration.

## 🧠 Core Technologies

- **Python**
- **TensorFlow / Keras** (MobileNetV2 transfer learning)
- **NumPy / Pandas**
- **SQLite3** (for recipe storage and querying)



## 🧪 How It Works

1. The user provides an image of a raw ingredient.
2. The trained MobileNetV2 model classifies the ingredient.
3. A query is made to a local SQLite database for matching recipes.
4. The top 5 relevant recipes are returned to the user.

## ✅ Goals

- Learn and apply **transfer learning** using MobileNetV2.
- Understand the full **ML pipeline**: data → training → inference → integration.
- Build a functional mini-app that connects **computer vision** with **practical utility**.

## 📌 Future Plans

- Add a simple web or desktop UI.
- Expand the dataset.
- Integrate multiple ingredients per image (multi-label prediction).
- Deploy as a lightweight app or API.

