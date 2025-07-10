# recipe_pipeline.py
import sqlite3
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing import image

# Configuration
MODEL_PATH = '../models/food_knower.h5'
DB_PATH = '../data/database/recipes.db'
CLASS_LABEL_PATH = '../models/class_indices.json'

# Load class labels
try:
    with open(CLASS_LABEL_PATH) as f:
        class_indices = json.load(f)
        CLASS_LABELS = {v: k for k, v in class_indices.items()}  # Ensure underscores
except FileNotFoundError:
    raise SystemExit(f"Error: Class labels not found at {CLASS_LABEL_PATH}. Train model first!")

# Image preprocessing
def preprocess_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Ingredient prediction
def predict_ingredients(img_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)[0]
    
    if len(predictions) != len(CLASS_LABELS):
        raise ValueError("Model output doesn't match class labels. Retrain model!")
    
    top_indices = np.argsort(predictions)[-3:][::-1]  # Top 3, highest first
    return [CLASS_LABELS[i] for i in top_indices]

# Database search
def search_recipes(ingredients):
    conn = sqlite3.connect(DB_PATH)
    query = ' AND '.join([f'ner:{ing}' for ing in ingredients])
    
    return conn.execute('''
        SELECT title, directions, link 
        FROM recipes_fts
        WHERE recipes_fts MATCH ?
        ORDER BY (LENGTH(ner) - LENGTH(REPLACE(ner, ' ', ''))) ASC
        LIMIT 3
    ''', (query,)).fetchall()

def image_to_recipes(img_path):
    print(f"\nProcessing image: {img_path}")
    ingredients = predict_ingredients(img_path)
    print(f"Detected ingredients: {', '.join(ingredients)}")
    
    recipes = search_recipes(ingredients)
    if not recipes:
        return print("No recipes found!")
    
    print("\nTop 3 recipes:")
    for i, (title, directions, link) in enumerate(recipes, 1):
        print(f"\n{i}. {title}")
        print(f"   Link: {link}")
        print(f"   Directions: {directions[:100]}...")

if __name__ == "__main__":
    image_path = '../data/Food-Ingredient-Recognition-1/manual_test/banana.png' 
    image_to_recipes(image_path)