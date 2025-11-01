# 🏡 Gurgaon House Price Prediction using Machine Learning

A data-driven **Machine Learning project** that predicts house prices in **Gurgaon, Haryana** based on multiple property features such as location, total rooms, income levels, and demographics.  

This project demonstrates the **end-to-end ML workflow** — from data preprocessing and model training to evaluation and deployment — aimed at solving a **real-world real-estate pricing problem**.

---

## 🚀 Project Overview

Real estate prices in Gurgaon fluctuate due to several factors like location, population, and proximity to amenities.  
This project uses **Python and ML algorithms** to build a predictive model that can estimate the median house value given property and demographic features.

---

## 🧩 Features

✅ Data Cleaning and Preprocessing  
✅ Stratified Sampling for train/test split  
✅ Feature Scaling and One-Hot Encoding  
✅ Model Training (Linear Regression, Decision Tree, Random Forest)  
✅ Cross Validation & RMSE Evaluation  
✅ Saved Model Pipeline for Inference  
✅ Automated Prediction on new data (input.csv → output.csv)

---

## 📂 Dataset

The dataset (modified from the California Housing dataset) includes property and demographic data of **Gurgaon, Haryana**.

**Columns used:**
- `longitude`  
- `latitude`  
- `housing_median_age`  
- `total_rooms`  
- `total_bedrooms`  
- `population`  
- `households`  
- `median_income`  
- `median_house_value`  
- `ocean_proximity` (renamed to represent Gurgaon regions/sectors)

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn** (for EDA)
- **Joblib** (for model persistence)

---

## 🧠 ML Algorithms Used

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor** (Best Performer)

---

## ⚙️ How It Works

1. **Training Phase**
   - Load and preprocess the dataset
   - Apply transformations using pipelines
   - Train the Random Forest model
   - Save trained model & pipeline using `joblib`

2. **Inference Phase**
   - Load trained model and preprocessing pipeline
   - Transform new data (from `input.csv`)
   - Predict and save results to `output.csv`

---

## 📈 Model Performance

| Model | RMSE (Approx.) | Remarks |
|-------|----------------|----------|
| Linear Regression | 70,000 | Simple baseline |
| Decision Tree | 45,000 | Overfitted slightly |
| Random Forest | **32,000** | Best performer |

---

## 🧾 How to Run

```bash
# Clone the repo
git clone https://github.com/krishnatanwar/gurgaon-house-price-prediction-ml.git

# Navigate to project directory
cd gurgaon-house-price-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
