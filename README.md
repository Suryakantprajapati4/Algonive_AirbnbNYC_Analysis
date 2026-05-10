# 🏠 Airbnb NYC — Data Analysis & Price Prediction

<p align="center">
  <img src="New_York_City_.png" alt="NYC Banner" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

> **Algonive Internship | Data Analytics Domain | Task 2**

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Author](#author)

---

## 🎯 About the Project

This project is part of the **Algonive Data Analytics Internship — Task 2**.

The goal is to analyze **Airbnb NYC 2019 listings data** to:
- Understand what factors affect listing prices
- Identify trends across different boroughs and room types
- Build a machine learning model to **predict listing prices**
- Create meaningful visualizations and an interactive dashboard

This is a complete end-to-end Data Science project covering data cleaning, EDA, feature engineering, ML modeling, and visualization.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| **Name** | New York City Airbnb Open Data |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) |
| **File** | AB_NYC_2019.csv |
| **Total Listings** | ~48,895 |
| **Total Features** | 16 columns |
| **Year** | 2019 |

### 📌 Key Columns

| Column | Description |
|--------|-------------|
| `name` | Name of the listing |
| `neighbourhood_group` | Borough (Manhattan, Brooklyn, etc.) |
| `neighbourhood` | Specific neighbourhood |
| `latitude / longitude` | Geographic coordinates |
| `room_type` | Entire home, Private room, Shared room |
| `price` | Listing price per night ($) |
| `minimum_nights` | Minimum nights required |
| `number_of_reviews` | Total reviews received |
| `reviews_per_month` | Average monthly reviews |
| `availability_365` | Days available per year |

---

## 📁 Project Structure

```
Algonive_AirbnbNYC_Analysis/
│
├── AB_NYC_2019.csv                  # Raw dataset
├── Airbnb_NYC_Analysis.ipynb        # Main project notebook
├── New_York_City_.png               # NYC banner image
└── README.md                        # Project documentation
```

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `Python 3.9` | Core programming language |
| `Pandas` | Data manipulation & analysis |
| `NumPy` | Numerical computations |
| `Matplotlib` | Static visualizations |
| `Seaborn` | Statistical visualizations |
| `Plotly` | Interactive charts & maps |
| `Scikit-learn` | Machine learning models |
| `XGBoost` | Gradient boosted trees |

---

## 🔄 Project Pipeline

```
Raw Data
   │
   ▼
Data Cleaning & Preprocessing
   │  ├── Handle missing values
   │  ├── Remove invalid prices
   │  └── IQR outlier removal
   ▼
Exploratory Data Analysis (EDA)
   │  ├── Price distribution
   │  ├── Borough analysis
   │  ├── Room type analysis
   │  ├── Geographic map
   │  └── Correlation heatmap
   ▼
Feature Engineering
   │  ├── Label encoding
   │  ├── New derived features
   │  └── Log transformation
   ▼
Machine Learning Models
   │  ├── Train 7 models
   │  ├── Compare performance
   │  └── Feature importance
   ▼
Dashboard & Visualization
   │  ├── Interactive Plotly dashboard
   │  └── Key findings summary
   ▼
Final Insights & Conclusion
```

---

## 📊 Exploratory Data Analysis

The EDA section includes **9 detailed visualizations:**

- 📈 Price distribution (normal + log scale)
- 🗺️ Interactive NYC map with all listings
- 🏙️ Number of listings by borough
- 💰 Average price by borough
- 🛏️ Room type distribution (pie chart)
- 💵 Average price by room type
- 📦 Boxplot — price by borough + room type
- 🔥 Correlation heatmap
- 🏆 Top 15 most expensive neighbourhoods

### Borough Price Comparison

| Borough | Price Level | Notes |
|---------|-------------|-------|
| Manhattan | Highest 💰 | Most expensive borough |
| Brooklyn | Medium | Most listings overall |
| Queens | Low-Medium | Affordable option |
| Staten Island | Low | Fewer listings |
| Bronx | Lowest | Most budget-friendly |

---

## ⚙️ Feature Engineering

New features created to improve model accuracy:

| Feature | Description |
|---------|-------------|
| `has_reviews` | 1 if listing has reviews, else 0 |
| `availability_ratio` | availability_365 / 365 |
| `high_availability` | 1 if available > 180 days/year |
| `popular_host` | 1 if host has > 5 listings |
| `log_price` | Log transformation of price (target variable) |

---

## 🤖 Machine Learning Models

### Models Trained & Compared

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Linear | Baseline model |
| Ridge Regression | Linear + L2 | Handles multicollinearity |
| Lasso Regression | Linear + L1 | Feature selection |
| Decision Tree | Tree-based | Captures non-linear patterns |
| Random Forest | Ensemble | ⭐ Best performance |
| Gradient Boosting | Ensemble | Strong runner-up |
| XGBoost | Boosting | High accuracy |

### Evaluation Metrics Used

- **MAE** — Mean Absolute Error (in $)
- **RMSE** — Root Mean Squared Error (in $)
- **R²** — R-squared score (on log scale)

### Best Model Results

```
Best Model  : Random Forest
R² Score    : ~0.60+
MAE         : ~$30
RMSE        : ~$43
```

---

## 🔑 Key Findings

1. 📍 **Location is everything** — Borough and neighbourhood are the strongest predictors of listing price
2. 🛏️ **Room type matters** — Entire home/apt listings cost significantly more than private or shared rooms
3. 🏙️ **Manhattan is most expensive** — Highest average price per night across all boroughs
4. 🏷️ **Bronx is most affordable** — Best budget-friendly option in NYC
5. ⭐ **Random Forest wins** — Best R² score among all 7 models due to non-linear price relationships
6. 📊 **Log transformation helps** — Price is right-skewed, log transform significantly improves model performance

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/Suryakantprajapati4/Algonive_AirbnbNYC_Analysis.git
cd Algonive_AirbnbNYC_Analysis
```

### Step 2 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly
```

### Step 3 — Open the notebook
```bash
jupyter notebook Airbnb_NYC_Analysis.ipynb
```

Or open in **VS Code** and click **Run All**

---

## 👤 Author

**Suryakant Prajapati**

- 🎓 Final Year Student — AI & Data Science
- 🏫 Khwaja Moinuddin Chishti Language University, Lucknow
- 💼 Data Analytics Intern @ Algonive
- 🔗 [LinkedIn](https://www.linkedin.com/in/suryakant-prajapati/)
- 🐙 [GitHub](https://github.com/Suryakantprajapati4)

---

## 🏢 About Algonive

> *"Empowering Intelligence Through Elegant Algorithms"*

Algonive provides students with real-world tech industry experience through hands-on internship programs.

- 🌐 LinkedIn: [@Algonive](https://www.linkedin.com/company/algonive/)
- 📧 services.algonivetech@gmail.com

---

<p align="center">
  Made with ❤️ as part of Algonive Data Analytics Internship
</p>
