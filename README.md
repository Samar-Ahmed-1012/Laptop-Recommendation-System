# Laptop-Recommendation-System
A Data Science project that automates laptop data collection via Selenium and provides intelligent recommendations using a Flask-based Machine Learning pipeline

# 💻 Laptop Recommendation System (Data Science Project)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/Machine--Learning-Scikit--Learn-orange.svg)
![Framework](https://img.shields.io/badge/Web--Framework-Flask-red.svg)

## 📊 Project Overview
This is a complete **Data Science & Software Engineering pipeline** designed to simplify the laptop buying process. The system scrapes real-time market data from e-commerce sites and uses an intelligent engine to recommend laptops based on user-defined specs (RAM, Storage, CPU) and budget.

## 🛠️ Tech Stack
- **Languages:** Python, HTML, CSS, JavaScript
- **Web Scraping:** Selenium (for handling dynamic JavaScript content)
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Random Forest, Decision Trees)
- **Backend:** Flask

## 🏗️ System Architecture
1. **Scraping Phase (`scraper.py`):** - Uses Selenium to automate browser navigation.
   - Extracts 200+ laptop listings including titles, prices, and hardware specs.
   - Saves clean data into `laptops.csv`.
   
2. **Machine Learning Phase (`main.py`):**
   - **Preprocessing:** Cleans prices, handles missing values, and performs feature scaling.
   - **Modeling:** Trains a **Random Forest Regressor** to understand price-to-spec correlations.
   - **Recommendation:** A content-based engine that filters and ranks laptops using user inputs.

3. **Frontend Phase:**
   - A modern, responsive web dashboard where users can input their preferences and see results in real-time.

## 🚀 Key Features
- ✅ **Live Data:** Not just static data, but freshly scraped market prices.
- ✅ **Visual Insights:** Automatically generates EDA charts (Price distribution, Brand analysis).
- ✅ **Smart Filters:** Filter by RAM (4GB to 64GB), Storage, and specific Budget ranges.
- ✅ **Price Prediction:** Uses ML to show if a laptop is priced fairly based on its specs.

## 📁 Repository Structure
```text
├── data/               # Scraped CSV files
├── models/             # Saved joblib models
├── static/             # CSS & Generated EDA Plots
├── templates/          # Flask HTML templates
├── main.py             # Flask App & ML Pipeline
└── scraper.py          # Selenium Scraper Logic
