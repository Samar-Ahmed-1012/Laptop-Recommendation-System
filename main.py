# main.py - COMPLETE ML PIPELINE WITH FLASK SERVER (FIXED VERSION)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ============================================================================
# PHASE 1: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: FEATURE ENGINEERING")
print("="*80)

# Load the data from scraper
df = pd.read_csv('data/laptops.csv')
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create a copy for feature engineering
df_engineered = df.copy()

# Remove unnecessary columns
df_engineered = df_engineered.drop(['name', 'url', 'source'], axis=1, errors='ignore')

# Handle missing values
for col in df_engineered.columns:
    if df_engineered[col].isnull().sum() > 0:
        if df_engineered[col].dtype == 'object':
            df_engineered[col].fillna(df_engineered[col].mode()[0], inplace=True)
        else:
            df_engineered[col].fillna(df_engineered[col].median(), inplace=True)

# Create new features
# 1. Price categories (for classification)
df_engineered['price_category'] = pd.cut(df_engineered['price_pkr'],
                                         bins=[0, 50000, 100000, 150000, 250000, float('inf')],
                                         labels=['Budget', 'Entry', 'Mid', 'Premium', 'Luxury'])

# 2. Performance score
df_engineered['performance_score'] = (df_engineered['ram_gb'] * 0.4 +
                                     (df_engineered['storage_gb'] / 1024) * 0.3 +
                                     df_engineered['display_inches'] * 0.3)

# 3. SSD indicator
df_engineered['has_ssd'] = df_engineered['storage_gb'].apply(lambda x: 1 if x >= 256 else 0)

# 4. Brand tiers
brand_tiers = {
    'Apple': 'Premium',
    'Microsoft': 'Premium',
    'Dell': 'Mid-High',
    'HP': 'Mid',
    'Lenovo': 'Mid',
    'ASUS': 'Mid',
    'Acer': 'Budget-Mid',
    'MSI': 'High',
    'Samsung': 'Mid',
    'Unknown': 'Unknown'
}
df_engineered['brand_tier'] = df_engineered['brand'].map(brand_tiers).fillna('Other')

# 5. Processor tiers
processor_tiers = {
    'Intel i9': 'High',
    'Intel i7': 'Mid-High',
    'Intel i5': 'Mid',
    'Intel i3': 'Budget',
    'AMD Ryzen 9': 'High',
    'AMD Ryzen 7': 'Mid-High',
    'AMD Ryzen 5': 'Mid',
    'Apple M3': 'Premium',
    'Apple M2': 'Premium',
    'Apple M1': 'Premium'
}
df_engineered['processor_tier'] = df_engineered['processor'].map(processor_tiers).fillna('Other')

# 6. Price per GB
df_engineered['price_per_ram_gb'] = df_engineered['price_pkr'] / df_engineered['ram_gb']
df_engineered['price_per_storage_gb'] = df_engineered['price_pkr'] / df_engineered['storage_gb']

# 7. Gaming laptop indicator
df_engineered['is_gaming'] = ((df_engineered['ram_gb'] >= 16) & 
                              (df_engineered['processor'].str.contains('i7|i9|Ryzen 7|Ryzen 9'))).astype(int)

# Encode categorical features
label_encoders = {}
categorical_cols = ['brand', 'processor', 'brand_tier', 'processor_tier']

for col in categorical_cols:
    le = LabelEncoder()
    df_engineered[col + '_encoded'] = le.fit_transform(df_engineered[col].astype(str))
    label_encoders[col] = le
    df_engineered = df_engineered.drop(col, axis=1)

print(f"✅ Feature engineering complete. New shape: {df_engineered.shape}")

# Save engineered data
df_engineered.to_csv('data/laptops_engineered.csv', index=False)
print("💾 Saved engineered data to: data/laptops_engineered.csv")

# ============================================================================
# BUSINESS INTELLIGENCE REPORT
# ============================================================================

def save_business_report(df, df_engineered):
    """Save comprehensive business report"""
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = []
    
    # 1. MARKET OVERVIEW
    total_laptops = len(df)
    avg_price = df['price_pkr'].mean()
    median_price = df['price_pkr'].median()
    min_price = df['price_pkr'].min()
    max_price = df['price_pkr'].max()
    
    report_content.append("="*80)
    report_content.append("📊 BUSINESS INTELLIGENCE REPORT - PAKISTAN LAPTOP MARKET")
    report_content.append(f"📅 Report Date: {current_date}")
    report_content.append("="*80)
    
    report_content.append("\n1️⃣ MARKET OVERVIEW")
    report_content.append("-" * 50)
    report_content.append(f"📈 Total Laptops Analyzed: {total_laptops:,}")
    report_content.append(f"💰 Average Price: Rs. {avg_price:,.0f}")
    report_content.append(f"📊 Median Price: Rs. {median_price:,.0f}")
    report_content.append(f"🎯 Price Range: Rs. {min_price:,} - Rs. {max_price:,}")
    
    # 2. BRAND ANALYSIS
    report_content.append("\n\n2️⃣ BRAND ANALYSIS")
    report_content.append("-" * 50)
    brand_distribution = df['brand'].value_counts().head(10)
    for brand, count in brand_distribution.items():
        percentage = (count / total_laptops) * 100
        report_content.append(f"{brand:<15} {count:>6,} laptops ({percentage:>5.1f}%)")
    
    # 3. PRICE SEGMENT ANALYSIS
    report_content.append("\n\n3️⃣ PRICE SEGMENT ANALYSIS")
    report_content.append("-" * 50)
    price_segments = df_engineered['price_category'].value_counts()
    for segment, count in price_segments.items():
        percentage = (count / total_laptops) * 100
        report_content.append(f"{segment:<15} {count:>6,} laptops ({percentage:>5.1f}%)")
    
    # 4. SPECIFICATIONS ANALYSIS
    report_content.append("\n\n4️⃣ SPECIFICATIONS ANALYSIS")
    report_content.append("-" * 50)
    
    # RAM Distribution
    ram_dist = df['ram_gb'].value_counts().sort_index()
    report_content.append("🔹 RAM Distribution:")
    for ram, count in ram_dist.head(5).items():
        report_content.append(f"   {ram}GB: {count:,} laptops")
    
    # Storage Distribution
    storage_dist = df['storage_gb'].value_counts().sort_index().head(5)
    report_content.append("\n🔹 Storage Distribution:")
    for storage, count in storage_dist.items():
        if storage >= 1024:
            report_content.append(f"   {storage/1024:.0f}TB: {count:,} laptops")
        else:
            report_content.append(f"   {storage}GB: {count:,} laptops")
    
    # 5. TOP BRANDS BY AVERAGE PRICE
    report_content.append("\n\n5️⃣ TOP BRANDS BY AVERAGE PRICE")
    report_content.append("-" * 50)
    brand_avg_price = df.groupby('brand')['price_pkr'].mean().sort_values(ascending=False).head(10)
    for brand, price in brand_avg_price.items():
        report_content.append(f"{brand:<15} Rs. {price:>12,.0f}")
    
    # 6. PROCESSOR DISTRIBUTION
    report_content.append("\n\n6️⃣ PROCESSOR DISTRIBUTION")
    report_content.append("-" * 50)
    processor_dist = df['processor'].value_counts().head(10)
    for processor, count in processor_dist.items():
        percentage = (count / total_laptops) * 100
        report_content.append(f"{processor:<20} {count:>6,} ({percentage:>5.1f}%)")
    
    # 7. GAMING LAPTOPS MARKET
    gaming_count = df_engineered['is_gaming'].sum()
    gaming_percentage = (gaming_count / total_laptops) * 100
    avg_gaming_price = df[df_engineered['is_gaming'] == 1]['price_pkr'].mean()
    
    report_content.append("\n\n7️⃣ GAMING LAPTOP MARKET")
    report_content.append("-" * 50)
    report_content.append(f"🎮 Total Gaming Laptops: {gaming_count:,} ({gaming_percentage:.1f}%)")
    report_content.append(f"💰 Average Gaming Laptop Price: Rs. {avg_gaming_price:,.0f}")
    
    # 8. RECOMMENDATIONS
    report_content.append("\n\n8️⃣ BUSINESS RECOMMENDATIONS")
    report_content.append("-" * 50)
    report_content.append("🎯 1. Focus on 'Budget' segment (Most popular)")
    report_content.append("🎯 2. HP and Lenovo have highest market share")
    report_content.append("🎯 3. Gaming laptops are premium segment opportunity")
    report_content.append("🎯 4. 8GB RAM and 512GB SSD are standard specs")
    
    # Save report to file
    report_filename = f"reports/business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"📄 Business report saved: {report_filename}")
    
    # Also save as CSV summary
    summary_data = {
        'Metric': [
            'Total Laptops', 'Average Price', 'Median Price', 
            'Top Brand', 'Most Common RAM', 'Most Common Storage',
            'Gaming Laptops %', 'Most Expensive Brand', 'Cheapest Brand'
        ],
        'Value': [
            f"{total_laptops:,}",
            f"Rs. {avg_price:,.0f}",
            f"Rs. {median_price:,.0f}",
            brand_distribution.index[0],
            f"{ram_dist.index[0]}GB",
            f"{storage_dist.index[0]}GB",
            f"{gaming_percentage:.1f}%",
            f"{brand_avg_price.index[0]} (Rs. {brand_avg_price.iloc[0]:,.0f})",
            f"{brand_avg_price.index[-1]} (Rs. {brand_avg_price.iloc[-1]:,.0f})"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = f"reports/market_summary_{datetime.now().strftime('%Y%m%d')}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"📊 Market summary saved: {summary_csv}")
    
    return '\n'.join(report_content[:50])

# Generate and save business report
print("\n" + "="*80)
print("📊 GENERATING BUSINESS INTELLIGENCE REPORT")
print("="*80)
business_report = save_business_report(df, df_engineered)
print(business_report)

# ============================================================================
# PHASE 2: MODEL BUILDING (REGRESSION & CLASSIFICATION)
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: MODEL BUILDING")
print("="*80)

# ============================================
# 2.1 PRICE PREDICTION (REGRESSION)
# ============================================
print("\n💰 PRICE PREDICTION MODELS (REGRESSION)")

# Prepare data for regression
X_reg = df_engineered.drop(['price_pkr', 'price_category'], axis=1)
y_reg = df_engineered['price_pkr']

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Feature scaling
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Initialize regression models
print("🤖 Initializing regression models...")
dt_reg = DecisionTreeRegressor(
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5
)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1
)

# ============================================
# 2.2 CATEGORY PREDICTION (CLASSIFICATION)
# ============================================
print("\n🏷️ CATEGORY PREDICTION MODELS (CLASSIFICATION)")

# Prepare data for classification
X_clf = df_engineered.drop(['price_pkr', 'price_category'], axis=1)
y_clf = df_engineered['price_category']

# Train-test split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Feature scaling
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Initialize classification models
print("🤖 Initializing classification models...")
dt_clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5
)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    n_jobs=-1
)

print("✅ All models initialized!")

# ============================================================================
# PHASE 3: TRAINING & EVALUATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: TRAINING & EVALUATION")
print("="*80)

# ============================================
# 3.1 REGRESSION MODEL TRAINING & EVALUATION
# ============================================
print("\n💰 REGRESSION MODELS EVALUATION")

# Train regression models
print("\n🌲 Training Decision Tree (Regression)...")
dt_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_dt_reg = dt_reg.predict(X_test_reg_scaled)

print("🌳 Training Random Forest (Regression)...")
rf_reg.fit(X_train_reg_scaled, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg_scaled)

# Evaluate regression models
def evaluate_regression(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {name} Regression Performance:")
    print(f"   MAE  (Mean Absolute Error): Rs. {mae:,.0f}")
    print(f"   MSE  (Mean Squared Error): {mse:,.0f}")
    print(f"   RMSE (Root Mean Squared Error): Rs. {rmse:,.0f}")
    print(f"   R²   (R-squared Score): {r2:.4f}")
    
    percentage_error = (mae / y_true.mean()) * 100
    print(f"   Average Error: {percentage_error:.1f}% of average price")
    
    return mae, rmse, r2

# Evaluate regression models
dt_mae, dt_rmse, dt_r2 = evaluate_regression("DECISION TREE", y_test_reg, y_pred_dt_reg)
rf_mae, rf_rmse, rf_r2 = evaluate_regression("RANDOM FOREST", y_test_reg, y_pred_rf_reg)

# Cross-validation for regression
print("\n🔬 Cross-validation (Regression):")
dt_reg_cv = cross_val_score(dt_reg, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')
rf_reg_cv = cross_val_score(rf_reg, X_train_reg_scaled, y_train_reg, cv=5, scoring='r2')

print(f"🌲 Decision Tree CV R²: {dt_reg_cv.mean():.4f} (+/- {dt_reg_cv.std()*2:.4f})")
print(f"🌳 Random Forest CV R²: {rf_reg_cv.mean():.4f} (+/- {rf_reg_cv.std()*2:.4f})")

# ============================================
# 3.2 CLASSIFICATION MODEL TRAINING & EVALUATION
# ============================================
print("\n" + "="*50)
print("🏷️ CLASSIFICATION MODELS EVALUATION")
print("="*50)

# Train classification models
print("\n🌲 Training Decision Tree (Classification)...")
dt_clf.fit(X_train_clf_scaled, y_train_clf)
y_pred_dt_clf = dt_clf.predict(X_test_clf_scaled)

print("🌳 Training Random Forest (Classification)...")
rf_clf.fit(X_train_clf_scaled, y_train_clf)
y_pred_rf_clf = rf_clf.predict(X_test_clf_scaled)

# Evaluate classification models
def evaluate_classification(name, y_true, y_pred, average_type='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
    
    print(f"\n📊 {name} Classification Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n   📋 Detailed Classification Report:")
    print("   " + classification_report(y_true, y_pred, zero_division=0).replace('\n', '\n   '))
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n   📊 Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"   {row}")
    
    return accuracy, precision, recall, f1

# Evaluate classification models
print("\n📈 CLASSIFICATION RESULTS:")
print("-" * 50)

dt_acc, dt_prec, dt_rec, dt_f1 = evaluate_classification("DECISION TREE", 
                                                         y_test_clf, y_pred_dt_clf)

rf_acc, rf_prec, rf_rec, rf_f1 = evaluate_classification("RANDOM FOREST", 
                                                         y_test_clf, y_pred_rf_clf)

# Cross-validation for classification
print("\n🔬 Cross-validation (Classification - Accuracy):")
dt_clf_cv_acc = cross_val_score(dt_clf, X_train_clf_scaled, y_train_clf, cv=5, scoring='accuracy')
rf_clf_cv_acc = cross_val_score(rf_clf, X_train_clf_scaled, y_train_clf, cv=5, scoring='accuracy')

print(f"🌲 Decision Tree CV Accuracy: {dt_clf_cv_acc.mean():.4f} (+/- {dt_clf_cv_acc.std()*2:.4f})")
print(f"🌳 Random Forest CV Accuracy: {rf_clf_cv_acc.mean():.4f} (+/- {rf_clf_cv_acc.std()*2:.4f})")

# ============================================
# 3.3 FEATURE IMPORTANCE
# ============================================
print("\n" + "="*50)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Regression feature importance
print("\n💰 REGRESSION - Feature Importance (Top 10):")
rf_reg_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in rf_reg_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Classification feature importance
print("\n🏷️ CLASSIFICATION - Feature Importance (Top 10):")
rf_clf_importance = pd.DataFrame({
    'feature': X_clf.columns,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in rf_clf_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# ============================================
# 3.4 SAVE MODELS
# ============================================
print("\n" + "="*50)
print("💾 SAVING MODELS")
print("="*50)

# Save regression models
joblib.dump(dt_reg, 'models/decision_tree_regressor.pkl')
joblib.dump(rf_reg, 'models/random_forest_regressor.pkl')
joblib.dump(scaler_reg, 'models/scaler_regressor.pkl')

# Save classification models
joblib.dump(dt_clf, 'models/decision_tree_classifier.pkl')
joblib.dump(rf_clf, 'models/random_forest_classifier.pkl')
joblib.dump(scaler_clf, 'models/scaler_classifier.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print("✅ Models saved:")
print("   - models/decision_tree_regressor.pkl")
print("   - models/random_forest_regressor.pkl")
print("   - models/decision_tree_classifier.pkl")
print("   - models/random_forest_classifier.pkl")
print("   - models/scaler_regressor.pkl")
print("   - models/scaler_classifier.pkl")
print("   - models/label_encoders.pkl")

# ============================================
# 3.5 SAVE ML PERFORMANCE REPORT
# ============================================
def save_ml_performance_report():
    """Save ML model performance report"""
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = []
    
    report_content.append("="*80)
    report_content.append("🤖 MACHINE LEARNING PERFORMANCE REPORT")
    report_content.append(f"📅 Report Date: {current_date}")
    report_content.append("="*80)
    
    # Regression Performance
    report_content.append("\n💰 REGRESSION MODELS (Price Prediction)")
    report_content.append("-" * 50)
    report_content.append(f"📊 Decision Tree:")
    report_content.append(f"   R² Score: {dt_r2:.4f}")
    report_content.append(f"   MAE: Rs. {dt_mae:,.0f}")
    report_content.append(f"   RMSE: Rs. {dt_rmse:,.0f}")
    
    report_content.append(f"\n📊 Random Forest:")
    report_content.append(f"   R² Score: {rf_r2:.4f}")
    report_content.append(f"   MAE: Rs. {rf_mae:,.0f}")
    report_content.append(f"   RMSE: Rs. {rf_rmse:,.0f}")
    
    # Classification Performance
    report_content.append("\n\n🏷️ CLASSIFICATION MODELS (Category Prediction)")
    report_content.append("-" * 50)
    report_content.append(f"📊 Decision Tree:")
    report_content.append(f"   Accuracy:  {dt_acc:.4f}")
    report_content.append(f"   Precision: {dt_prec:.4f}")
    report_content.append(f"   Recall:    {dt_rec:.4f}")
    report_content.append(f"   F1-Score:  {dt_f1:.4f}")
    
    report_content.append(f"\n📊 Random Forest:")
    report_content.append(f"   Accuracy:  {rf_acc:.4f}")
    report_content.append(f"   Precision: {rf_prec:.4f}")
    report_content.append(f"   Recall:    {rf_rec:.4f}")
    report_content.append(f"   F1-Score:  {rf_f1:.4f}")
    
    # Best Models
    best_reg = "Decision Tree" if dt_r2 > rf_r2 else "Random Forest"
    best_clf = "Decision Tree" if dt_acc > rf_acc else "Random Forest"
    
    report_content.append("\n\n🎯 BEST PERFORMING MODELS")
    report_content.append("-" * 50)
    report_content.append(f"📈 Price Prediction: {best_reg}")
    report_content.append(f"🏷️ Category Prediction: {best_clf}")
    
    # Feature Importance
    report_content.append("\n\n📊 TOP 5 IMPORTANT FEATURES")
    report_content.append("-" * 50)
    report_content.append("🔹 For Price Prediction:")
    for i, row in rf_reg_importance.head(5).iterrows():
        report_content.append(f"   {row['feature']}: {row['importance']:.4f}")
    
    report_content.append("\n🔹 For Category Prediction:")
    for i, row in rf_clf_importance.head(5).iterrows():
        report_content.append(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Business Impact
    report_content.append("\n\n💼 BUSINESS IMPACT")
    report_content.append("-" * 50)
    report_content.append(f"🎯 Price prediction accuracy: {max(dt_r2, rf_r2)*100:.1f}%")
    report_content.append(f"🎯 Category prediction accuracy: {max(dt_acc, rf_acc)*100:.1f}%")
    report_content.append(f"💰 Average prediction error: Rs. {min(dt_mae, rf_mae):,.0f}")
    
    # Save to file
    report_filename = f"reports/ml_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"📄 ML Performance report saved: {report_filename}")
    
    return '\n'.join(report_content[:30])

# Save ML performance report
print("\n" + "="*80)
print("📊 SAVING ML PERFORMANCE REPORT")
print("="*80)
ml_report = save_ml_performance_report()
print(ml_report)

# ============================================================================
# SIMPLE LOCAL HOST SERVER WITH EXTERNAL HTML
# ============================================================================

print("\n" + "="*80)
print("🌐 STARTING SIMPLE LOCAL HOST SERVER")
print("="*80)

# Create Flask app with static and templates folders
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Load models
print("📦 Loading trained models...")
try:
    dt_reg_model = joblib.load('models/decision_tree_regressor.pkl')
    rf_reg_model = joblib.load('models/random_forest_regressor.pkl')
    dt_clf_model = joblib.load('models/decision_tree_classifier.pkl')
    rf_clf_model = joblib.load('models/random_forest_classifier.pkl')
    scaler_reg_model = joblib.load('models/scaler_regressor.pkl')
    scaler_clf_model = joblib.load('models/scaler_classifier.pkl')
    label_encoders_model = joblib.load('models/label_encoders.pkl')
    
    # Load the engineered data for recommendations
    df_engineered = pd.read_csv('data/laptops_engineered.csv')
    print("✅ All models and data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit(1)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

from flask import render_template

@app.route('/')
def home():
    """Serve the HTML file"""
    return render_template('index.html')

# For your JavaScript API calls (to match existing code)
@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API endpoint for stats - matches your JavaScript"""
    try:
        stats = {
            "total_laptops": len(df),
            "brands": df['brand'].nunique(),
            "avg_price": int(df['price_pkr'].mean()),
            "min_price": int(df['price_pkr'].min()),
            "max_price": int(df['price_pkr'].max()),
            "average_price": int(df['price_pkr'].mean()),  # duplicate for compatibility
            "brands_count": df['brand'].nunique()  # duplicate for compatibility
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def api_recommend():
    """
    STRICT FILTERING FOR ALL SPECS:
    RAM, Storage, Brand, Processor, and Display
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided", "success": False}), 400

        # User Inputs
        u_ram = data.get('ram')
        u_storage = data.get('storage')
        u_brand = data.get('brand')
        u_processor = data.get('processor')
        u_display = data.get('display_size') # Make sure JS se 'display_size' aa raha ho
        u_max_price = data.get('max_price')

        # Start with full dataset
        query = df.copy()

        # 1. RAM Exact Match
        if u_ram and str(u_ram).isdigit():
            query = query[query['ram_gb'] == int(u_ram)]
        
        # 2. Storage Exact Match
        if u_storage and str(u_storage).isdigit():
            query = query[query['storage_gb'] == int(u_storage)]
            
        # 3. Display Exact Match (Ab exact check karega)
        if u_display:
            try:
                query = query[query['display_inches'] == float(u_display)]
            except: pass
            
        # 4. Brand Exact Match
        if u_brand and u_brand != "All":
            query = query[query['brand'].str.lower() == u_brand.lower()]
            
        # 5. Processor Match (Contains logic best hai processor ke liye)
        if u_processor and u_processor != "All":
            query = query[query['processor'].str.contains(u_processor, case=False, na=False)]

        # 6. Budget Limit
        if u_max_price:
            query = query[query['price_pkr'] <= int(u_max_price)]

        # Agar koi match nahi mila
        if query.empty:
            return jsonify({
                "success": True, 
                "recommendations": [], 
                "message": "Aapki chuni hui exact specs (RAM, Storage, Display) ke mutabiq koi laptop nahi mila."
            })

        # Result format karein
        recommendations = []
        for _, row in query.head(10).iterrows():
            recommendations.append({
                "name": row['name'],
                "brand": row['brand'],
                "price": int(row['price_pkr']),
                "ram": int(row['ram_gb']),
                "storage": int(row['storage_gb']),
                "processor": row['processor'],
                "display": float(row['display_inches']),
                "url": row.get('url', '#'),
                "score": 9.5
            })

        return jsonify({"success": True, "count": len(recommendations), "recommendations": recommendations})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Original endpoints (keep for backward compatibility)
@app.route('/market_stats', methods=['GET'])
def market_stats():
    """Get market statistics"""
    try:
        stats = {
            "total_laptops": len(df),
            "average_price": int(df['price_pkr'].mean()),
            "min_price": int(df['price_pkr'].min()),
            "max_price": int(df['price_pkr'].max()),
            "brands_count": df['brand'].nunique(),
            "ram_options": sorted(df['ram_gb'].unique().tolist())
        }
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Recommend laptops based on criteria"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided", "success": False}), 400
        
        # Get form data
        ram = int(data.get('ram', 8))
        max_price = int(data.get('max_price', 150000))
        storage = int(data.get('storage', 512))
        display_size = float(data.get('display_size', 15.6))
        
        # Filter laptops
        filtered_laptops = df[
            (df['ram_gb'] >= ram) &
            (df['price_pkr'] <= max_price) &
            (df['storage_gb'] >= storage)
        ]
        
        if len(filtered_laptops) == 0:
            filtered_laptops = df.sort_values('price_pkr')
        
        recommendations = []
        for idx, row in filtered_laptops.head(5).iterrows():
            rec = {
                "name": f"{row.get('brand', 'Laptop')}",
                "price": int(row['price_pkr']),
                "ram": int(row['ram_gb']),
                "storage": int(row['storage_gb']),
                "display": float(row['display_inches']),
                "processor": row.get('processor', 'N/A'),
                "brand": row.get('brand', 'Unknown'),
                "score": 8.5
            }
            recommendations.append(rec)
        
        stats = {
            "total_matched": len(filtered_laptops),
            "average_price": int(filtered_laptops['price_pkr'].mean()) if len(filtered_laptops) > 0 else 0,
            "min_price": int(filtered_laptops['price_pkr'].min()) if len(filtered_laptops) > 0 else 0,
            "max_price": int(filtered_laptops['price_pkr'].max()) if len(filtered_laptops) > 0 else 0,
            "total_laptops": len(df)
        }
        
        response = {
            "success": True,
            "recommendations": recommendations,
            "stats": stats,
            "count": len(recommendations)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": True
    })

# Run server
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SERVER STARTING")
    print("="*80)
    print("\n📡 Server running at:")
    print("   http://127.0.0.1:5000")
    print("   http://localhost:5000")
    print("\n📋 Open in browser: http://127.0.0.1:5000")
    print("\n🔌 Available endpoints:")
    print("   GET  /              - HTML interface")
    print("   GET  /api/stats     - Market stats (for JS)")
    print("   POST /api/recommend - Get recommendations (for JS)")
    print("   GET  /market_stats  - Market stats")
    print("   POST /recommend     - Get recommendations")
    print("   GET  /health        - Health check")
    print("\n" + "="*80)
    print("✅ SERVER READY - Press Ctrl+C to stop")
    print("="*80)
    
    app.run(host='127.0.0.1', port=5000, debug=False)
