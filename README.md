# Task 3: Car Price Prediction with Machine Learning

## 📋 Project Overview

This project implements a machine learning solution to predict car prices based on various features such as brand, year, mileage, fuel type, and transmission. The model helps buyers estimate fair market value and assists sellers in determining competitive pricing.

## 🎯 Objectives

- Predict car selling prices using regression models
- Analyze feature importance and correlations
- Handle data preprocessing and feature engineering
- Compare multiple machine learning algorithms
- Provide insights for real-world pricing decisions

## 📊 Dataset

**File**: `car_data.csv`

- **Records**: 303 cars
- **Features**: 9 columns
  - `Car_Name`: Brand and model
  - `Year`: Manufacturing year
  - `Selling_Price`: Target variable (in lakhs)
  - `Present_Price`: Current market price (in lakhs)
  - `Driven_kms`: Total kilometers driven
  - `Fuel_Type`: Petrol, Diesel, or CNG
  - `Selling_type`: Dealer or Individual
  - `Transmission`: Manual or Automatic
  - `Owner`: Number of previous owners

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical visualizations
  - `scikit-learn` - Machine learning models and metrics

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
```bash
cd Task_3
python car_price_prediction.py
```

## 🔍 Methodology

### 1. Data Exploration (EDA)
- Loaded and inspected dataset structure
- Checked for missing values and duplicates
- Analyzed statistical distributions
- Visualized correlations between features

### 2. Feature Engineering
- **Created `Age` feature**: `Age = 2024 - Year`
- **Dropped columns**: `Car_Name` (high cardinality), `Year` (replaced by Age)
- **One-Hot Encoding**: Applied to categorical variables (Fuel_Type, Selling_type, Transmission)

### 3. Model Training
- **Train-Test Split**: 80% training, 20% testing
- **Models Trained**:
  - Linear Regression
  - Random Forest Regressor (100 estimators, max_depth=10)

### 4. Model Evaluation
- **Metrics Used**:
  - R² Score (Coefficient of Determination)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

## 📈 Results

### Model Performance

| Model | R² Score | MAE | RMSE | Performance |
|-------|----------|-----|------|-------------|
| **Random Forest** | **0.9609** | **0.6308** | **0.8042** | **Excellent** ✅ |
| Linear Regression | 0.8489 | 1.2164 | 1.5234 | Good |

### Key Insights

- **Best Model**: Random Forest Regressor achieved **96.09% accuracy**
- **Most Important Features**:
  1. Present_Price (current market value)
  2. Age (vehicle age)
  3. Driven_kms (mileage)
  4. Fuel_Type and Transmission

- **Average Prediction Error**: ±0.63 lakh (₹63,000)

## 📊 Visualizations Generated

1. **`selling_price_distribution.png`** - Distribution and boxplot of selling prices
2. **`correlation_matrix.png`** - Heatmap showing feature correlations
3. **`actual_vs_predicted.png`** - Scatter plots comparing actual vs predicted prices
4. **`feature_importance.png`** - Top 10 most important features (Random Forest)
5. **`residual_plots.png`** - Residual analysis for both models

## 💡 Real-World Applications

This model can be used by:

- **Car Dealerships**: Price inventory competitively based on market data
- **Buyers**: Estimate fair market value before purchasing
- **Sellers**: Determine optimal pricing strategy to maximize returns
- **Insurance Companies**: Assess vehicle worth for coverage calculations
- **Financial Institutions**: Evaluate collateral value for auto loans

## 📁 Project Structure

```
Task_3/
├── car_data.csv                      # Dataset
├── car_price_prediction.py           # Main script
├── selling_price_distribution.png    # Visualization
├── correlation_matrix.png            # Visualization
├── actual_vs_predicted.png           # Visualization
├── feature_importance.png            # Visualization
├── residual_plots.png                # Visualization
└── README.md                         # This file
```

## 🔑 Key Takeaways

✅ Random Forest significantly outperformed Linear Regression (96% vs 85% accuracy)  
✅ Present_Price and Age are the strongest predictors of car value  
✅ Feature engineering (Age from Year) improved model performance  
✅ One-hot encoding effectively handled categorical variables  
✅ Model is production-ready for real-world car price estimation

## 📝 Future Improvements

- Incorporate additional features (engine size, safety ratings, brand reputation)
- Experiment with ensemble methods (Gradient Boosting, XGBoost)
- Implement hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Deploy as a web application for user-friendly predictions
- Add time-series analysis for price trend forecasting

## 👨‍💻 Author

**Code Alpha Internship - Task 3**  
*Machine Learning for Car Price Prediction*

---

**Status**: ✅ Completed  
**Accuracy**: 96.09% (Random Forest)  
**Date**: January 2026
