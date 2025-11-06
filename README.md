# 🏠 Housing Price Prediction using Linear Regression

![Lab Logo](https://www.smu.tn/storage/app/media/logos/medtechlogo-b.png)

A comprehensive machine learning project implementing multiple linear regression models for predicting housing prices. This project was developed as part of **CS434 Lab 4: Feature Engineering & Regression Models** at South Mediterranean University.

**Author**: Fatma Alzahra Mohamed, SE-G1

---

## 📊 Project Overview

This project explores various regression techniques to predict housing prices based on multiple features including area, number of bedrooms, location amenities, and more. The analysis includes extensive feature engineering, data visualization, and model comparison to achieve optimal prediction accuracy.

### Key Highlights

- **Best Model Performance**: Lasso Regression with Test R² of **0.7193**
- **Test RMSE**: 1,042,786.48
- **Mean Percentage Error**: 16.34%
- **Dataset**: 545 housing records with 13 features
- **Models Implemented**: Linear Regression, Ridge, Lasso, ElasticNet, and Ensemble methods

---

## 📂 Dataset

**Source**: [Housing Prices Dataset on Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

### Features

**Numerical Features**:
- `price` - House price (target variable)
- `area` - Total area in square feet
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `stories` - Number of stories
- `parking` - Number of parking spaces

**Categorical Features**:
- `mainroad` - Whether connected to main road (yes/no)
- `guestroom` - Presence of guestroom (yes/no)
- `basement` - Presence of basement (yes/no)
- `hotwaterheating` - Hot water heating available (yes/no)
- `airconditioning` - Air conditioning available (yes/no)
- `prefarea` - Located in preferred area (yes/no)
- `furnishingstatus` - Furnished, semi-furnished, or unfurnished

### Dataset Statistics
- **Total Records**: 545
- **Train/Test Split**: 78% / 22% (425 / 107 samples)
- **Missing Values**: None
- **Target Variable Range**: 1,750,000 to 13,300,000

---

## 🛠️ Technologies & Libraries

```python
- Python 3.11+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
```

---

## 🔬 Methodology

### 1. Data Loading & Exploration
- Comprehensive dataset inspection
- Statistical summary analysis
- Missing value detection
- Data type verification

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of target variable
- Q-Q plots for normality assessment
- Correlation matrix visualization
- Feature relationship analysis
- Outlier detection using box plots and IQR method

### 3. Feature Engineering
- **Log transformation** of skewed features
- **Box-Cox transformation** for normality
- One-hot encoding of categorical variables
- Feature scaling using StandardScaler
- Creation of interaction features:
  - `area_per_bedroom`
  - `area_per_bathroom`
  - `total_rooms`
  - `luxury_score`

### 4. Model Development
Multiple regression models were implemented and compared:

1. **Linear Regression** (Baseline)
2. **Ridge Regression** (L2 Regularization)
3. **Lasso Regression** (L1 Regularization)
4. **ElasticNet Regression** (Combined L1/L2)
5. **Ensemble Methods**:
   - Average Ensemble
   - Weighted Ensemble
   - Stacking Ensemble

### 5. Model Evaluation
- Train/Test split (78/22)
- Cross-validation (5-fold)
- Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score
  - Percentage Error Analysis

---

## 📈 Results

### Best Model: Lasso Regression

| Metric | Value |
|--------|-------|
| **Test RMSE** | 1,042,786.48 |
| **Test MAE** | 749,710.64 |
| **Test R²** | 0.7193 |
| **Mean % Error** | 16.34% |
| **Median % Error** | 10.48% |

### Prediction Accuracy Breakdown
- Predictions within **5% error**: 18.7%
- Predictions within **10% error**: 45.8%
- Predictions within **15% error**: 59.8%
- Predictions within **20% error**: 72.0%

### Model Comparison

| Model | Train R² | Test R² | Test RMSE |
|-------|----------|---------|-----------|
| Linear Regression | 0.7351 | 0.7174 | 1,049,351 |
| Ridge | 0.7350 | 0.7175 | 1,049,014 |
| **Lasso** | **0.7342** | **0.7193** | **1,042,786** |
| ElasticNet | 0.7341 | 0.7194 | 1,042,532 |

---

## 📊 Key Visualizations

The project includes comprehensive visualizations:

1. **Distribution Analysis**
   - Target variable distribution
   - Q-Q plots for normality assessment
   - Feature distribution histograms

2. **Correlation Analysis**
   - Heatmap of feature correlations
   - Top features correlated with price

3. **Model Performance**
   - Actual vs Predicted scatter plots
   - Residual plots
   - Cross-validation score distributions

4. **Feature Importance**
   - Coefficient analysis for linear models
   - Feature contribution visualization

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Running the Notebook

1. Clone this repository:
```bash
git clone https://github.com/yourusername/house-prices-linear-regression.git
cd house-prices-linear-regression
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)

3. Place the `Housing.csv` file in the appropriate directory

4. Open and run the Jupyter notebook:
```bash
jupyter notebook house-prices-linear-regression.ipynb
```

---

## 📁 Project Structure

```
house-prices-linear-regression/
│
├── house-prices-linear-regression.ipynb    # Main Jupyter notebook
├── Housing.csv                              # Dataset (download separately)
├── housing_predictions.csv                  # Model predictions output
├── model_report.txt                         # Detailed model report
└── README.md                                # Project documentation
```

---

## 🔍 Key Findings

1. **Most Important Features**:
   - Area (highest correlation with price: ~0.54)
   - Bathrooms
   - Stories
   - Air conditioning

2. **Feature Engineering Impact**:
   - Log transformation significantly improved model performance
   - Interaction features captured non-linear relationships
   - Feature scaling was crucial for regularized models

3. **Model Insights**:
   - Lasso performed best due to feature selection capabilities
   - Ridge and ElasticNet showed similar performance
   - Ensemble methods didn't significantly outperform single models
   - Regularization helped prevent overfitting

4. **Prediction Patterns**:
   - Model performs well for mid-range priced houses
   - Higher error rates for luxury properties
   - Location features (mainroad, prefarea) showed significant impact

---

## 📝 Future Improvements

- [ ] Implement advanced ensemble techniques (XGBoost, Random Forest)
- [ ] Feature selection using recursive feature elimination
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deep learning models (Neural Networks)
- [ ] Geographic feature engineering
- [ ] Additional external datasets integration
- [ ] Model deployment using Flask/FastAPI

---

## 📚 References

- Dataset: [Housing Price Prediction - Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- Scikit-learn Documentation: [sklearn.linear_model](https://scikit-learn.org/stable/modules/linear_model.html)
- Feature Engineering Best Practices

---

## 👤 Author

**Fatma Alzahra Mohamed**
- Course: CS434 - Machine Learning
- Group: SE-G1
- Institution: South Mediterranean University

---

## 📄 License

This project is part of academic coursework and is available for educational purposes.

---

## 🙏 Acknowledgments

- South Mediterranean University - MedTech
- CS434 Course Instructors
- Kaggle community for the dataset

---

⭐ If you found this project helpful, please consider giving it a star!
