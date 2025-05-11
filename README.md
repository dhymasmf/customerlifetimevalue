# Customer Lifetime Value (CLV) Prediction for Auto Insurance

## Project Overview
This repository contains a machine learning project aimed at predicting **Customer Lifetime Value (CLV)** for an auto insurance company. CLV represents the total revenue expected from a customer over their relationship with the business. The project leverages supervised regression models to enable data-driven decision-making, focusing on:

- **Predicting CLV** using customer demographic, behavioral, and financial data.
- **Identifying high-value customers** early to optimize marketing and retention strategies.
- **Supporting personalized marketing** and upselling initiatives.

The final model, a **Tuned Random Forest**, achieves an **R² of 0.9350**, **RMSE of 800.96**, and a low MAE, demonstrating high predictive accuracy.

## Dataset
The dataset (`data_customer_lifetime_value.csv`) includes 5,669 records with 11 features:
- **Categorical Features**: `Vehicle Class`, `Coverage`, `Renew Offer Type`, `EmploymentStatus`, `Marital Status`, `Education`
- **Numerical Features**: `Number of Policies`, `Monthly Premium Auto`, `Total Claim Amount`, `Income`
- **Target Variable**: `Customer Lifetime Value` (CLV)

Key insights:
- No missing values in the dataset.
- Outliers in CLV were capped at 16,624.75 to improve model performance.
- Strong correlations: `Monthly Premium Auto` (0.48) and `Income` (0.45) with CLV.

## Project Structure
```
├── data_customer_lifetime_value.csv  # Dataset
├── Customer Lifetime Value.ipynb     # Jupyter Notebook with analysis and modeling
├── README.md                        # Project documentation
```

## Installation
To run the project locally, ensure you have Python 3.8+ and the required libraries installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/customer-lifetime-value.git
   cd customer-lifetime-value
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**:
   ```
   pandas
   numpy
   missingno
   seaborn
   matplotlib
   joblib
   scipy
   xgboost
   statsmodels
   scikit-learn
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook "Customer Lifetime Value.ipynb"
   ```

## Methodology
1. **Data Cleaning**:
   - Handled missing values (none found).
   - Capped CLV outliers at 16,624.75.
   - Encoded categorical variables using `OneHotEncoder`.
   - Scaled numerical features with `RobustScaler`.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized CLV distribution and feature relationships (e.g., boxplots, heatmaps).
   - Identified key predictors: `Monthly Premium Auto`, `Income`, `Number of Policies`.

3. **Modeling**:
   - Tested models: Linear Regression, ElasticNet, Random Forest, XGBoost.
   - Selected **Tuned Random Forest** for best performance (R²: 0.9350, RMSE: 800.96).
   - Evaluated using RMSE, MAE, and R².

4. **Feature Importance**:
   - `Monthly Premium Auto` and `Income` were the most influential predictors.
   - Categorical features like `Coverage` and `Renew Offer Type` also contributed significantly.

## Results
- **Model Performance**:
  - **RMSE**: 800.96 (low error for CLV range of 2,000–16,000)
  - **R²**: 0.9350 (explains 93.5% of CLV variance)
  - **MAE**: Low, ensuring manageable individual prediction errors
- **Key Insights**:
  - Customers with `Extended Coverage` or `Offer2/Offer4` renewals tend to have higher CLV.
  - Minimal multicollinearity among features (e.g., correlation of 0.0044 between `Number of Policies` and `Monthly Premium Auto`).

## Recommendations
1. **Business Applications**:
   - Target customers with CLV > 8,000 within six months of onboarding.
   - Design personalized campaigns for high `Monthly Premium Auto` or multi-policy holders.
   - Upsell premium tiers to customers with moderate CLV (4,000–8,000).
   - Reallocate 20% of marketing budget to top 10% high-value customers (CLV > 12,000) for a projected 15% revenue increase.

2. **Model Enhancements**:
   - Incorporate behavioral data (e.g., claim frequency, customer service interactions).
   - Experiment with ensemble methods (e.g., stacking Random Forest with XGBoost).
   - Conduct quarterly retraining to adapt to new data.

3. **Monitoring**:
   - Implement A/B testing to measure ROI of CLV-driven strategies.
   - Analyze residuals for high-CLV predictions to refine accuracy.

## Usage
To use the trained model for predictions:
1. Load the dataset and trained model (saved via `joblib` in the notebook).
2. Preprocess new data using the same `ColumnTransformer` (OneHotEncoder + RobustScaler).
3. Run predictions with the Tuned Random Forest model.

Example:
```python
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv('data_customer_lifetime_value.csv')
model = joblib.load('tuned_random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Prepare features
X = df.drop('Customer Lifetime Value', axis=1)
X_preprocessed = preprocessor.transform(X)

# Predict CLV
predictions = model.predict(X_preprocessed)
```

## Contact
[LinkedIn](https://www.linkedin.com/in/dhymasmf/)
Email : [Gmail](dhymas.maulidin@gmail.com)

**Author**
 [Dimas Maulidin Firdaus](https://github.com/dhymasmf)  
