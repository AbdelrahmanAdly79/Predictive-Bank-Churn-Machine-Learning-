import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def run_linear_regression():
    print("--- Bank Customer Data: Linear Regression Analysis ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('Bank_Churn.csv')
        print(f"[OK] Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("[ERROR] Bank_Churn.csv not found.")
        return

    # 2. Data Preparation
    # We will predict 'EstimatedSalary' based on other customer attributes
    # Features selection (excluding ID and Surname)
    target = 'EstimatedSalary'
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    X = df[features]
    y = df[target]

    # 3. Preprocessing
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[OK] Split data into Train ({len(X_train)}) and Test ({len(X_test)})")

    # 5. Scaling (Good practice for linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model Training
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("[OK] Model training complete.")

    # 7. Predictions & Evaluation
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("Note: In this dataset, EstimatedSalary is often poorly correlated with features, resulting in a low R2.")

    # 8. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # A. Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=axes[0,0])
    axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Salary')
    axes[0,0].set_ylabel('Predicted Salary')
    axes[0,0].set_title('Actual vs Predicted Salaries')

    # B. Residual Plot
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, ax=axes[0,1], color='purple')
    axes[0,1].set_title('Distribution of Residuals (Errors)')
    axes[0,1].set_xlabel('Prediction Error')

    # C. Coefficient Importance
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis', ax=axes[1,0])
    axes[1,0].set_title('Feature Coefficients (Impact on Salary)')

    # D. Residual vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1,1], color='orange')
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Predicted Value')
    axes[1,1].set_ylabel('Residual')
    axes[1,1].set_title('Residuals vs Predicted Values')

    plt.tight_layout()
    plt.savefig('linear_regression_results.png')
    print("\n[OK] Visualizations saved to 'linear_regression_results.png'")
    plt.show()

if __name__ == "__main__":
    run_linear_regression()
