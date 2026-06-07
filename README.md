# 📊 Bank Customer Churn Analysis & Machine Learning Modeling

An end-to-end data analytics and predictive modeling project that identifies key factors contributing to retail bank customer churn. Using Python, Jupyter Notebooks, and machine learning classifiers, this project cleans raw transaction records, performs comprehensive exploratory analysis, builds predictive models, and presents performance metrics via detailed charts and an interactive dashboard.

---

## 📂 Project Structure

```text
data analytics/
├── Customer bank churn.pptx             # Summary presentation of results and insights
└── Bank+Customer+Churn/
    └── Bank+Customer+Churn/
        ├── Bank_Churn_Analysis.ipynb    # Main Jupyter Notebook detailing the analysis & modeling
        ├── Bank_Churn_Analysis.html     # HTML export of the analysis notebook
        ├── Bank_Churn.csv               # Cleaned analytical dataset
        ├── Bank_Churn_Data_Dictionary.csv # Schema and column description guide
        ├── Bank_Churn_Messy.xlsx        # Original messy Excel spreadsheet (before cleaning)
        ├── advanced_churn_modeling.py   # Machine learning pipeline with evaluation
        ├── creative_churn_analysis.py   # Script for advanced EDA and visualizations
        ├── linear_regression_model.py   # Linear / Logistic baseline model execution
        ├── interactive_churn_dashboard.html # Standalone interactive web dashboard
        ├── regression_summary.txt       # Quantitative summary of regression coefficient details
        └── *.png                        # Visual charts (ROC curves, heatmaps, violin plots)
```

---

## ⚙️ Analytical Pipeline & Features

1. **Data Cleaning & Wrangling**:
   - Compares raw records in `Bank_Churn_Messy.xlsx` with standard formatting to parse fields, handle missing values, and prepare `Bank_Churn.csv` for modeling.
2. **Exploratory Data Analysis (EDA)**:
   - Evaluates demographic patterns, showing relationships like **Age vs. Churn** (visualized using violin plots) and **Geography vs. Churn**.
   - Generates a **Correlation Heatmap** to identify multicollinearity and highlight predictors most closely aligned with churn.
3. **Machine Learning Modeling**:
   - Builds classification models (such as Logistic Regression, Random Forest, and Decision Trees) to forecast the probability of a client leaving the bank.
   - Evaluates feature importance to reveal what drives churn (e.g., age, number of products, active membership status).
4. **Evaluation Metrics**:
   - Compares classifiers using **ROC curves (AUC score)**, **Precision-Recall curves**, **Confusion Matrices**, and coefficient tables.
5. **Interactive Dashboard**:
   - Packages insights into a user-friendly standalone HTML interface (`interactive_churn_dashboard.html`) for dynamic business reporting.

---

## 📊 Sample Visualizations Included

* **`correlation_heatmap.png`**: Visual grid showing relationships between numeric variables.
* **`advanced_feature_importance.png`**: Ranks bank metrics (like Credit Score, Balance, Active Status, Age) by predictive weight.
* **`age_churn_violin.png`**: Highlights the density distribution showing churn rates peaking at specific age cohorts.
* **`advanced_roc_curve.png` & `advanced_pr_curve.png`**: Performance evaluations comparing model sensitivities.

---

## 🛠️ Technology Stack

* **Language**: Python 3.x
* **Core Libraries**: Pandas, NumPy, OpenPyXL (Excel integration)
* **Visualization**: Matplotlib, Seaborn, Plotly/Bokeh (Interactive dashboard rendering)
* **Modeling & Stats**: Scikit-Learn (ML modeling), Statsmodels (Statistical summary coefficients)
* **Presentation**: Jupyter Notebook, HTML/CSS

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure Python 3.8+ is installed on your computer.

### 2. Create and Activate Virtual Environment
Navigate to the deep subfolder containing the Python scripts:
```bash
cd "Bank+Customer+Churn/Bank+Customer+Churn"
python -m venv venv
```
Activate the environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl statsmodels notebook
```

### 4. Running the Analysis
To view the notebook:
```bash
jupyter notebook Bank_Churn_Analysis.ipynb
```
To run the modeling script:
```bash
python advanced_churn_modeling.py
```
To run the interactive dashboard script:
```bash
python creative_churn_analysis.py
```
To view the interactive dashboard dashboard in your browser, double-click or open `interactive_churn_dashboard.html` directly.
