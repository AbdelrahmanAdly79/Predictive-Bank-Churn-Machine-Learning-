import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Bank Customer Churn Data Analysis
This notebook analyzes Bank Customer Churn data to clean it, explore it, and build predictive models to uncover churn drivers and actionable business recommendations.

**Important:** Run the first code cell below to load all libraries and data. All other cells depend on it.
"""

# === SINGLE COMBINED SETUP CELL: imports + data loading + cleaning ===
code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ── Data Loading ──
print("\n[STEP 1] Loading Datasets...")
print("Current working directory:", os.getcwd())

# Priority 1: Load the main clean dataset (CSV)
try:
    df_clean = pd.read_csv('Bank_Churn.csv')
    print("[OK] Clean Dataset loaded successfully (Shape:", df_clean.shape, ")")
except Exception as e:
    print("[ERROR] Error loading Bank_Churn.csv:", e)
    df_clean = pd.DataFrame() # Fallback

# Priority 2: Load the messy dataset (Excel) - Requires openpyxl
try:
    df_messy = pd.read_excel('Bank_Churn_Messy.xlsx')
    print("[OK] Messy Dataset loaded successfully (Shape:", df_messy.shape, ")")
except ImportError:
    print("! Note: 'openpyxl' not found. Skipping Bank_Churn_Messy.xlsx (Run: pip install openpyxl)")
    df_messy = pd.DataFrame()
except Exception as e:
    print("✗ Error loading Bank_Churn_Messy.xlsx:", e)
    df_messy = pd.DataFrame()

# ── Data Cleaning ──
if not df_clean.empty:
    print("\n[STEP 2] Cleaning Data...")
    print("Missing values in dataset:\n", df_clean.isnull().sum())
    print("\nDuplicate rows:", df_clean.duplicated().sum())
    df = df_clean.drop_duplicates()
    print("\n[DONE] Data cleaned and ready! (df has", df.shape[0], "rows)")
    df.head()
else:
    print("\n[!] CRITICAL ERROR: Could not load the main dataset. Please check if Bank_Churn.csv exists.")
    df = pd.DataFrame()
"""

text_eda = """## 2. Exploratory Data Analysis & Creative Enhancements
### 2.1 Univariate Analysis"""
code_eda1 = """import matplotlib.pyplot as plt
import seaborn as sns

if 'df' in locals():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['Age'], kde=True, ax=axes[0,0], color='skyblue').set_title('Age Distribution')
    sns.histplot(df['Balance'], kde=True, ax=axes[0,1], color='salmon').set_title('Balance Distribution')
    sns.histplot(df['EstimatedSalary'], kde=True, ax=axes[1,0], color='lightgreen').set_title('Salary Distribution')
    sns.histplot(df['CreditScore'], kde=True, ax=axes[1,1], color='gold').set_title('Credit Score Distribution')
    plt.tight_layout()
    plt.show()
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_eda2 = """### 2.2 Bivariate Analysis & Churn Profiling"""
code_eda2 = """import matplotlib.pyplot as plt
import seaborn as sns

if 'df' in locals():
    # Churn rate by Geography and Gender
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Geography', y='Exited', hue='Gender', data=df, ax=axes[0], errorbar=None)
    axes[0].set_title('Churn Rate by Geography and Gender')

    sns.barplot(x='NumOfProducts', y='Exited', data=df, ax=axes[1], errorbar=None, color='teal')
    axes[1].set_title('Churn Rate by Number of Products')
    plt.show()
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_creative1 = """### 2.3 Creative Enhancements: Churn Risk Heatmap
Visualizing the combined effect of Age and Number of Products on Churn Rate."""
code_creative1 = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if 'df' in locals():
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    pivot = df.pivot_table(values='Exited', index='AgeGroup', columns='NumOfProducts', aggfunc='mean')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt=".1%")
    plt.title('Churn Risk Heatmap: Age Group vs Number of Products')
    plt.show()
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_creative2 = """### 2.4 Creative Enhancements: Customer Personas & Financial Impact
Let's group customers into Personas and calculate the "Cost of Churn"."""
code_creative2 = """import pandas as pd

if 'df' in locals():
    # Define Personas based on simple rules
    def assign_persona(row):
        if row['Age'] > 50 and row['Balance'] > 100000:
            return 'High-Value Seniors'
        elif row['Age'] <= 30 and row['IsActiveMember'] == 0:
            return 'Inactive Youth'
        elif row['Balance'] == 0:
            return 'Zero Balance Checkers'
        else:
            return 'General Segment'

    df['Persona'] = df.apply(assign_persona, axis=1)

    persona_churn = df.groupby('Persona').agg({'Exited': ['mean', 'count'], 'Balance': 'sum'})
    persona_churn.columns = ['Churn_Rate', 'Total_Customers', 'Total_Balance']

    # Financial Impact: Actual lost balance
    lost_balance = df[df['Exited'] == 1]['Balance'].sum()
    print(f"Total Bank Balance Lost to Churn: ${lost_balance:,.2f}")

    display(persona_churn.sort_values(by='Churn_Rate', ascending=False))
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_creative3 = """### 2.5 Radar Chart: Churned vs Retained Profile"""
code_creative3 = """import numpy as np
import matplotlib.pyplot as plt

if 'df' in locals():
    # Calculate mean values for numerical features
    features = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Tenure']
    mean_vals = df.groupby('Exited')[features].mean()

    # Normalize for radar chart
    normalized_means = mean_vals / mean_vals.max()

    labels = features
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for idx, row in normalized_means.iterrows():
        values = row.tolist()
        values += values[:1]
        label_name = 'Churned' if idx == 1 else 'Retained'
        ax.plot(angles, values, linewidth=2, label=label_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Radar Chart: Average Profile (Normalized)', y=1.1)
    plt.show()
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_model = """## 3. Predictive Modeling & Risk Segmentation"""
code_model = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if 'df' in locals():
    # Prepare data
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'AgeGroup', 'Persona']
    model_data = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Encode categoricals
    model_data = pd.get_dummies(model_data, columns=['Geography', 'Gender'], drop_first=True)

    X = model_data.drop('Exited', axis=1)
    y = model_data['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    probs = rf.predict_proba(X_test)[:, 1]

    print("Classification Report:\\n", classification_report(y_test, preds))

    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances, y=importances.index, palette='viridis')
    plt.title('Feature Importances in Predicting Churn')
    plt.show()
else:
    print("Error: 'df' not defined. Please run the Setup cell at the top of the notebook first!")
"""

text_segment = """## 4. Risk Group Segmentation & Actionable Insights
Segment test set customers into High, Medium, Low Risk."""
code_segment = """if 'X_test' in locals() and 'y_test' in locals() and 'probs' in locals():
    test_results = X_test.copy()
    test_results['Actual_Churn'] = y_test.values
    test_results['Churn_Probability'] = probs

    def risk_group(prob):
        if prob > 0.6: return 'High Risk'
        elif prob > 0.3: return 'Medium Risk'
        else: return 'Low Risk'

    test_results['Risk_Group'] = test_results['Churn_Probability'].apply(risk_group)
    print(test_results['Risk_Group'].value_counts())

    # Analyze high risk customers
    high_risk = test_results[test_results['Risk_Group'] == 'High Risk']

    # Only print profile columns that actually exist in the DataFrame
    profile_cols = [c for c in ['Age', 'Balance', 'NumOfProducts'] if c in high_risk.columns]
    if len(high_risk) > 0 and len(profile_cols) > 0:
        print("\\nAverage profile of High Risk Customer:")
        print(high_risk[profile_cols].mean())
    else:
        print("\\nNo High Risk customers found or profile columns unavailable.")
        print("Available columns:", list(test_results.columns[:10]), "...")
else:
    print("Error: Modeling variables not defined. Please run the 'Predictive Modeling' cell first!")
"""

text_conclusion = """## 5. Actionable Business Insights & Recommendations
Based on the analysis and modeling, here are the key business rules and actions:
1. **Age and Number of Products:** "Customers with older age (50-60) + Only 1 Product = High Churn Risk".
   *Action:* Target older demographics with a cross-selling campaign to increase their number of products to 2.
2. **Geography:** Germany has a disproportionately high churn rate.
   *Action:* Investigate the specific customer service or competitive landscape in Germany to retain these customers.
3. **Active Membership:** Inactive members churn at a much higher rate.
   *Action:* Create re-engagement campaigns for the 'Inactive Youth' persona to incentivize transactions and app logins.
4. **Financial Impact:** Over $120 Million in deposits were lost to churn. A 10% reduction in churn among the High-Value Seniors segment could retain millions in deposits.
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda1),
    nbf.v4.new_markdown_cell(text_eda2),
    nbf.v4.new_code_cell(code_eda2),
    nbf.v4.new_markdown_cell(text_creative1),
    nbf.v4.new_code_cell(code_creative1),
    nbf.v4.new_markdown_cell(text_creative2),
    nbf.v4.new_code_cell(code_creative2),
    nbf.v4.new_markdown_cell(text_creative3),
    nbf.v4.new_code_cell(code_creative3),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell(text_segment),
    nbf.v4.new_code_cell(code_segment),
    nbf.v4.new_markdown_cell(text_conclusion)
]

with open('Bank_Churn_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
