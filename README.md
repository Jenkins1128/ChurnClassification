# Telecommunication Churn Classification

Isaiah Jenkins

## Project Overview

This project analyzes a telecommunication churn dataset to predict customer churn using classification models. The goal is to identify factors that influence customer retention, as retaining existing customers is typically more cost-effective than acquiring new ones. The analysis leverages Python libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Imblearn for data processing, modeling, and visualization. Three classification models—Logistic Regression, Linear SVC, and Random Forest Classifier—were implemented, with the Random Forest Classifier achieving the highest accuracy after dataset balancing.

## Dataset

The dataset, `ChurnData.csv`, contains 200 records with 28 features related to customer demographics, account information, and service usage. Key features include:
- **Demographic Features**: `age`, `income`, `ed` (education level), `employ` (employment status).
- **Account Information**: `tenure`, `longmon` (monthly long-distance charges), `tollmon` (toll charges), `cardmon` (calling card charges), `wiremon` (wireless charges).
- **Service Usage**: `equip`, `callcard`, `wireless`, `voice`, `pager`, `internet`, `callwait`, `confer`, `ebill`.
- **Target Variable**: `churn` (binary: 1 for churn, 0 for no churn).
- **Preprocessing**: Handled missing values, scaled numerical features using StandardScaler and RobustScaler, and addressed class imbalance using RandomUnderSampler, resulting in a balanced dataset.

## Analysis

The analysis included:
1. **Data Exploration**: Inspected dataset structure, checked for missing values (none found), and computed descriptive statistics for numerical features and value counts for categorical features.
2. **Data Visualization**: Generated correlation matrices and box plots to identify relationships and potential outliers.
3. **Feature Engineering**: Scaled numerical features and dropped highly correlated features (e.g., `loglong`, `logtoll`, `lninc`) to reduce multicollinearity.
4. **Modeling**:
   - **Logistic Regression**: Initial accuracy of ~70%.
   - **Linear SVC**: Moderate improvement over logistic regression.
   - **Random Forest Classifier**: Achieved the highest accuracy of 82.5% with balanced classes, improving the F1-score for the churn class by ~18%.
5. **Class Imbalance Handling**: Applied RandomUnderSampler to balance the dataset, significantly improving model performance.

## Key Findings

- **Model Performance**: The Random Forest Classifier outperformed Logistic Regression and Linear SVC, achieving an accuracy of 82.5% and improved F1-score for the churn class after balancing.
- **Feature Insights**: Features like `tenure`, `age`, `income`, and service usage (e.g., `callcard`, `wireless`) were significant predictors of churn.
- **Challenges**: Initial class imbalance skewed model performance toward the majority class. RandomUnderSampler improved recall and precision for the minority (churn) class.
- **Key Takeaway**: Preprocessing (scaling, handling multicollinearity) and balancing the dataset were critical for improving model accuracy and predictive power.

## Installation

To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Download the `ChurnData.csv` dataset and place it in the `data/` directory.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/telecom-churn-classification.git
   cd telecom-churn-classification
   ```

2. Set up the dataset:
   - Place `ChurnData.csv` in the `data/` directory.

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook churn_classification-2.ipynb
   ```

4. Follow the notebook to explore data, train models, and review results.

## Next Steps

- **Hyperparameter Tuning**: Optimize Random Forest Classifier parameters (e.g., `n_estimators`, `max_depth`) to further improve predictive power.
- **Feature Expansion**: Incorporate additional features or derive new ones (e.g., interaction terms) to capture more complex patterns.
- **Alternative Models**: Experiment with other algorithms like Gradient Boosting (e.g., XGBoost, LightGBM) or neural networks.
- **Cross-Validation**: Implement k-fold cross-validation to ensure robust model evaluation.
- **Real-World Application**: Deploy the model in a production environment to predict churn and inform customer retention strategies.
