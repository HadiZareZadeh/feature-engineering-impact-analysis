# The Impact of Feature Engineering on Tabular Machine Learning Models

## Project Description

This project demonstrates the critical importance of feature engineering in machine learning. We compare model performance before and after applying various feature engineering techniques on the UCI Adult Income dataset. Feature engineering is often the most impactful step in the ML pipeline, and this project provides hands-on experience with the most important techniques.

**Why This Project Matters**:
- **Real-world Impact**: Feature engineering often has a larger impact on model performance than algorithm selection
- **Practical Skills**: Teaches essential preprocessing techniques used in every ML project
- **Data Understanding**: Forces deep understanding of the dataset and its characteristics
- **Performance Gains**: Demonstrates how proper feature engineering can significantly improve model accuracy
- **Industry Relevance**: Feature engineering is a core skill for data scientists and ML engineers

**Key Research Questions**:
- How much does feature engineering improve model performance?
- Which feature engineering techniques are most effective for different data types?
- How do encoding strategies affect model performance?
- What is the impact of feature interactions and polynomial features?
- How important is feature scaling for different algorithms?

## Dataset Description

**Dataset Name**: UCI Adult Income Dataset (also known as Census Income Dataset)

**Source**: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/adult)

**Dataset Details**:
- **Number of samples**: ~32,561 (after removing missing values)
- **Number of features**: 14 features
  - **Numerical features (6)**: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - **Categorical features (8)**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Target variable**: income (binary classification)
  - <=50K: Low income
  - >50K: High income
- **Task**: Binary classification (predicting income level)
- **Missing values**: Some features contain '?' which are treated as missing and removed

**Why This Dataset**:
- **Mixed data types**: Contains both numerical and categorical features, perfect for demonstrating various encoding techniques
- **Real-world problem**: Income prediction is a practical classification task with clear business value
- **Feature engineering opportunities**: Many categorical features that benefit from different encoding strategies (one-hot, label, target encoding)
- **Well-known benchmark**: Standard dataset for demonstrating feature engineering, widely used in ML education
- **Interpretable**: Results can be easily understood and explained, making it ideal for learning
- **Challenging**: Requires proper preprocessing to achieve good performance

**Data Loading**:
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df = df.dropna()
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real UCI Adult Income dataset loaded directly from the UCI repository.

## Project Structure

```
project3_feature_engineering/
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_baseline_model.ipynb
    ├── 03_feature_engineering.ipynb
    └── 04_comparison_analysis.ipynb
```

## Key Techniques Demonstrated

1. **Encoding**: One-hot encoding, label encoding, target encoding
2. **Scaling**: Standardization, normalization
3. **Feature Creation**: Interaction features, polynomial features
4. **Feature Selection**: Correlation analysis, importance-based selection
5. **Handling Missing Values**: Imputation strategies

## Learning Objectives

1. Understand the impact of feature engineering on model performance
2. Learn various encoding and scaling techniques
3. Explore feature interaction and creation
4. Compare baseline vs engineered features
5. Analyze feature importance

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter notebooks in order (01 → 04)
3. Run all cells to reproduce experiments

