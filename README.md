# Breast Cancer Classification using Machine Learning

This project aims to build a machine learning model to classify breast cancer as either malignant or benign based on given features. Leveraging data preprocessing, exploratory data analysis (EDA), and model evaluation, the project achieves accurate predictions for breast cancer classification.

## Project Overview
Breast cancer is one of the most common cancers affecting women globally. Early and accurate diagnosis can significantly improve patient outcomes. This project uses machine learning algorithms to classify tumors as malignant or benign based on a dataset of tumor characteristics.

## Features of the Project
- **Data Preprocessing**: Handling missing values, scaling features, and encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualizing correlations and distributions of tumor characteristics.
- **Machine Learning Models**: Implementing various ML models, including:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
- **Model Evaluation**: Using metrics such as accuracy, precision, recall, and F1 score.

## Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Features**: Radius, texture, perimeter, area, smoothness, etc.
- **Target**: Tumor type (Malignant/Benign)

## Project Steps
1. **Data Loading**: Loading the dataset into a Pandas DataFrame.
2. **Data Preprocessing**: Cleaning and preparing the data for analysis.
3. **EDA**: Visualizing feature distributions and correlations.
4. **Model Building**: Training various ML models.
5. **Model Evaluation**: Evaluating model performance using accuracy and other metrics.
6. **Prediction**: Predicting tumor type for new data points.

## Results
The best-performing model achieved:
- **Accuracy**: 95%
- **Precision**: 96%
- **Recall**: 94%
- **F1 Score**: 95%

## Tools and Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Visualization
Below is a conceptual diagram representing the project's workflow:

![Breast Cancer Classification](images/diagram.png)

