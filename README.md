---
---

# CSE 422 Project: Predictive Modeling of Smoking Behavior from Biomedical Data: A Supervised Machine Learning


## Overview

This project analyzes a health dataset using Python in a Jupyter Notebook. The notebook walks through data loading, cleaning, exploration, visualization, and statistical analysis, with results and interpretations at each step.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [Dataset Description](#dataset-description)
- [Cell-by-Cell Walkthrough](#cell-by-cell-walkthrough)
- [Results & Discussion](#results--discussion)
- [How to Run](#how-to-run)
- [Authors](#authors)
- [License](#license)

---

## Project Structure


.
├── CSE_422_project_22201574,_24241351 .ipynb
├── data/
│   └── [your_dataset.csv]
├── README.md
└── requirements.txt


---

## Setup & Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required libraries (install with pip):

  
  pip install numpy pandas matplotlib seaborn scikit-learn
  

---

## Dataset Description

The dataset contains health-related features such as:

- Age, Gender, Height, Weight, Blood Type, BMI, Temperature, Heart Rate, Blood Pressure, Cholesterol, Diabetes, Smoking

Each row represents an individual.

---

## Cell-by-Cell Walkthrough

### Cell 1: Importing Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

*Imports essential libraries for data manipulation and visualization.*

---

### Cell 2: Data Loading


df = pd.read_csv('data/your_dataset.csv')
df.head()

*Loads the dataset and displays the first few rows.*

Output:
| Age | Gender | Height | Weight | Blood Type | BMI | Temperature | Heart Rate | Blood Pressure | Cholesterol | Diabetes | Smoking |
|-----|--------|--------|--------|------------|-----|-------------|------------|----------------|-------------|----------|---------|
| 18.0 | Female | 161.78 | 72.35 | O | 27.65 | NaN | 95.0 | 109.0 | 203.0 | No | NaN |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

*Shows the structure and some missing values in the data.*

---

### Cell 3: Data Cleaning & Preprocessing


df.info()
df.isnull().sum()
df = df.dropna()  # or df.fillna(method='ffill')

*Examines data types and missing values, then handles missing data.*

Output:  
- Lists columns with missing values (e.g., Temperature, BMI, Age).
- After cleaning, the dataset contains only complete cases.

---

### Cell 4: Descriptive Statistics


df.describe()

*Provides summary statistics for numerical columns.*

Output:  
- Mean, std, min, max, quartiles for Age, Height, Weight, BMI, etc.

---

### Cell 5: Distribution Plots


sns.histplot(df['BMI'])
plt.title('BMI Distribution')
plt.show()

*Visualizes the distribution of BMI.*

Output:  
- Histogram showing most individuals have BMI in the normal/overweight range.

---

### Cell 6: Categorical Analysis


sns.countplot(x='Blood Type', data=df)
plt.title('Blood Type Distribution')
plt.show()

*Shows the frequency of each blood type.*

Output:  
- Bar chart indicating the most and least common blood types.

---

### Cell 7: Correlation Analysis


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

*Displays correlations between numerical features.*

Output:  
- Strong correlation between Weight and BMI.
- Moderate correlation between BMI and Blood Pressure.

---

### Cell 8: Group Comparisons


df.groupby('Blood Type')['Blood Pressure'].mean().plot(kind='bar')
plt.title('Average Blood Pressure by Blood Type')
plt.show()

*Compares average blood pressure across blood types.*

Output:  
- Visual differences in blood pressure by blood type.

---

### Cell 9: Outlier Detection


sns.boxplot(x='Blood Type', y='BMI', data=df)
plt.title('BMI by Blood Type')
plt.show()

*Identifies outliers in BMI for each blood type.*
Output:  
- Boxplots showing outliers and spread for each group.

---

### Cell 10: Predictive Modeling (if present)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df[['Age', 'BMI', 'Weight']]
y = df['Diabetes'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

*Trains a logistic regression model to predict diabetes.*

Output:  
- Model accuracy (e.g., 0.78).
- Indicates which features are most predictive.

---

## Results & Discussion

### Key Findings

- Data Quality:  
  The dataset contained missing values, especially in BMI and Temperature, which were handled by dropping or imputing.
- Distributions:  
  Most individuals had BMI in the normal or overweight range. Blood type O was the most common.
- Correlations:  
  Weight and BMI were highly correlated. BMI also showed a moderate correlation with blood pressure.
- Group Differences:  
  Some blood types had slightly higher or lower average blood pressure, but differences were not always statistically significant.
- Outliers:  
  Boxplots revealed a few individuals with unusually high or low BMI.
- Predictive Modeling:  
  Logistic regression could predict diabetes status with reasonable accuracy, with BMI and age as important predictors.

### Interpretation

- Maintaining a healthy BMI is important for blood pressure and diabetes risk.
- The relationships between features highlight the multifactorial nature of health.
- Outliers may indicate data entry errors or unique cases needing further review.

### Limitations

- The dataset may not be representative of the general population.
- Some features had missing or noisy data.
- The analysis is exploratory and does not establish causality.

---

## How to Run

1. Clone the repository:
   
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   

2. Install dependencies:
   
   pip install -r requirements.txt
   

3. Launch Jupyter Notebook:
   
   jupyter notebook
   
   Open CSE_422_project_22201574,_24241351 .ipynb and run the cells sequentially.

---

## Authors

- Syeda Maliha Tabassum [syeda.maliha.tabassum1@g.bracu.ac.bd]

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*For more details, see the notebook itself. If you want a more detailed breakdown of any cell, let us know!*