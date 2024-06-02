import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
math_data = pd.read_csv('student-mat.csv')
por_data = pd.read_csv('student-por.csv')

# Descriptive statistics for numerical and categorical data
print(math_data.describe())  # Summary statistics for numerical columns
print(math_data.describe(include=['O']))  # Summary statistics for categorical columns

# Check for missing values in the data
print(math_data.isnull().sum())

# Histogram of final grades for both datasets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(math_data['G3'], kde=True, bins=20)
plt.title('Distribution of Final Grades for Math Course')
plt.subplot(1, 2, 2)
sns.histplot(por_data['G3'], kde=True, bins=20)
plt.title('Distribution of Final Grades for Portuguese Course')
plt.show()

# Scatter plot of study time vs final grades
plt.figure(figsize=(10, 5))
sns.scatterplot(data=math_data, x='studytime', y='G3')
plt.title('Study Time vs Final Grades in Math Course')
plt.xlabel('Study Time')
plt.ylabel('Final Grade')
plt.show()

# Box plot of parental education level and final grades
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Medu', y='G3', data=math_data)
plt.title('Mother\'s Education vs Student\'s Final Grades (Math)')
plt.subplot(1, 2, 2)
sns.boxplot(x='Fedu', y='G3', data=math_data)
plt.title('Father\'s Education vs Student\'s Final Grades (Math)')
plt.show()

# Correlation matrix to find relationships between numerical variables
correlation_matrix = math_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title('Correlation Matrix for Math Course Data')
plt.show()
