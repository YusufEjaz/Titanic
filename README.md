# Titanic Passenger Data
---

## Introduction
This project involves an end-to-end analysis of the Titanic dataset to explore key patterns, clean and preprocess the data, and build a predictive model for survival. The Titanic dataset is a widely-used dataset in machine learning and data analysis, offering a rich variety of features (e.g., passenger demographics, ticket details, class, and survival status) for exploration.

The objectives of this project are:

1. To perform data cleaning and preprocessing, including handling missing values, outliers, and categorical variables.
2. To explore key correlations and patterns in the dataset using visualizations.
3. To build a logistic regression model to predict passenger survival based on key features.
4. To evaluate the model and identify areas for improvement.


## Steps Followed
### Step 1: Download and Load the Dataset
* The Titanic dataset was downloaded using the KaggleHub package.

```python
# Import necessary libraries
import kagglehub
import os

# Step 1: Download the Titanic dataset using kagglehub
path = kagglehub.dataset_download("alyelbadry/titanic-survive-model")
print("Path to dataset files:", path)

# Step 1.1: List all files in the downloaded path
files = os.listdir(path)
print("Files in the dataset folder:", files)
```
```
Path to dataset files: /.cache/kagglehub/datasets/alyelbadry/titanic-survive-model/versions/1
Files in the dataset folder: ['titanic-passengers.csv']
```
* The file `titanic-passengers.csv` was loaded into a pandas DataFrame.

```python
import pandas as pd
titanic_file_path = "/.cache/kagglehub/datasets/alyelbadry/titanic-survive-model/versions/1/titanic-passengers.csv"
df = pd.read_csv(titanic_file_path)
```
* Initial inspection (`head()`, `info()`) was performed to understand the structure and content of the dataset.

```python
# Load the Titanic dataset with the correct delimiter
df = pd.read_csv(titanic_file_path, delimiter=';')

# Step 2.1: Inspect the dataset again
print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary of the dataset:")
print(df.info())
```
```
First 5 rows of the dataset:
   PassengerId Survived  Pclass                                         Name  \
0          343       No       2                   Collander, Mr. Erik Gustaf   
1           76       No       3                      Moen, Mr. Sigurd Hansen   
2          641       No       3                       Jensen, Mr. Hans Peder   
3          568       No       3  Palsson, Mrs. Nils (Alma Cornelia Berglund)   
4          672       No       1                       Davidson, Mr. Thornton   

      Sex   Age  SibSp  Parch      Ticket     Fare  Cabin Embarked  
0    male  28.0      0      0      248740  13.0000    NaN        S  
1    male  25.0      0      0      348123   7.6500  F G73        S  
2    male  20.0      0      0      350050   7.8542    NaN        S  
3  female  29.0      0      4      349909  21.0750    NaN        S  
4    male  31.0      1      0  F.C. 12750  52.0000    B71        S  

Summary of the dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    object 
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(4), object(6)
memory usage: 83.7+ KB
None
```

### Step 2: Handle Missing Values
* Missing values in the dataset were identified.

```python
# Step 3.1: Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())
```
```
Missing values in each column:
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
* Specific strategies were applied:
 * `Age`: Filled with the mean value.
 * `Cabin`: Dropped due to a large number of missing values.
 * `Embarked`: Filled with the mode value. 

```python
# Step 3.2: Fill missing values in 'Age' with the mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Step 3.3: Drop the 'Cabin' column if too many values are missing
df.drop(columns=['Cabin'], inplace=True)

# Step 3.4: Fill missing values in 'Embarked' with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Verify that there are no more missing values
print("\nMissing values after handling:")
print(df.isnull().sum())

```
```
Missing values after handling:
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
```
### Step 3: Handle Outliers
* Outliers in the `Fare` and `Age` columns were identified using box plots.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Step 4.1: Box plot for 'Fare'
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

# Step 4.1: Box plot for 'Age'
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()
```

![](https://github.com/YusufEjaz/Titanic/blob/main/Images/preboxplotage.png)
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/preboxplot.png)

* The Interquartile Range (IQR) method was applied to cap outliers, ensuring extreme values did not skew the analysis.

```python
# Step 4.2: Define a function to cap outliers
def cap_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return series.clip(lower_bound, upper_bound)

# Apply the function to 'Fare' and 'Age'
df['Fare'] = cap_outliers(df['Fare'])
df['Age'] = cap_outliers(df['Age'])

# Verify changes with box plots again
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare (After Capping Outliers)")
plt.show()

sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age (After Capping Outliers)")
plt.show()
```
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/fare.png)
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/age.png)

### Step 4: Handle Categorical Encoding
* Categorical columns were converted into numerical format:
 * Sex was label-encoded (`male` → 0, `female` → 1). 

```python
# Step 5.1: Label encode the 'Sex' column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("Unique values in 'Sex' after encoding:", df['Sex'].unique())
```
```
Unique values in 'Sex' after encoding: [0 1]
```

 * `Embarked` was one-hot encoded to create separate binary columns for each embarkation point (`Q` and `S`).

```python
# Step 5.2: One-hot encode the 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("Columns after one-hot encoding 'Embarked':", df.columns)
```
```
Columns after one-hot encoding 'Embarked': Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked_Q', 'Embarked_S'],
      dtype='object')
```

### Step 5: Visualize Key Patterns
Several visualizations were created to explore the dataset:

1. Correlation Heatmap: Showed relationships between features, highlighting the strong correlation of `Survived` with `Sex`, `Pclass`, and `Fare`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Step 6.1: Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
```
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/correlation.png)

Key Observations:
Strong Negative Correlation:

Pclass and Fare have a strong negative correlation of -0.72.
This means passengers in lower classes (higher Pclass values) tend to have lower ticket fares.
Moderate Positive Correlations with Survived:

Sex and Survived: ~0.25
Gender (Sex) is moderately correlated with survival. Female passengers (Sex=1) were more likely to survive compared to males (Sex=0).
Fare and Survived: ~0.23
Higher ticket fares are associated with a greater likelihood of survival, which aligns with the idea that passengers in higher classes had better access to lifeboats.
Weak or No Correlation:

Age and Survived: ~-0.08
Age does not strongly correlate with survival, though younger passengers may have had a slight advantage.
Parch and Survived: ~0.25
Traveling with family (parent/child relationships captured in Parch) shows a weak positive correlation with survival.
Other Relationships:

SibSp and Parch: ~0.41
Passengers traveling with siblings/spouses often had parents/children traveling with them as well.
Embarked_Q and Embarked_S: ~-0.50
Indicates that passengers embarking from Q and S are negatively related due to one-hot encoding (only one embarkation point can apply per passenger).

2. Bar Plot of Survival by Sex: Revealed that female passengers had a significantly higher survival rate than males.

```python
# Step 6.2: Bar plot for survival distribution
sns.countplot(x='Survived', data=df)
plt.title("Survival Counts")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
```
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/survival%20counts.png)


3. Bar Plot of Survival by Pclass: Showed that passengers in first class were more likely to survive than those in second or third class.

```python
# Survival by Pclass
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Passenger Class")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passenger Count")
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()
```

![](https://github.com/YusufEjaz/Titanic/blob/main/Images/sur%20pclass.png)


4. Age and Fare Distributions: Visualized the spread of ages and ticket fares across passengers.

```python
# Step 6.3: Histogram for Age distribution
df['Age'].hist(bins=20, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
# Step 6.4: Histogram for Fare distribution
df['Fare'].hist(bins=20, edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

```

![](https://github.com/YusufEjaz/Titanic/blob/main/Images/age%20d.png)
![](https://github.com/YusufEjaz/Titanic/blob/main/Images/fare%20d.png)


### Step 6: Build a Logistic Regression Model

* Features and target (`Survived`) were separated, and data was split into training and test sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1.1: Separate features and target
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])  # Drop unnecessary columns
y = df['Survived']

# Step 1.2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1.3: Standardize numerical features (e.g., Age, Fare)
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

```
```python
from sklearn.linear_model import LogisticRegression

# Step 2: Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

```

```
LogisticRegression(random_state=42)
```

* Numerical features (`Age`, `Fare`) were standardized using `StandardScaler`.
* A logistic regression model was trained on the training data to predict survival.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 3.1: Predict the test data
y_pred = model.predict(X_test)

# Step 3.2: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
```

```
Model Accuracy: 0.79

Confusion Matrix:
 [[97 17]
 [21 44]]

Classification Report:
               precision    recall  f1-score   support

          No       0.82      0.85      0.84       114
         Yes       0.72      0.68      0.70        65

    accuracy                           0.79       179
   macro avg       0.77      0.76      0.77       179
weighted avg       0.79      0.79      0.79       179
```

### Step 7: Evaluate the Model

* The model achieved an accuracy of 79% on the test dataset.
* Confusion Matrix:
 * True Negatives: 97
 * True Positives: 44
 * False Negatives: 21 (missed survivors)
 * False Positives: 17
* Classification Metrics:
 * Precision, recall, and F1-score showed strong performance for the majority class (`Not Survived`) but lower recall for the minority class (`Survived`). 

> The prediction model, built using logistic regression, predicts whether a passenger survived the Titanic disaster based on features like gender, passenger class, age, and fare. The model achieved an accuracy of 79% on the test data. It correctly predicted 97 passengers who did not survive and 44 passengers who survived. However, it missed 21 actual survivors (false negatives) and incorrectly predicted survival for 17 passengers who did not survive (false positives). The precision for predicting survival was 72%, while the recall was 68%, indicating the model performs moderately well in identifying survivors but struggles slightly with false negatives. These results suggest that gender (`Sex`), passenger class (`Pclass`), and fare (`Fare`) are strong predictors of survival, while further improvements could be made to better capture the minority class (`Survived`).
