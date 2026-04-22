import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set(style="whitegrid")

# ## PART 1: DATA UNDERSTANDING & PREPARATION


# STEP 1: LOADING THE DATA
## Importing insurance dataset

df = pd.read_csv("insurance.csv")

print(df.head())


# STEP 2: SUMMARY STATISTICS
## Understanding data distribution and relationships

print(df.describe())
print(df.cov(numeric_only=True))
print(df.corr(numeric_only=True))


# STEP 3: MISSING VALUE ANALYSIS
## Checking if dataset contains null values

print(df.isnull().sum())


# STEP 4: DATA CLEANING
## Removing duplicate entries

df = df.drop_duplicates()


# STEP 5: DATA TRANSFORMATION
## Converting categorical variables into numerical form

df['sex'] = df['sex'].map({'male':0,'female':1})
df['smoker'] = df['smoker'].map({'yes':1,'no':0})
df = pd.get_dummies(df, columns=['region'], drop_first=True)



# ## PART 2: VISUALIZATION & OBJECTIVE ANALYSIS


# OBJECTIVE 1: CORRELATION ANALYSIS
## Understanding relationships between all variables

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# OBJECTIVE 2: DISTRIBUTION ANALYSIS
## Distribution of charges

plt.figure(figsize=(8,6))
sns.histplot(df['charges'], kde=True, color='#6A5ACD')
plt.title("Distribution Plot of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()


# OBJECTIVE 3: BMI DISTRIBUTION
## Distribution of BMI

plt.figure(figsize=(8,6))
sns.histplot(df['bmi'], kde=True, color='#FF8C00')
plt.title("Distribution Plot of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()


# OBJECTIVE 4: OUTLIER DETECTION (CHARGES)
## Using boxplot to detect outliers

plt.figure(figsize=(8,6))
sns.boxplot(x=df['charges'], color='#20B2AA')
plt.title("Boxplot of Charges (Outlier Detection)")
plt.xlabel("Charges")
plt.show()


# OBJECTIVE 5: OUTLIER DETECTION (BMI)
## Using boxplot to detect BMI outliers

plt.figure(figsize=(8,6))
sns.boxplot(x=df['bmi'], color='#3CB371')
plt.title("Boxplot of BMI (Outlier Detection)")
plt.xlabel("BMI")
plt.show()


# OBJECTIVE 6: AGE VS CHARGES
## Relationship between age and insurance charges

plt.figure(figsize=(8,6))
sns.scatterplot(x="age", y="charges", data=df, hue="smoker", palette="Set2")
plt.title("Objective 1: Age vs Charges")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.tight_layout()
plt.show()


# OBJECTIVE 7: BMI VS CHARGES
## Relationship between BMI and charges

plt.figure(figsize=(8,6))
sns.scatterplot(x="bmi", y="charges", data=df, hue="smoker", palette="coolwarm")
plt.title("Objective 2: BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.tight_layout()
plt.show()



# ## PART 3: OUTLIER REMOVAL (IQR METHOD)


# STEP 6: IQR METHOD FOR CHARGES

Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['charges'] >= lower) & (df['charges'] <= upper)]


# STEP 7: IQR METHOD FOR BMI

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['bmi'] >= lower) & (df['bmi'] <= upper)]


# ## PART 4: MACHINE LEARNING (SLR MODEL)


# OBJECTIVE 8: SIMPLE LINEAR REGRESSION
## Predicting charges using BMI

X = df[['bmi']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# OBJECTIVE 9: REGRESSION VISUALIZATION
## Plotting regression line

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='#4682B4', label='Actual Data')
plt.plot(X_test, y_pred, color='#FF4500', linewidth=2.5, label='Regression Line')
plt.title("Simple Linear Regression (BMI vs Charges)")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.legend()
plt.show()



# ## PART 5: ADVANCED VISUALIZATION


fig, ax = plt.subplots(2, 2, figsize=(12,8))

sns.histplot(df['charges'], kde=True, color='purple', ax=ax[0,0])
ax[0,0].set_title("Charges Distribution")

sns.histplot(df['bmi'], kde=True, color='orange', ax=ax[0,1])
ax[0,1].set_title("BMI Distribution")

sns.boxplot(x=df['charges'], color='cyan', ax=ax[1,0])
ax[1,0].set_title("Charges Outliers")

sns.boxplot(x=df['bmi'], color='green', ax=ax[1,1])
ax[1,1].set_title("BMI Outliers")

plt.tight_layout()
plt.show()


colors = sns.color_palette("viridis", len(df['children'].unique()))

fig, ax = plt.subplots(1, 2, figsize=(12,5))

sns.barplot(x='smoker', y='charges', data=df, ax=ax[0])
ax[0].set_title("Smoker vs Charges")

sns.barplot(x='children', y='charges', data=df,ax=ax[1])
ax[1].set_title("Children vs Charges")

plt.tight_layout()
plt.show()




# ## PART 6: HYPOTHESIS TESTING (T-TEST)


# OBJECTIVE 10: T-TEST ANALYSIS
## Comparing smokers vs non-smokers charges

group1 = df[df['smoker'] == 1]['charges']
group2 = df[df['smoker'] == 0]['charges']

t_stat, p_value = stats.ttest_ind(group1, group2)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis: Significant difference in charges between smokers and non-smokers")
else:
    print("Fail to Reject Null Hypothesis: No significant difference")




