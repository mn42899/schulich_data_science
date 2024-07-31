#USE THIS IMPORT LIST

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, roc_curve, confusion_matrix, classification_report, mean_squared_error, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV #GridSearch is for hyperparameter tuning
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier



#CODING NOTES HOW TO INPUT DATA VALUES

file_path = '/Users/steven.souksavath/Downloads/churn_data.csv'
df = pd.read_csv(file_path)

#If its a website

df = pd.read_csv('https://raw.githubusercontent.com/delinai/schulich_ds1_2024/main/Datasets/Assignment1_StreamFlow_Subscription_Data.csv')

df = pd.read_csv('https://raw.githubusercontent.com/mn42899/schulich_data_science/main/Dataset.csv')





#CODING HOW TO UNDERSTAND THE DATA STRUCTURE

df.info() # Data type
df.head() # First 5 rows of the DataFrame
df.describe() # Overview of DataFrame (Count, Mean, Std, Min, 25%, 50%, 75%, Max)
df.shape

# Checking and dropping Null Values
df.isnull().sum()
df.dropna()



#Drop uneeded variables
df.drop(' <DF column >', axis=1, inplace=True)

#Dropping columns with NaN volumes (Multi-Columns)
df = df.drop(columns=['Unnamed: 0', 'id'])

#Convert Output variable to Binary
df['Attrition'] = df['Attrition'].replace({'No': 0, 'Yes': 1})

#Map Gender to Binary
df['Gender'] = df['Gender'].map({'Female': 0, 'Male' : 1})



#Imputing 
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())

#KNN Imputing
knn_imputer = KNNImputer(n_neighbors=5)








#EXAMPLES OF EXPLORATORY DATA ANALYSIS (EDA)


#HIST PLOTS

#Age distribution across dataframe
sns.histplot (x=df['Age'])

# examine churn via univariate analysis with a bar chart
sns.histplot(x=df['Churn'])

#Create # of churned based on location
loc = c_df.groupby('Location')
sns.barplot(x=loc['User_ID'].count().index, y=loc['User_ID'].count())
plt.show()

# Does the travel class affect satisfaction?
sns.displot(data=df,x='Class',col='satisfaction',color='#ff9999')


#COUNT PLOT
# Proportion of Loyal Customers and Disloyal Customers
sns.countplot(data=df,x='Customer Type',palette='hls')


#BOX PLOTS

#Checking number of Y across X
sns.boxplot(x='Attrition', y='Age', data=df)
plt.xlabel('Attrition')
plt.ylabel('Age')
plt.title('Age Distribution by Number of Attrition')
plt.show()


#Univariate boxplots to detect outliers for specific columns
sns.boxplot(y=df['Month_2_Spend'])
plt.show()

sns.boxplot(x='hour', y='count', data=bikes)


# compare age vs churn

# group by age
ages = df.groupby('Age')

# generate bar plot for how many customers have churned per age
# Note: Using count instead of sum would be incorrect, count will just return the number of people at that age
# Assuming has churned is represented as 1, then the sum of the churn column represents how many customers have churned at that age
plt.figure(figsize=(20,10))
sns.barplot(x=ages['Churn'].sum().index, y=ages['Churn'].sum())
plt.title("Have Churned")
plt.show()

# check for people who have not churned
noChurn = df[df['Churn'] == 0]
noChurnAge = noChurn.groupby('Age')
# using count here since sum of all who have not churned would be 0
plt.figure(figsize=(20,10))
sns.barplot(x=noChurnAge['Churn'].count().index, y=noChurnAge['Churn'].count())
plt.title("Have Not Churned")
plt.show()

## data exploration - Multi box plot to see patterns 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
sns.boxplot(x='Attrition', y='Age', data=df, ax=ax1)
sns.boxplot(x='Attrition', y='Gender', data=df, ax=ax2)
sns.boxplot(x='Attrition', y='chol', data=df, ax=ax3)
sns.boxplot(x='Attrition', y='thalach', data=df, ax=ax4)


# multivariaate analysis of location vs churn

churn = df[df['Churn']==1]
loc = churn.groupby('Location')
sns.barplot(x=loc['Churn'].count().index, y=loc['Churn'].count())
plt.title('Have Churned')
plt.show()

noChurn = df[df['Churn']==0]
loc = noChurn.groupby('Location')
sns.barplot(x=loc['Churn'].count().index, y=loc['Churn'].count())
plt.title('Have Not Churned')
plt.show()

#HEATMAP
plt.figure(figsize=(18, 10))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
# Display the plot
plt.show()


#CORRELATION MATRIX SETUP

# Correlation Analysis / Matrix

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numerical_features].corr()

# Correlation with the target variable
attr_corr = corr_matrix['Attrition'].drop('Attrition').sort_values(ascending=False)
print("Attrition(Original Features):")
print(attr_corr)

# Bar plot for correlation with attrition
plt.figure(figsize=(10, 8))
sns.barplot(x=attr_corr.values, y=attr_corr.index, palette='viridis')
plt.title('Correlation with Attrition (Original Features)', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.show()









#FEATURE ENGINEERING

#Create features to separate variables in quarters (Example)

df['Q1_Spend'] = (df['Month_1_Spend'] + df['Month_2_Spend'] + df['Month_3_Spend']) / 3
df['Q1_Items'] = (df['Month_1_Items'] + df['Month_2_Items'] + df['Month_3_Items']) / 3
df['Q1_Support_Calls'] = (df['Month_1_Support_Calls'] + df['Month_2_Support_Calls'] + df['Month_3_Support_Calls']) / 3

df['Q2_Spend'] = (df['Month_4_Spend'] + df['Month_5_Spend'] + df['Month_6_Spend']) / 3
df['Q2_Items'] = (df['Month_4_Items'] + df['Month_5_Items'] + df['Month_6_Items']) / 3
df['Q2_Support_Calls'] = (df['Month_4_Support_Calls'] + df['Month_5_Support_Calls'] + df['Month_6_Support_Calls']) / 3

df['Q3_Spend'] = (df['Month_7_Spend'] + df['Month_8_Spend'] + df['Month_9_Spend']) / 3
df['Q3_Items'] = (df['Month_7_Items'] + df['Month_8_Items'] + df['Month_9_Items']) / 3
df['Q3_Support_Calls'] = (df['Month_7_Support_Calls'] + df['Month_8_Support_Calls'] + df['Month_9_Support_Calls']) / 3

df['Q4_Spend'] = (df['Month_10_Spend'] + df['Month_11_Spend'] + df['Month_12_Spend']) / 3
df['Q4_Items'] = (df['Month_10_Items'] + df['Month_11_Items'] + df['Month_12_Items']) / 3
df['Q4_Support_Calls'] = (df['Month_10_Support_Calls'] + df['Month_11_Support_Calls'] + df['Month_12_Support_Calls']) / 3

#Average Spending Feature
df['Average_Spend'] = df['Purchase_ Amount'] / df['Number_of_Transactions']


#Categorical Mapping Function (Example) -- Modify for our categorical variables
def satisfaction(x):
    if x >= 8:
        return 'High'
    elif x <= 7 & x >= 5:
        return 'Medium'
    else:
        return 'Low'
    
df['Job Satisfaction Level '] = df['Job Satisfaction'].apply(satisfaction)





## Create Hypothesis statement before starting pipeline, "Based on my analysis, it is predicted that these features included"\
## have an impact on the probability that a customer will churn or not


## SELECT YOUR FEATURES AND YOUR OUTPUT FIRST (EXAMPLE)
X = final_data[['Customer Type', 'Age', 'Class', 'Leg room service', 'Checkin service',
        'Flight_Service', 'Online_Service', 'Baggage_Service',
       'Log_Flight_Distance']]
y = final_data['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


### OR THE METHOD BELOW ************************


#CODING NOTES TO BUILD PIPELINE

categorical = ['']
numerical = ['']
#ranking = ['']

target = ['']

# train and test set
X_train, X_test, y_train, y_test = train_test_split(df[categorical + numerical], df[target], test_size=0.3, random_state=123)

#Feature Transformers - Applying OneHotEncoding and MinMaxScaler
# prevent one hot encoder from returning a sparse matrix (GuassianNB will throw an error otherwise)
catTransformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
numTransformer = Pipeline(steps = [('scaler', MinMaxScaler())])

#Build Preprocessor
preprocessor = ColumnTransformer(transformers = [('cat', catTransformer, categorical), ('num', numTransformer, numerical)])




# Check if all specified columns are present in the DataFrame **
missing_columns = set(numerical + categorical) - set(X_train.columns)

if missing_columns:
    print(f"Missing columns in the DataFrame: {missing_columns}")
else:
    print("All columns are present.")

 # Check data types of the columns in the DataFrame
print(X_train.dtypes)

# Check for duplicate columns
duplicate_columns = X_train.columns[X_train.columns.duplicated()]
if not duplicate_columns.empty:
    print(f"Duplicate columns found: {duplicate_columns}")
else:
    print("No duplicate columns found.")

# Check for unique values in categorical columns to ensure encoding will work correctly
for column in categorical:
    unique_values = X_train[column].unique()
    print(f"Column: {column}, Unique Values: {unique_values}")




#MACHINE LEARNING MODELS TO IMPLEMENT
## If Predictions are too perfect for individual scores use (Example" LogisticRegression(class_weight='balanced') "


#Logistic Regression SETUP
logreg = Pipeline(steps = [('preprocessor', preprocessor), ('logistic', LogisticRegression())])

# fit model
logreg.fit(X_train, y_train)
# predictions
pred = logreg.predict(X_test)

# performance metrics (binary)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, pred, average='binary')
print("Precision:", precision)
recall = recall_score(y_test, pred, average='binary')
print("Recall Score:", recall)
f1 = f1_score(y_test, pred, average='binary')
print("F1 Score:", f1)

# cross validation
crossVal = cross_val_score(logreg, X_train, y_train, cv=10, scoring='f1')

# print scores
print("Cross Validation F1 Scores:", crossVal)
print("Mean cross validation F1 Score:", np.mean(crossVal))


#Naive Bayes Setup
nb = Pipeline(steps = [('preprocessor', preprocessor), ('nb', GaussianNB())])
# fit
nb.fit(X_train, y_train)
# predict
pred = nb.predict(X_test)


# performance metrics (binary)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, pred, average='binary')
print("Precision:", precision)
recall = recall_score(y_test, pred, average='binary')
print("Recall Score:", recall)
f1 = f1_score(y_test, pred, average='binary')
print("F1 Score:", f1)


# cross validation
crossVal = cross_val_score(nb, X_train, y_train, cv=10, scoring='f1')

# print scores
print("Cross Validation F1 Scores:", crossVal)
print("Mean cross validation F1 Score:", np.mean(crossVal))




#KNN Setup
knn = Pipeline(steps = [('preprocessor', preprocessor), ('knn', KNeighborsClassifier())])

# fit model
knn.fit(X_train, y_train)
# predictions
pred = knn.predict(X_test)
# performance metrics (binary)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, pred, average='binary')
print("Precision:", precision)
recall = recall_score(y_test, pred, average='binary')
print("Recall Score:", recall)
f1 = f1_score(y_test, pred, average='binary')
print("F1 Score:", f1)

# cross validation
crossVal = cross_val_score(knn, X_train, y_train, cv=10, scoring='f1')

# print scores
print("Cross Validation F1 Scores:", crossVal)
print("Mean cross validation F1 Score:", np.mean(crossVal))



#Random Forest Classifier
rf = Pipeline(steps= [('preprocessor', preprocessor), ('rf', RandomForestClassifier(n_estimators=100, random_state=100))])

rf.fit(X_train, y_train)
rf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, pred, average='binary')
print("Precision:", precision)
recall = recall_score(y_test, pred, average='binary')
print("Recall Score:", recall)
f1 = f1_score(y_test, pred, average='binary')
print("F1 Score:", f1)

# cross validation
crossVal = cross_val_score(rf, X_train, y_train, cv=10, scoring='f1')

# print scores
print("Cross Validation F1 Scores:", crossVal)
print("Mean cross validation F1 Score:", np.mean(crossVal))






# GRID SEARCH AND HYPER PARAMETER TUNING SETUP FOR FINAL MODEL

# Depending on the model

# -- RANDOM FOREST GRID SEARCH -- 
# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV to find the best hyperparameter combination
grid_search = GridSearchCV(rf, param_grid, cv=5, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)

# Output the best hyperparameter combination
print("Best hyperparameters for Random Forest:", grid_search.best_params_)


# -- KNN GRID SEARCH -- 
# Define hyperparameter grid
param_grid = {
    'classifier__n_neighbors': [5, 7, 9, 15, 35, 45, 55],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}

# Use GridSearchCV to find the best hyperparameter combination
grid_search = GridSearchCV(knn , param_grid, cv=5, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)

# Output the best hyperparameter combination
print("Best hyperparameters for KNN:", grid_search.best_params_)



# -- LOGISTIC REGRESSION -- 
# Define the hyperparameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__solver': ['saga']  # saga solver supports l1, l2, and elasticnet
}


# Use GridSearchCV to find the best hyperparameter combination
grid_search = GridSearchCV(logreg, param_grid, cv=5, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)


# Output the best hyperparameter combination
print("Best hyperparameters for Logistic Regression:", grid_search.best_params_)


# --- NAIVE BAYES -- 
# Define the hyperparameter grid
param_grid = {
    'classifier__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05]
}


# Use GridSearchCV to find the best hyperparameter combination
grid_search = GridSearchCV(nb_pipeline, param_grid, cv=5, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)

# Output the best hyperparameter combination
print("Best hyperparameters for Naive Bayes:", grid_search.best_params_)





## -- USE THE BEST PARAMS AND REBUILD THE PIPELINE (EXAMPLE) -- See if theres any difference

logreg = Pipeline(steps= [('preprocessor', preprocessor), ('logistic', LogisticRegression(C=0.1, class_weight='balanced', penalty='l1', solver='saga'))])

logreg.fit(X_train, y_train)

pred = logreg.predict(X_test)

# performance metrics (binary)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, pred)
print("Precision:", precision)
recall = recall_score(y_test, pred, average='binary')
print("Recall Score:", recall)
f1 = f1_score(y_test, pred, average='binary')
print("F1 Score:", f1)

# cross validation
crossVal = cross_val_score(logreg, X_train, y_train, cv=10, scoring='f1')

# print scores
print("Cross Validation F1 Scores:", crossVal)
print("Mean cross validation F1 Score:", np.mean(crossVal))









# FURTHER MODEL EVALUATION

# Calculate ROC curve and AUC score (Example)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title('Confusion Matrix')
plt.show()


#Classification Report

report = classification_report(y_test, pred)
print(report)