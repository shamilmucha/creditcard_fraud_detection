import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

# Load dataset

data = pd.read_csv(r'/home/UID/workspace/ML Projects/Credit card fraud detection/creditcard.csv')
print(data.head())
print(data['Class'].value_counts())

# check for missing values

print(data.isnull().sum())

# Data preprocessing (feature scaling)

scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

print(data.head())

# Split dataset into features and target variable

X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# handle class imbalance

# option 1: class weighting

model = LogisticRegression(class_weight='balanced', max_iter=1000)

'''
# option 2: random undersampling/ SMOTE (optional)

sm = SMOTE(random_state=42)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

'''

# Training the model

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Model evaluation

print(classification_report(Y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(Y_test, y_prob))

# Confusion Matrix visualization

cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve visualization

fpr, tpr, _ = roc_curve(Y_test, y_prob)
plt.plot(fpr, tpr, label= "AUC = {:,.3f}".format(roc_auc_score(Y_test, y_prob)))
plt.plot([0,1],[0,1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
