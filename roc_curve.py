# -------------------------------------------------------------------------
# AUTHOR: Aslak Djuve
# FILENAME: roc_curve.py
# SPECIFICATION: Read data, split into train/test sets, build decision tree, and plot ROC curve
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 4 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# Function to convert income string to float (handling 'k' suffix)
def parse_income(income_str):
    if 'k' in income_str.lower():
        return float(income_str.lower().replace('k', '')) * 1000
    else:
        return float(income_str)

# Read the dataset cheat_data.csv
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
print("Columns in the dataset:", df.columns.tolist())

# Transform the original training features to numbers and labels
X = []
y = []

for i in range(len(df)):
    # Get row data directly from DataFrame
    row = df.iloc[i]
    
    # Refund feature (Yes=1, No=0)
    refund = 1 if row['Refund'] == "Yes" else 0
    
    # Marital Status (one-hot encoding)
    marital_status = row['Marital Status']
    single = 1 if marital_status == "Single" else 0
    divorced = 1 if marital_status == "Divorced" else 0
    married = 1 if marital_status == "Married" else 0
    
    # Taxable Income (convert to float)
    taxable_income = parse_income(row['Taxable Income'])
    
    # Add features to X
    X.append([refund, single, divorced, married, taxable_income])
    
    # Add class label to y (Yes=1, No=0)
    y.append(1 if row['Cheat'] == "Yes" else 0)

X = np.array(X)
y = np.array(y)

# Split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

# Generate a no skill prediction (random classifier - scores should be all constant like zero)
ns_probs = [0 for _ in range(len(testy))]

# Fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# Predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# Keep probabilities for the positive outcome only
dt_probs = dt_probs[:, 1]

# Calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# Summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# Plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# Axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# Show the legend
pyplot.legend()

# Show the plot
pyplot.show()
