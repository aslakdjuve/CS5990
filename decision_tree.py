# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: decision_tree.py
# SPECIFICATION: Read training and test data, build decision trees, and evaluate performance
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to convert income string to float (handling 'k' suffix)
def parse_income(income_str):
    if 'k' in income_str.lower():
        return float(income_str.lower().replace('k', '')) * 1000
    else:
        return float(income_str)

# Function to process data from a DataFrame
def process_data(df):
    X = []
    Y = []
    
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
        
        # Add class label to Y (Yes=1, No=2)
        Y.append(1 if row['Cheat'] == "Yes" else 2)
    
    return np.array(X), np.array(Y)

# Function to transform a single test instance (from DataFrame row)
def transform_instance(row):
    refund = 1 if row['Refund'] == "Yes" else 0
    
    marital_status = row['Marital Status']
    single = 1 if marital_status == "Single" else 0
    divorced = 1 if marital_status == "Divorced" else 0
    married = 1 if marital_status == "Married" else 0
    
    taxable_income = parse_income(row['Taxable Income'])
    
    return np.array([[refund, single, divorced, married, taxable_income]])

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:
    print(f"\nProcessing dataset: {ds}")
    
    # Read training data
    df_train = pd.read_csv(ds, sep=',', header=0)
    print(f"Training data columns: {df_train.columns.tolist()}")
    
    # Process training data
    X, Y = process_data(df_train)
    
    # Initialize variables to track model accuracy
    accuracy_sum = 0
    
    # Load test data once
    df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
    print(f"Test data columns: {df_test.columns.tolist()}")
    
    #loop your training and test tasks 10 times here
    for i in range(10):
        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree 
        #tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        #plt.show()
        
        # Track correct predictions for this run
        correct_predictions = 0
        total_instances = len(df_test)

        for index, row in df_test.iterrows():
            # Transform test instance
            test_instance = transform_instance(row)
            
            # Use the decision tree to make a prediction
            class_predicted = clf.predict(test_instance)[0]
            
            # Compare the prediction with the true label
            true_label = 1 if row['Cheat'] == "Yes" else 2
            if class_predicted == true_label:
                correct_predictions += 1
        
        # Calculate accuracy for this run
        run_accuracy = correct_predictions / total_instances
        accuracy_sum += run_accuracy

    # Find the average accuracy of this model during the 10 runs
    final_accuracy = accuracy_sum / 10
    
    # Print the accuracy of this model during the 10 runs
    print(f"final accuracy when training on {ds}: {final_accuracy:.1f}")