import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import os
import matplotlib.pyplot as plt


# Read data from CSV file
def read_csv_data(path):
    csv_data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            subject_id = row[0]
            dtype = row[1]
            cls = row[2]
            values = [float(x) for x in row[3:]]
            csv_data.append([subject_id, dtype, cls, values])
    return csv_data


# Extract the features, i.e, mean, variance, min, max
def extract_features(csv_data, data_type):
    data_types_array = ["dia", "sys", "eda", "res"]
    data_array = {"dia": [], "sys": [], "eda": [], "res": []}
    for type in data_types_array:
        data_array[type] = [x for x in csv_data if type in x[1].lower()]
    features = {"dia": [], "sys": [], "eda": [], "res": [], "all": []}
    labels = {"dia": [], "sys": [], "eda": [], "res": [], "all": []}
    subjects = {"dia": [], "sys": [], "eda": [], "res": [], "all": []}

    # Compute features
    for type in data_array:
        data = data_array[type]
        for subject_id, dtype, cls, values in data:
            mean = np.mean(values)
            var = np.var(values)
            min_val = np.min(values)
            max_val = np.max(values)
            feature = [mean, var, min_val, max_val]
            features[type].append(feature)
            labels[type].append(cls)
            subjects[type].append(subject_id)
    if data_type == "all":
        length = len(features["dia"])
        for i in range(length):
            row = features["dia"][i] + features["sys"][i] + \
                features["eda"][i] + features["res"][i]
            features["all"].append(row)
        labels["all"] = labels["dia"]
        subjects["all"] = subjects["dia"]
    
    return features, labels[data_type], subjects[data_type]


def PrintEvalMetrics(pred, indices, y,  data_type, fold_index):
    # manually merge predictions and testing labels from each of the folds to make confusion matrix
    finalPredictions = []
    groundTruth = []

    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    cm = confusion_matrix(finalPredictions, groundTruth)
    precision = precision_score(
        groundTruth, finalPredictions, average='macro')
    recall = recall_score(groundTruth, finalPredictions, average='macro')
    accuracy = accuracy_score(groundTruth, finalPredictions)
    
 
    new_dir = 'Result_Folds/'
    os.makedirs(new_dir, exist_ok=True)
    filename = data_type+".txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Fold ' + str(fold_index)+'\n')
            file.write("\tConfusion matrix: "+ '\n' + ' ' + str(cm) + '\n')
            file.write("\tPrecision: " + str(precision) + '\n')
            file.write("\tRecall: " + str(recall) + '\n')
            file.write("\tAccuracy: " + str(accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)
    return cm, precision, recall, accuracy


def subject_independent_cross_validation(X, y, clf, data_type, subjects):
    # confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = subject_independent_cross_validation(
        # X, y, clf, data_type, subjects)
    # set the number of folds to 10
    n_folds = 10

    # get the groups for the cross-validation (in this case, the subjects)
    groups = subjects

    # create a GroupKFold cross-validator with the specified number of folds
    gkf = GroupKFold(n_splits=n_folds)
    gkf.get_n_splits(X, y, groups)

    # initialize lists to store the evaluation metrics for each fold
    confusion_matrix_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    # iterate over each fold
    for i in range(10):
        # iterate over each train-test split in the current fold
        for fold_index, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
            # if the current train-test split belongs to the current fold
            if i == fold_index:
                # split the data into training and testing sets based on the current train-test split
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # train the classifier on the training data and evaluate it on the testing data
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                cm, precision, recall, accuracy = PrintEvalMetrics(
                    [pred], [test_index], y, data_type, fold_index + 1)

                # append the evaluation metrics for the current fold to the respective lists
                confusion_matrix_scores.append(cm)
                precision_scores.append(precision)
                recall_scores.append(recall)
                accuracy_scores.append(accuracy)

    # return the lists of evaluation metrics for all folds
    return confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores


def evaluate(X, y, data_type, subjects):
#  evaluate(features, labels, data_type, subjects)
    X = X[data_type]

    X = np.array(X)

    y = np.array(y)

    subjects = np.array(subjects)

    clf = RandomForestClassifier()

    confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = subject_independent_cross_validation(
        X, y, clf, data_type, subjects)

    # Compute and return the average score across all folds
    avg_cm = np.mean((confusion_matrix_scores), axis=0)
    avg_precision = np.mean((precision_scores), axis=0)
    avg_recall = np.mean((recall_scores), axis=0)
    avg_accuracy = np.mean((accuracy_scores), axis=0)

    print("\n------ " + data_type.upper() + " ------\n")
    print("Confusion matrix: "+'\n' + ' ' +str(avg_cm) + '\n')
    print("Precision: ", avg_precision)
    print("Recall: ", avg_recall)
    print("Accuracy: ", avg_accuracy)
    


    # Save the average score across all folds in a file
    new_dir = 'Result_Avg/'
    os.makedirs(new_dir, exist_ok=True)
    filename = data_type+".txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Average Scores \n')
            file.write("\tConfusion matrix: "+'\n' + ' ' +str(avg_cm) + '\n')
            file.write("\tPrecision: " + str(avg_precision) + '\n')
            file.write("\tRecall: " + str(avg_recall) + '\n')
            file.write("\tAccuracy: " + str(avg_accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)

# Code for boxplot

def box_plot(features):

    mean = []
    variance = []
    min = []
    max = []    

    dia = features["dia"]
    sys = features["sys"]
    eda = features["eda"]
    res = features["res"]

    

    data = np.concatenate((dia, sys, eda, res))

    mean = data[:, 0]
    variance = data[:, 1]
    min = data[:, 2]
    max = data[:, 3]

    # Combine data into a single list
    all_data = [mean, variance, min, max]

    fig, ax = plt.subplots()
    ax.boxplot(all_data)

    # Set the tick labels
    ax.set_xticklabels(['Mean', 'Variance', 'Min', 'Max'])

    # Show the plot
    plt.show()



# Code for plotting the physiological signals in one line graph
def PhysiologicalSignalsPlot():
    # read the data from the CSV file
    data = read_csv_data('Project2Data.csv')
    # data = data_file


    print("\n----- GRAPH IS PRINTING -----\n")
    # get the data for the signals in rows 117 to 120
    # get the data for F015 Pain
    signal1 = data[117] 
    signal2 = data[118]
    signal3 = data[119]
    signal4 = data[120]


    # get the values for the first signal
    values1 = signal1[3]
    values2 = signal2[3]
    values3 = signal3[3]
    values4 = signal4[3]

    # get the classifier for the first signal
    classifier1 = signal1[1]
    classifier2 = signal2[1]
    classifier3 = signal3[1]
    classifier4 = signal4[1]

    # plot the signals
    plt.figure(2) 
    plt.plot(values1, label=classifier1) # plot the first signal
    plt.plot(values2, label=classifier2) # plot the second signal
    plt.plot(values3, label=classifier3) # plot the third signal
    plt.plot(values4, label=classifier4) # plot the fourth signal
    plt.legend()
    plt.title('Physiological Signals') # set the title of the plot
    plt.xlabel('Time') # set the x-axis label
    plt.ylabel('Value') # set the y-axis label
    
    new_dir = 'Images/'
    os.makedirs(new_dir, exist_ok=True)
    plt.savefig(new_dir)
    plt.show() # show the plot
    

if __name__ == "__main__":

    # Command Line Arguments
    data_type = sys.argv[1]
    data_file = sys.argv[2]

    data = read_csv_data(data_file)

    features, labels, subjects = extract_features(data, data_type)

    evaluate(features, labels, data_type, subjects)
    box_plot(features)
    PhysiologicalSignalsPlot()