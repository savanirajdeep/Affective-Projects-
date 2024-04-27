import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import os


def print_evaluation_metrics(predictions, indices, true_labels, classifier, data_type, fold_index):
    # Collect predictions and ground truth labels from all folds
    final_predictions = []
    ground_truth = []

    for pred in predictions:
        final_predictions.extend(pred)
    for i in indices:
        ground_truth.extend(true_labels[i])

    # Calculate evaluation metrics
    cm = confusion_matrix(final_predictions, ground_truth)
    precision = precision_score(ground_truth, final_predictions, average='macro')
    recall = recall_score(ground_truth, final_predictions, average='macro')
    accuracy = accuracy_score(ground_truth, final_predictions)

    # Create directories for storing outputs
    new_dir = 'Final_Outputs/' + classifier
    os.makedirs(new_dir, exist_ok=True)
    filename = f"{classifier}_{data_type}.txt"
    
    # Write evaluation metrics to a file
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write(f'Fold {fold_index}\n')
            file.write("\tConfusion matrix:\n" + str(cm) + '\n')
            file.write("\tPrecision: " + str(precision) + '\n')
            file.write("\tRecall: " + str(recall) + '\n')
            file.write("\tAccuracy: " + str(accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)

    return cm, precision, recall, accuracy


def subject_independent_cross_validation(X, y, clf, classifier, data_type, subjects):
    # Perform subject-independent cross-validation
    n_folds = 10
    groups = subjects
    gkf = GroupKFold(n_splits=n_folds)
    gkf.get_n_splits(X, y, groups)

    confusion_matrix_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for fold_index in range(n_folds):
        for fold_num, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
            if fold_num == fold_index:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                cm, precision, recall, accuracy = print_evaluation_metrics(
                    [pred], [test_index], y, classifier, data_type, fold_index + 1)

                confusion_matrix_scores.append(cm)
                precision_scores.append(precision)
                recall_scores.append(recall)
                accuracy_scores.append(accuracy)

                new_groups = set(groups[test_index]) - set(groups[train_index])
                new_index = [i for i, subj in enumerate(groups) if subj in new_groups]
                X_train = np.concatenate((X_train, X[new_index]))
                y_train = np.concatenate((y_train, y[new_index]))

    return confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores


def evaluate(X, y, classifier, data_type, subjects):
    # Evaluate the classifier using subject-independent cross-validation
    clf = None
    if classifier == "SVM":
        clf = svm.LinearSVC(dual=False)
    elif classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "TREE":
        clf = DecisionTreeClassifier()

    confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = subject_independent_cross_validation(
        X, y, clf, classifier, data_type, subjects)
    
    # Calculate average evaluation metrics across folds     
    avg_cm = np.mean(confusion_matrix_scores, axis=0)
    avg_precision = np.mean(precision_scores, axis=0)
    avg_recall = np.mean(recall_scores, axis=0)
    avg_accuracy = np.mean(accuracy_scores, axis=0)

    new_dir = 'Final_Outputs/' + classifier
    os.makedirs(new_dir, exist_ok=True)
    filename = f"{classifier}_{data_type}.txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file:
            file.write('Average Scores\n')
            file.write("\tConfusion matrix:\n" + str(avg_cm) + '\n')
            file.write("\tPrecision: " + str(avg_precision) + '\n')
            file.write("\tRecall: " + str(avg_recall) + '\n')
            file.write("\tAccuracy: " + str(avg_accuracy) + '\n')
    except Exception as e:
        print("Error while writing to file:", e)
