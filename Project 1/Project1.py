import sys
import numpy as np
from classification import evaluate
from data_processing import read_data
import random
import matplotlib.pyplot as plt


def run_experiment(classifier, data_type, data_directory):
    print("------------------------------------------------------")
    print(f"Classifier: {classifier}, Data Type: {data_type}")

    # Read and transform the dataset
    X, y, subjects = read_data(data_directory, data_type)
    X = np.array(X) # Converting features to numpy array
    y = np.array(y)  # Converting labels to numpy array
    subjects = np.array(subjects)  # Converting subjects to numpy array

    if data_type in ["Original", "Translated"]:  # Checking if data type is either Original or Translated
        evaluate(X, y, classifier, data_type, subjects)  # Calling evaluate function from classification module
    elif data_type == "Rotated":  # Checking if data type is Rotated
        evaluate_rotated(X, y, classifier, subjects)  # Calling evaluate_rotated function


    print("------------------------------------------------------")


def evaluate_rotated(X, y, classifier, subjects):
    for axis in ['X', 'Y', 'Z']: # Looping through each axis
        X_axis = X[:, 0] if axis == 'X' else (X[:, 1] if axis == 'Y' else X[:, 2]) # Selecting data based on axis
        evaluate(X_axis, y, classifier, f"Rotated{axis}", subjects) # Calling evaluate function for rotated data


def plot_sample_points(X, label, color):
    # Take a random sample from dataset to plot the face
    sample_index = random.randint(0, len(X))
    # Reshaping the data for plotting
    X_sample = X[sample_index].reshape(83, 3)

    fig = plt.figure()  # Creating a new figure
    ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot
    ax.scatter(X_sample[:, 0], X_sample[:, 1], X_sample[:, 2], c=color, label=label)  # Scatter plotting the sample data

    # Add labels for each axis
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Add a legend to the plot
    ax.legend()

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Command Line Arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py [Classifier] [Data_Type] [Data_Directory]")
        sys.exit(1)

    classifier = sys.argv[1]# Getting classifier from command line argument
    data_type = sys.argv[2]  # Getting data type from command line argument
    data_directory = sys.argv[3]  # Getting data directory from command line argument   

    # Check if classifier argument is valid
    if classifier not in ["SVM", "RF", "TREE"]:
        print("Invalid classifier")
        sys.exit(1)

    # Check if data type argument is valid
    if data_type not in ["Original", "Translated", "Rotated"]:
        print("Invalid Data Type")
        sys.exit(1)

    # Run experiment
    run_experiment(classifier, data_type, data_directory)
