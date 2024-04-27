import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_processing import translate_landmarks, rotate_landmarks

# # Read data from .bnd file
# filepath = "FacialLandMarks/F001/Angry/000.bnd"
# data_type = "Rotated"   # Original, Translated, Rotated
# direction = "Y"         # X, Y, Z

# Read data from .bnd file
filepath = "FacialLandMarks/F001/Angry/000.bnd"
data_type = sys.argv[1]   # Original, Translated, Rotated
direction = sys.argv[2]         # X, Y, Z


with open(filepath, 'r') as f:
    landmarks = []
    # Extract the lines from the file
    bnd_data = f.readlines()
    # Extract the coordinates in string i.e x,y,z
    for line in bnd_data[:84]:
        x, y, z = line.split()[1:]
        landmarks.append([float(x), float(y), float(z)])


    if data_type == "Original":
        landmarks = np.array(landmarks)
    elif data_type == "Translated":
        landmarks = translate_landmarks(landmarks)
    elif data_type == "Rotated":
        landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotate_landmarks(landmarks)
        landmarks = landmarksRotatedX if direction == "X" else landmarksRotatedY if direction == "Y" else landmarksRotatedZ

    # Ensure landmarks is a 2D numpy array
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Print the shape of landmarks to diagnose the issue
    print("Shape of landmarks array:", landmarks.shape)

    # Plot landmarks
    # ax.scatter(landmarks[:, 1], landmarks[:, 2], landmarks[:, 3], c='brown', marker='o')
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='green', marker='o')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend([(data_type+(direction if data_type == "Rotated" else ""))])

    plt.show()