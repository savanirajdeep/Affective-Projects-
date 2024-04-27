import zipfile
import os
import numpy as np
from math import pi, acos


def get_class_label(label):
    # Mapping textual labels to numerical values
    class_labels = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5
    }
    return class_labels.get(label)


def translate_landmarks(landmarks):
    # Translate landmarks to have their centroid at the origin
    landmarks = np.array(landmarks)

    mean_x = np.mean(landmarks[:, 0])
    mean_y = np.mean(landmarks[:, 1])
    mean_z = np.mean(landmarks[:, 2])

    landmarks[:, 0] -= mean_x
    landmarks[:, 1] -= mean_y
    landmarks[:, 2] -= mean_z

    return landmarks


def rotate_landmarks(landmarks):
    # Rotate landmarks around each axis by 180 degrees
    landmarks = np.array(landmarks)

    cos = np.cos(pi)
    sine = np.sin(pi)

    x_axis = np.array([[1, 0, 0], [0, cos, sine], [0, -sine, cos]])
    y_axis = np.array([[cos, 0, -sine], [0, 1, 0], [sine, 0, cos]])
    z_axis = np.array([[cos, sine, 0], [-sine, cos, 0], [0, 0, 1]])

    rotated_x = x_axis.dot(landmarks.T).T
    rotated_y = y_axis.dot(landmarks.T).T
    rotated_z = z_axis.dot(landmarks.T).T

    return rotated_x, rotated_y, rotated_z


def read_data(path, data_type):
    features = []
    classes = []
    subjects = []

    if path.endswith('.zip'):
        # If the path is a zip file
        with zipfile.ZipFile(path) as myzip:
            for subject_dir in sorted(myzip.namelist()):
                for expression_dir in sorted(myzip.namelist()):
                    for bnd_file in sorted(myzip.namelist()):
                        if bnd_file.endswith('.bnd'):
                            # Extracting information from the file
                            label = os.path.basename(os.path.dirname(bnd_file))
                            label_val = get_class_label(label)
                            subject = os.path.basename(
                                os.path.dirname(os.path.dirname(bnd_file)))
                            bnd_data = myzip.read(bnd_file)
                            bnd_str = bnd_data.decode('utf-8')
                            landmarks = []
                            
                            # Parsing the landmark data
                            for line in bnd_str.split("\n"):
                                if len(line) > 0:
                                    x, y, z = line.split()[1:]
                                    landmarks.append(
                                        [float(x), float(y), float(z)])

                            # Preprocess data based on data type
                            if data_type == "Original":
                                landmarks = np.array(landmarks).flatten()
                            elif data_type == "Translated":
                                landmarks = translate_landmarks(landmarks)
                            elif data_type == "Rotated":
                                landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotate_landmarks(
                                    landmarks)
                                landmarks = [landmarksRotatedX,
                                             landmarksRotatedY, landmarksRotatedZ]

                            # Append data to lists
                            features.append(landmarks)
                            classes.append(label_val)
                            subjects.append(subject)
    else:
        # If the path is a directory
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.bnd'):
                    # Extracting information from the file
                    label = subdir.split(os.path.sep)[-1]
                    label_val = get_class_label(label)
                    subject = os.path.join(subdir, file)
                    filepath = os.path.join(subdir, file)
                    with open(filepath, 'r') as f:
                        landmarks = []
                        bnd_data = f.readlines()
                        # Parsing the landmark data
                        for line in bnd_data[:84]:
                            x, y, z = line.split()[1:]
                            landmarks.append([float(x), float(y), float(z)])

                        # Preprocess data based on data type
                        if data_type == "Original":
                            landmarks = np.array(landmarks).flatten()
                        elif data_type == "Translated":
                            landmarks = translate_landmarks(landmarks)
                        elif data_type == "Rotated":
                            landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotate_landmarks(
                                landmarks)
                            landmarks = [landmarksRotatedX,
                                         landmarksRotatedY, landmarksRotatedZ]

                    # Append data to lists
                    features.append(landmarks)
                    classes.append(label_val)
                    subjects.append(subject)
    return features, classes, subjects
