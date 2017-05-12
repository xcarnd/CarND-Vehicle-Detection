# -*- encoding: utf-8 -*-
import pickle
import glob
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import features as f
import numpy as np
import utils
import pathlib
import settings as s
import matplotlib.pyplot as plt


def extract_feature_vector(image):
    """Extract feature vector from the given image.
    """
    ctrans_image = utils.convert_color_space(image, color_space=s.color_space)
    return f.get_feature_vector(ctrans_image)


def prepare_features_and_labels(vehicles, non_vehicles):
    """Prepare features and labels from the given vehicle samples and non-vehicle samples.
    """
    num_vehicles = len(vehicles)
    num_non_vehicles = len(non_vehicles)
    features_vehicles = [extract_feature_vector(smpl) for smpl in vehicles]
    features_non_vehicles = [extract_feature_vector(smpl) for smpl in non_vehicles]
    all_features = np.concatenate((features_vehicles, features_non_vehicles), axis=0)
    all_labels = np.concatenate((np.ones(num_vehicles), np.zeros(num_non_vehicles)))
    return all_features, all_labels


def train_classifier(features, labels):
    """Train a LinearSVC with the provided features and labels.
    """
    train_input, test_input, train_label, test_label = train_test_split(features, labels, test_size=0.2)

    print("Shape of training input: ", train_input.shape)
    print("Number of training input features: ", len(train_label))

    svc = svm.LinearSVC()
    svc.fit(train_input, train_label)
    print("Training accuracy: ", svc.score(train_input, train_label))

    print("Evaluating classifier")
    print("Shape of testing input: ", test_input.shape)
    print("Number of training input features: ", len(test_label))
    print(test_input.shape)
    print("Accuracy: ", svc.score(test_input, test_label))

    return svc


def create_pickled_dataset():
    """Read in all the samples and store it in a pickle file for future use.
    """
    file_check = pathlib.Path("./dataset_bgr.p")
    if file_check.exists():
        return
    vehicles = glob.glob("dataset/vehicles/*/*.png")
    non_vehicles = glob.glob("dataset/non-vehicles/*/*.png")
    samples_vehicle = []
    samples_non_vehicles = []

    print("Reading dataset of vehicles")
    for v in tqdm(vehicles):
        image = cv2.imread(v)
        if image is not None:
            samples_vehicle.append(image)

    print("Reading dataset of non-vehicles")
    for v in tqdm(non_vehicles):
        image = cv2.imread(v)
        if image is not None:
            samples_non_vehicles.append(image)

    with open('dataset_bgr.p', 'wb') as f:
        pickle.dump([samples_vehicle, samples_non_vehicles], f)


def load_samples():
    """Load pickled samples.
    """
    create_pickled_dataset()
    with open('dataset_bgr.p', 'rb') as f:
        vehicles, non_vehicles = pickle.load(f)
    return vehicles, non_vehicles


if __name__ == '__main__':
    # load samples
    samples_vehicles, samples_non_vehicles = load_samples()
    print("Number of vehicle samples: ", len(samples_vehicles))
    print("Number of non-vehicle samples: ", len(samples_non_vehicles))

    print("Preparing feature and label from samples")
    features, labels = prepare_features_and_labels(samples_vehicles, samples_non_vehicles)

    print("Normalizing features")
    scaler = StandardScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)

    print(scaled_features[0])

    print("Training classifier")
    clf = train_classifier(scaled_features, labels)

    # Test code
    # test_img = cv2.imread('./test5.png')
    # feat_v = extract_feature_vector(test_img)
    # prediction = clf.predict(scaler.transform(feat_v.reshape(1, -1)))
    # print("Prediction: ", prediction)

    # pickle trained classifier and scaler for future use
    with open('clf.p', 'wb') as clf_data_file:
        pickle.dump({"classifier": clf,
                     "scaler": scaler}, clf_data_file)
        clf_data_file.close()

