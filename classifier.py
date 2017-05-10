# -*- encoding: utf-8 -*-
import pickle
import glob
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import features as f
import numpy as np
import utils


class Classifier(object):
    """The classifier class.
    """

    def __init__(self, scaler):
        self.clf = None
        self.scaler = scaler

    def fit(self, features, labels):
        svc = svm.LinearSVC(C=1.0)
        # parameters = {'C': [1, 10]}
        # self.clf = GridSearchCV(svc, parameters, verbose=2, n_jobs=2)
        self.clf = svc
        features = self.scaler.transform(features)
        self.clf.fit(features, labels)
        # print("Best parameter(s): ", self.clf.best_params_)

    def predict(self, features):
        assert self.clf is not None, "Train the classifier first."
        single_sample = False
        if len(features.shape) == 1:
            single_sample = True
            features = features.reshape(1, -1)
        features = self.scaler.transform(features)
        if single_sample:
            return self.clf.predict(features).astype(np.int)[0]
        else:
            return self.clf.predict(features).astype(np.int)

    def score(self, features, labels):
        assert self.clf is not None, "Train the classifier first."
        features = self.scaler.transform(features)
        return self.clf.score(features, labels)

    def save(self, name):
        output_file = name + ".p"
        with open(output_file, 'wb') as f:
            pickle.dump([self.clf, self.scaler], f)
            print("Classifier saved at {}".format(output_file))

    @staticmethod
    def restore(name):
        input_file = name + ".p"
        with open(input_file, 'rb') as f:
            clf, scaler = pickle.load(f)
        classifier = Classifier(scaler)
        classifier.clf = clf
        return classifier


def read_single_sample(path, color_space='YCrCb'):
    image = cv2.imread(path)
    if image is None:
        return None
    image = utils.convert_color_space(image, color_space=color_space)
    feature_vector = f.get_feature_vector(image, hog_pixels_per_cell=8, hog_cells_per_block=2, hog_channel=-1)
    return feature_vector


def read_samples(color_space='YCrCb'):
    vehicles = glob.glob("dataset/vehicles/*/*.png")
    non_vehicles = glob.glob("dataset/non-vehicles/*/*.png")
    all_features = []
    num_vehicles = 0
    num_non_vehicles = 0

    print("Reading dataset of vehicles")
    for v in tqdm(vehicles):
        feature_vector = read_single_sample(v, color_space=color_space)
        if feature_vector is not None:
            all_features.append(feature_vector)
            num_vehicles += 1

    print("Reading dataset of non-vehicles")
    for v in tqdm(non_vehicles):
        feature_vector = read_single_sample(v, color_space=color_space)
        if feature_vector is not None:
            all_features.append(feature_vector)
            num_non_vehicles += 1

    all_features = np.stack(all_features, axis=0)
    all_labels = np.concatenate((np.ones(num_vehicles), np.zeros(num_non_vehicles)))

    return all_features, all_labels


def train_classifier(features, labels, name='model'):
    train_input, test_input, train_label, test_label = train_test_split(features, labels, test_size=0.2)

    features_scaler = f.get_feature_normalizer(features.astype(np.float64))

    print("Shape of training input: ", train_input.shape)
    print("Number of training input features: ", len(train_label))
    print("Training classifier")
    classifier = Classifier(features_scaler)
    classifier.fit(train_input, train_label)

    print("Evaluating classifier")
    print("Shape of testing input: ", test_input.shape)
    print("Number of training input features: ", len(test_label))
    print(test_input.shape)
    print("Accuracy: ", classifier.score(test_input, test_label))
    classifier.save(name)
    return classifier


if __name__ == '__main__':
    color_space = 'YCrCb'
    sample_features, sample_labels = read_samples(color_space=color_space)
    train_classifier(sample_features, sample_labels)
    print("Restore classifier and scoring against the whole data set.")
    clf = Classifier.restore('model')
    fv = read_single_sample("./test2.png", color_space=color_space)
    print(clf.predict(fv))

    # print("Accuracy:", clf.score(features, labels))
