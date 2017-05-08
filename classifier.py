# -*- encoding: utf-8 -*-
import pickle
import glob
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
from features import *


class Classifier(object):
    """The classifier class.
    """

    def __init__(self):
        self.clf = None

    def fit(self, features, labels):
        svc = svm.SVC(kernel='linear', C=1.0)
        # parameters = {'C': [1, 10]}
        # self.clf = GridSearchCV(svc, parameters, verbose=2, n_jobs=2)
        self.clf = svc
        self.clf.fit(features, labels)
        # print("Best parameter(s): ", self.clf.best_params_)

    def predict(self, features):
        assert self.clf is not None, "Train the classifier first."
        return self.clf.predict(features)

    def score(self, features, labels):
        return self.clf.score(features, labels)

    def save(self, name):
        output_file = name + ".p"
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
            print("Classifier saved at {}".format(output_file))

    @staticmethod
    def restore(name):
        input_file = name + ".p"
        with open(input_file, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    vehicles = glob.glob("dataset/vehicles/*/*.png")
    non_vehicles = glob.glob("dataset/non-vehicles/*/*.png")
    all_features = []
    num_vehicles = 0
    num_non_vehicles = 0

    print("Reading dataset of vehicles")
    for v in tqdm(vehicles):
        image = cv2.imread(v)
        if image is None:
            continue
        features = get_feature_vector(image, color_space='HSV')
        all_features.append(features)
        num_vehicles += 1

    print("Reading dataset of non-vehicles")
    for v in tqdm(non_vehicles):
        image = cv2.imread(v)
        if image is None:
            continue
        features = get_feature_vector(image, color_space='HLS')
        all_features.append(features)
        num_non_vehicles += 1

    all_features = normalize_features(np.stack(all_features, axis=0))
    all_labels = np.concatenate((np.ones(num_vehicles), np.zeros(num_non_vehicles)))
    train_input, test_input, train_label, test_label = train_test_split(all_features, all_labels, test_size=0.2)

    print("Shape of training input: ", train_input.shape)
    print("Number of training input features: ", len(train_label))
    print("Training classifier")
    classifier = Classifier()
    classifier.fit(train_input, train_label)
    print("Evaluating classifier")
    print("Shape of testing input: ", test_input.shape)
    print("Number of training input features: ", len(test_label))
    print(classifier.score(test_input, test_label))
    classifier.save('model')
