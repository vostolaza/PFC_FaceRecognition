import face_recognition
from email.mime import base
from utils import IMAGE_PATH
import numpy as np
import random as rd
import os
import time
import pickle
import utils
import json
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

N_SAMPLES = 10
PCA_N_COMPONENTS=25
N_NEIGHBORS=10

def get_image(path):
    picture = face_recognition.load_image_file(path)
    return face_recognition.face_encodings(picture)[0]


def generate_C1(set):
    return get_image(set[0]) - get_image(set[1])

def generate_C2(set):
    return get_image(set[0]) - get_image(set[2])

# def generate_C2(dir, set):
#     vectors = []
#     for image in set:
#         face = get_image(image)
#         with open("train_set.json") as f:
#             individuals = json.load(f)
#         count = 0
#         for train_dir, set_dir in individuals.items():
#             if count > N_SAMPLES: break
#             if train_dir == dir:
#                 continue
#             for img in set_dir:
#                 vectors.append(face - get_image(img))
#                 count += 1
#     return vectors


def generate_dataset(dir, set):
    print("Generating C1...")
    start = time.time()
    """ 
    C1: difference space for photos of the same subject
    """
    C1 = generate_C1(set)
    end = time.time()
    print("Generated C1 in {} seconds".format(end - start))

    print("Generating C2...")
    start = time.time()
    """
    C2: difference space for photos of the subject vs the other subjects
    """
    C2 = generate_C2(dir, set)
    end = time.time()
    print("Generated C2 in {} seconds".format(end - start))

    print("C1 Size:", np.array(C1).shape)
    print("C2 Size:", np.array(C2).shape)
    X = np.array(C1 + C2)
    X = X.reshape(len(X), -1)
    y = np.array([1] * len(C1) + [0] * len(C2))
    return X, y

def train_svm(dir, set, C=0.1, kernel='linear'):
    """
    path: directory with photos of a single subject
    """
    X, y = generate_dataset(dir, set)
    clf = svm.SVC(kernel=kernel, C=C)
    print("Fitting {} kernel...".format(kernel))
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print("Fitted in {} seconds".format(end - start))

    return clf

def train_knn_pca(dir, set, n_components=PCA_N_COMPONENTS,k=N_NEIGHBORS):
    X, y = generate_dataset(dir, set)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)
    # ya se redujo la dimensionalidad
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    return knn

def new_train_knn_pca(dir, set, n_components=PCA_N_COMPONENTS,k=N_NEIGHBORS):
    with open("train_set.json") as f:
        individuals = json.load(f)

    C1 = []
    C2 = []

    for dir, set in individuals.items():
        print("Generating C1 for {}...".format(dir))
        C1.append(generate_C1(set))
        print("Generating C2 for {}...".format(dir))
        C2.append(generate_C2(set))
    
    X = [*C1, *C2]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, [*[1 for individual in C1], *[0 for individual in C2]])
    return knn

def new_train_svm(kernel='linear', C=0.1):
    with open("train_set.json") as f:
        individuals = json.load(f)

    C1 = []
    C2 = []

    for dir, set in individuals.items():
        print("Generating C1 for {}...".format(dir))
        C1.append(generate_C1(set))
        print("Generating C2 for {}...".format(dir))
        C2.append(generate_C2(set))

    clf = svm.SVC(kernel=kernel, C=C)
    
    print(f"Fitting {kernel} kernel...")
    start = time.time()
    clf.fit([*C1, *C2], [*[1 for individual in C1], *[0 for individual in C2]])
    end = time.time()
    print("Fitted in {} seconds".format(end - start))
    
    return clf

def predict(pca_n_components=PCA_N_COMPONENTS):
    PV = 0
    PF = 0
    with open("test_set.json") as f:
        individuals = json.load(f)
    for dir, set in individuals.items():
        X = []
        y = []
        
        X += [get_image(set[0]) - get_image(set[1])]
        y += [1]

        for newDir, newSet in individuals.items():
            if newDir == dir:
                continue
            X += [get_image(set[0]) - get_image(newSet[0])]
            y += [0]
            break
        X = np.array(X)
        X = X.reshape(len(X), -1)
        y = np.array(y)
        print(dir, "PV", "PF", sep="\t")
        with open("svms/" + dir + ".pkl", "rb") as f:
            svm = pickle.load(f)
            y_pred = svm.predict(X)
            svm_pv, svm_pf = utils.score(y, y_pred)
            print("SVM:", svm_pv, svm_pf, sep="\t")
            PV += svm_pv
            PF += svm_pf
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        pca = PCA(n_components=pca_n_components)
        pca.fit(X)
        X = pca.transform(X)
        with open("knn_pcas/" + dir + ".pkl", "rb") as f:
            knn_pca = pickle.load(f)
            y_pred = knn_pca.predict(X)
            knn_pca_pv, knn_pca_pf = utils.score(y, y_pred)
            print("SVM:", knn_pca_pv, knn_pca_pf, sep="\t")

    print("MEAN", PV / len(y_pred), PF / len(y_pred))


def save_obj(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f)
    print("Saved Object to " + path + ".pkl")


if __name__ == "__main__":
    with open("train_set.json") as f:
        individuals = json.load(f)
    """
    Train SVMs and KNN - PCAs
    """
    # for dir, set in individuals.items():
    #     print("Training on {}".format(dir))
    #     single_svm = train_svm(dir, set)
    #     save_obj(single_svm, "svms/" + dir.split("/")[-1])
    #     del single_svm
    #     single_knn_pca = train_knn_pca(dir, set)
    #     save_obj(single_knn_pca, "knn_pcas/" + dir.split("/")[-1])
    #     del single_knn_pca
    # predict()
    clf = new_train_svm()
    print(clf.predict(get_image("images/00215/00215fb010_940128.tif") - get_image("images/00215/00215fa010_940128.tif")))
    print(clf.predict(get_image("images/00008/00008fb010_930831.tif") - get_image("images/00950/00950fb010_960627.tif")))
    print(clf.predict(get_image("images/00940/00940fa010_960627.tif") - get_image("images/00940/00940fb010_960627.tif")))
    print(clf.predict(get_image("images/00714/00714fb010_960530.tif") - get_image("images/00696/00696fb010_941121.tif")))
