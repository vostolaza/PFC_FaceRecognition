import numpy as np
import random as rd
import pandas as pd
import os
import time
import pickle
import utils
from imageio import imread
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

N_SAMPLES = 10


def get_image(path):
    return np.array(imread(path))


def generate_C1(path):
    vectors = []
    for file in os.listdir(path):
        if file.endswith(".tif"):
            ti = get_image(path + "/" + file)
            for file2 in os.listdir(path):
                if file2 != file and file2.endswith(".tif"):
                    tj = get_image(path + "/" + file2)
                    vectors.append(ti - tj)
    return vectors


def chooseDirectory(vis, subject_path):
    subject_id = int(subject_path.split("/")[-1])

    base_path = "/".join(subject_path.split("/")[:-4])
    idx = None
    while True:
        idx = rd.randint(1, 1209)
        if idx in vis or idx == subject_id:
            continue
        p = base_path + (
            "/gray_feret_cd1/data/images/"
            if idx < 700
            else "/gray_feret_cd2/data/images/"
        )
        p += str(idx).zfill(5)

        if not os.path.exists(p):
            continue
        break

    vis.add(idx)
    dir = base_path + (
        "/gray_feret_cd1/data/images/" if idx < 700 else "/gray_feret_cd2/data/images/"
    )
    dir += str(idx).zfill(5)
    return dir


def C2_helper(subject_path, ti):
    visited = set()
    vectors = []
    for i in range(N_SAMPLES):
        dir = chooseDirectory(visited, subject_path)
        person_imgs = os.listdir(dir)
        for img_path in person_imgs:
            if img_path.endswith(".tif"):
                tj = get_image(dir + "/" + img_path)
                vectors.append(ti - tj)
    return vectors


def generate_C2(subject_path):
    vectors = []
    for file in os.listdir(subject_path):
        if file.endswith(".tif"):
            ti = get_image(subject_path + "/" + file)
            vectors += C2_helper(subject_path, ti)
    return vectors


def generate_dataset(path):
    print("Generating C1...")
    start = time.time()
    """ 
    C1: difference space for photos of the same subject
    """
    C1 = generate_C1(path)
    end = time.time()
    print("Generated C1 in {} seconds".format(end - start))

    print("Generating C2...")
    start = time.time()
    """
    C2: difference space for photos of the subject vs the other subjects
    """
    C2 = generate_C2(path)
    end = time.time()
    print("Generated C2 in {} seconds".format(end - start))

    print("C1 Size:", np.array(C1).shape)
    print("C2 Size:", np.array(C2).shape)
    X = np.array(C1 + C2)
    X = X.reshape(len(X), -1)
    y = np.array([1] * len(C1) + [0] * len(C2))
    return X, y


def train_svm(path):
    """
    path: directory with photos of a single subject
    """
    X, y = generate_dataset(path)
    clf = svm.SVC(kernel="linear", C=1)
    print("Fitting...")
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print("Fitted in {} seconds".format(end - start))

    return clf


def train_knn_pca(path):
    X, y = generate_dataset(path)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(0.95)  # mantener 95% de varianza en el dataset, pca elige el n_components
    pca.fit(X)
    X = pca.transform(X)
    # ya se redujo la dimensionalidad
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn


def predict(ID, photo):
    with open("svms/" + ID + ".pkl", "rb") as f:
        svm = pickle.load(f)
    photo = get_image(photo)
    photo = photo.reshape(1, -1)
    y_pred = svm.predict(photo)

    return y_pred


def save_obj(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f)
    print("Saved Object to " + path + ".pkl")


if __name__ == "__main__":
    basePath = "colorferet/dvd2/"
    individuals = utils.get_train_test_subjects(basePath)
    """
    Train SVMs and KNN - PCAs
    """
    for cd in [1, 2]:
        for dir in individuals:
            print("Training on {}".format(dir))
            single_svm = train_svm(dir)
            save_obj(single_svm, "svms/" + dir.split("/")[-1])
            del single_svm
            single_knn_pca = train_knn_pca(dir)
            save_obj(single_knn_pca, "knn_pcas/" + dir.split("/")[-1])
            del single_knn_pca
