from email.mime import base
import numpy as np
import random as rd
import os
import time
import pickle
import utils
import json
from imageio import imread
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

N_SAMPLES = 10
PCA_N_COMPONENTS=50


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

def train_svm(path, C=0.1, kernel='linear'):
    """
    path: directory with photos of a single subject
    """
    X, y = generate_dataset(path)
    clf = svm.SVC(kernel=kernel, C=C)
    print("Fitting {} kernel...".format(kernel))
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print("Fitted in {} seconds".format(end - start))

    return clf

def train_knn_pca(path, n_components=PCA_N_COMPONENTS):
    X, y = generate_dataset(path)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X)
    X = pca.transform(X)
    # ya se redujo la dimensionalidad
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X, y)
    return knn


def predict(pca_n_components=PCA_N_COMPONENTS):
    PV = 0
    PF = 0
    for dir in os.listdir("test_set/"):
        basePath = "colorferet/dvd2/gray_feret_cd"
        basePath += "1" if int(dir) <= 699 else "2"
        basePath += "/data/images/" + dir + "/"
        X = []
        y = []
        for file in os.listdir("test_set/" + dir):
            vectors = []
            ti = get_image("test_set/" + dir + "/" + file)
            for file2 in os.listdir(basePath):
                tj = get_image(basePath + "/" + file2)
                vectors.append(ti - tj)
            X += vectors
            if dir in file:
                y += [1] * len(vectors)
            else:
                y += [0] * len(vectors)
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
    basePath = "colorferet/dvd2/"
    # individuals = utils.get_train_test_subjects(basePath)
    with open("train_subjects.json") as f:
        individuals = json.load(f)
    """
    Train SVMs and KNN - PCAs
    """
    for dir in individuals:
        print("Training on {}".format(dir))
        #single_svm = train_svm(dir)
        #save_obj(single_svm, "svms/" + dir.split("/")[-1])
        #del single_svm
        single_knn_pca = train_knn_pca(dir)
        save_obj(single_knn_pca, "knn_pcas/" + dir.split("/")[-1])
        del single_knn_pca
