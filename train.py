import numpy as np
import random as rd
import os
import shutil
import time
import pickle

from imageio import imread
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    subject_id = int(subject_path.split('/')[-1])

    base_path = '/'.join(subject_path.split('/')[:-4])

    while (idx := rd.randint(1, 1209)) in vis and idx != subject_id:
        idx = rd.randint(1, 1209)
    vis.add(idx)
    dir = base_path + ("/gray_feret_cd1/data/images/" if idx < 700 else "/gray_feret_cd2/data/images/")
    dir += str(idx).zfill(5)
    return dir

def C2_helper(subject_path, ti):
    visited = set()
    vectors = []
    for i in range(N_SAMPLES):
        dir = chooseDirectory(visited, subject_path)
        person_imgs = os.listdir(dir)
        for img_path in person_imgs:
            if img_path.endswith(".tif") and "fa" in img_path:
                tj = get_image(dir + "/" + img_path)
                vectors.append(ti - tj)
    return vectors


def generate_C2(subject_path):
    vectors = []
    for file in os.listdir(subject_path):
        if file.endswith(".tif") and "fa" in file:
            ti = get_image(subject_path + "/" + file)
            vectors += C2_helper(subject_path, ti)
    return vectors


def generate_PCA(dataset):
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        dataset.data, dataset.imgname, test_size=0.2, random_state=0
    )
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)

    pca = PCA(0.95)  # mantener 95% de varianza en el dataset, pca elige el n_components
    pca.fit(train_img)
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)
    # ya se redujo la dimensionalidad


def train(path):
    """
    path: directory with photos of a single subject 
    """

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

    clf = svm.SVC(kernel="linear", C=1)
    print("Fitting...")
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print("Fitted in {} seconds".format(end - start))

    return clf

def predict(ID, photo):
    with open("svms/" + ID + ".pkl", 'rb') as f: 
        svm = pickle.load(f)
    photo = get_image(photo)
    photo = photo.reshape(1, -1)
    y_pred = svm.predict(photo)

    return y_pred

def save_svm(svm, path):
    with open("svms/" + path + ".pkl", 'wb') as f: 
        pickle.dump(svm, f)
    print("Saved SVM to svms/" + path + ".pkl")

if __name__ == "__main__":
    basePath = "colorferet/dvd2/"

    for cd in [1, 2]:
        for dir in os.listdir(basePath + f"gray_feret_cd{cd}/data/images"):
            print("Training on {}".format(dir))
            single_svm = train(basePath + f"gray_feret_cd{cd}/data/images/" + dir)
            save_svm(single_svm, dir)
            del single_svm