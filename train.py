import numpy as np
import random as rd
import os
import time

from imageio import imread
from sklearn import svm

N_SAMPLES = 10


def get_image(path):
    return np.array(imread(path))


def generate_C1(path):
    vectors = []
    for file in os.listdir(path):
        if file.endswith(".tif"):
            ti = get_image(path + '/' + file)
            for file2 in os.listdir(path):
                if file2 != file and file2.endswith(".tif"):
                    tj = get_image(path + '/' + file2)
                    vectors.append(ti - tj)
    return vectors


def chooseDirectory():
    idx = rd.randint(1, 1012)
    dir = '/data/images/' + str(idx).zfill(5)
    if idx < 740:
        dir = 'gray_feret_cd1' + dir
    else:
        dir = 'gray_feret_cd2' + dir
    # check if dir exists
    if not os.path.exists(dir):
        return chooseDirectory()
    return dir


def C2_helper(image_path, ti):
    vectors = []
    for i in range(N_SAMPLES):
        dir = chooseDirectory()
        person_imgs = os.listdir(dir)
        for img_path in person_imgs:
            if img_path.endswith(".tif") and 'fa' in img_path:
                tj = get_image(dir + '/' + img_path)
                vectors.append(ti - tj)
    return vectors


def generate_C2(path):
    vectors = []
    for file in os.listdir(path):
        if file.endswith(".tif") and 'fa' in file:
            ti = get_image(path + '/' + file)
            vectors += C2_helper(path, ti)
    return vectors


def train(path, clf):
    print("Generating C1...")
    start = time.time()
    C1 = generate_C1(path)
    end = time.time()
    print("Generated C1 in {} seconds".format(end - start))

    print("Generating C2...")
    start = time.time()
    C2 = generate_C2(path)
    end = time.time()
    print("Generated C2 in {} seconds".format(end - start))

    print("C1 Size:", np.array(C1).shape)
    print("C2 Size:", np.array(C2).shape)
    X = np.array(C1 + C2)
    X = X.reshape(len(X), -1)
    y = np.array([1] * len(C1) + [0] * len(C2))

    print("Fitting...")
    start = time.time()
    clf.fit(X, y)
    end = time.time()
    print("Fitted in {} seconds".format(end - start))


if __name__ == '__main__':
    clf = svm.SVC(kernel='linear', C=1)
    basePath = '/data/images/'
    for dir in os.listdir('gray_feret_cd1' + basePath):
        print("Training on {}".format(dir))
        train('gray_feret_cd1' + basePath + dir, clf)
    for dir in os.listdir('gray_feret_cd2' + basePath):
        print("Training on {}".format(dir))
        train('gray_feret_cd2' + basePath + dir, clf)
