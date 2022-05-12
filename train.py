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
        if file.endswith(".ppm"):
            ti = get_image(path + '/' + file)
            for file2 in os.listdir(path):
                if file2 != file and file2.endswith(".ppm"):
                    tj = get_image(path + '/' + file2)
                    vectors.append(ti - tj)
    return vectors


def chooseDirectory():
    idx = rd.randint(1, 1012)
    dir = '/data/images/' + str(idx).zfill(5)
    if idx < 740:
        dir = 'dvd1' + dir
    else:
        dir = 'dvd2' + dir
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
            if img_path.endswith(".ppm"):
                tj = get_image(dir + '/' + img_path)
                vectors.append(ti - tj)
    return vectors


def generate_C2(path):
    vectors = []
    for file in os.listdir(path):
        if file.endswith(".ppm"):
            ti = get_image(path + '/' + file)
            vectors += C2_helper(path, ti)
    return vectors


def train(path):
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

    X = np.array(C1 + C2)
    X = X.reshape(len(X), -1)
    y = np.array([1] * len(C1) + [0] * len(C2))

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, y)


if __name__ == '__main__':
    basePath = '/data/images/'
    for dir in os.listdir('dvd1' + basePath):
        train(basePath + dir)
    for dir in os.listdir('dvd2' + basePath):
        train(basePath + dir)
