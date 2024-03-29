import os
import json

IMAGE_PATH = 'images/'

def get_train_test_subjects(basePath):
    id_len = {}
    for cd in [1, 2]:
        for dir in os.listdir(basePath + f"gray_feret_cd{cd}/data/images"):
            num_file = len(
                os.listdir(basePath + f"gray_feret_cd{cd}/data/images/" + dir)
            )
            id_len[dir] = num_file
    id_len = dict(sorted(id_len.items(), key=lambda item: item[1]))
    individuals = []
    for key in id_len:
        if id_len[key] == 6:
            p = (
                basePath
                + (
                    "gray_feret_cd1/data/images/"
                    if int(key) < 700
                    else "gray_feret_cd2/data/images/"
                )
                + key
            )
            individuals.append(p)
    individuals.sort()
    print(len(individuals))
    return individuals[:10]


def score(y_real, y_pred):
    p_v = 0
    p_f = 0
    for y_r, y_p in zip(y_real, y_pred):
        if y_r == 1 and y_p == 1:
            p_v += 1
        if y_r == 1 and y_p == 0:
            p_f += 1
    return p_v / len(y_real), p_f / len(y_pred)
