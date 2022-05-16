import os
import shutil
import json

# import utils

from random import randint, choice, sample
from imageio import imread


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


if __name__ == "__main__":
    basePath = "colorferet/dvd2/"

    # individuals = utils.get_train_test_subjects(basePath)
    with open("train_subjects.json") as f:
        individuals = json.loads(f)
    test_set = {individual: None for individual in individuals}

    clear_folder("test_set/")

    for individual in individuals:
        individual_code = individual.split("/")[-1]
        os.mkdir("test_set/" + individual_code)
        imgs = [img for img in os.listdir(individual) if img.endswith(".tif")]
        sampled_images = sample(imgs, 2)

        for img in sampled_images:
            img_path = individual + "/" + img
            # shutil.move(img_path, "test_set/" + individual_code + "/")

        test_set[individual] = sampled_images

    with open("selected_for_test.json", "w") as f:
        json.dump(test_set, f, indent=2)
