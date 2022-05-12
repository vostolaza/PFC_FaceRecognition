import os
import shutil
import json

from random import randint, choice, sample
from imageio import imread

def get_test_subjects(test_size=0.1):
    dirs = ['dvd1/data/images','dvd2/data/images']
    individuals = {dirs[0]: os.listdir(dirs[0]), dirs[1]: os.listdir(dirs[1])}
    n_total = len(os.listdir(dirs[0])) + len(os.listdir(dirs[1]))
    test_size = int(n_total*test_size)

    chosen_individuals = []

    while test_size > 0:
        chosen_dir = choice(dirs) 
        chosen_individual = choice(individuals[chosen_dir])
        chosen_individuals.append(chosen_dir + '/' + chosen_individual)
        individuals[chosen_dir].remove(chosen_individual)
        test_size -= 1

    return chosen_individuals

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':

    individuals = get_test_subjects()
    test_set = {individual: None for individual in individuals}
    
    clear_folder('test_set/')

    for individual in individuals:
        individual_code = individual.split('/')[-1]
        os.mkdir('test_set/' + individual_code)
        imgs = [img for img in os.listdir(individual) if img.endswith('ppm')]
        sampled_images = sample(imgs, 2)

        for img in sampled_images:
            img_path = individual + '/' + img
            shutil.move(img_path, 'test_set/' + individual_code + '/')

        test_set[individual] = sampled_images

    with open('selected_for_test.json', 'w') as f:
        json.dump(test_set, f, indent=2)

