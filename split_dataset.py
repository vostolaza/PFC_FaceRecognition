from random import randint
import os
import json

NSAMPLES = 100

def generate_set(output_path):
    d = {}
    for i in range(NSAMPLES):
        set = []
        while (len(set) < 2):
            idx = randint(0, 1209)
            folder = 'images/' + str(idx).zfill(5)
            if not os.path.exists(folder): continue
            for img in os.listdir(folder):
                if img.endswith(".tif") and ('fa' in img or 'fb' in img):
                    img_path = folder + '/' + img
                    set.append(img_path)
                if len(set) == 2: break
        d[idx] = set
    with open(output_path, 'w') as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    generate_set('train_set.json')
    generate_set('test_set.json')