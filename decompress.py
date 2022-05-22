import os
import bz2
import shutil

for i in range(1,3):
    dir = os.listdir(f'colorferet/dvd2/gray_feret_cd{i}/data/images')
    for img in dir:
        if img[-4:] != ".bz2":
            continue
        fullname = f'colorferet/dvd2/gray_feret_cd{i}/data/images/' + img
        with bz2.open(fullname) as fh, open('images/'+img[:-4], "wb") as fw:
            shutil.copyfileobj(fh, fw)
    for img in os.listdir('images'):
        if img.endswith('.tif'):
            code = img[:5]
            print(code)
            if not os.path.exists('images/' + code):
                os.mkdir('images/' + code)
            shutil.move('images/' + img,
                        'images/' + code + '/' + img)
