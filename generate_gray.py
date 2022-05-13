import os
import shutil

if __name__ == '__main__':
    dir_list = os.listdir('images')
    for img in dir_list:
        if img.endswith('.tif'):
            code = img[:5]
            print(code)
            if not os.path.exists('images/' + code):
                os.mkdir('images/' + code)
            shutil.move('images/' + img,
                        'images/' + code + '/' + img)
