import os
import bz2
import shutil

path = "."

for path, dirs, files in os.walk(path):
    for filename in files:
        basename, ext = os.path.splitext(filename)
        if ext.lower() != ".bz2":
            continue
        fullname = os.path.join(path, filename)
        newname = os.path.join(path, basename)
        with bz2.open(fullname) as fh, open(newname, "wb") as fw:
            shutil.copyfileobj(fh, fw)
