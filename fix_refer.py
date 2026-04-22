import re

path = "/net/tscratch/people/plgabedychaj/ctrl-o/refer/refer.py"
with open(path) as f:
    src = f.read()

src = re.sub(r'\bprint (.*)', r'print(\1)', src)
src = src.replace("import cPickle as pickle", "import pickle")
src = src.replace("from external import mask", "from pycocotools import mask")
# Python 3: pickle requires binary mode
src = src.replace("pickle.load(open(ref_file, 'r'))", "pickle.load(open(ref_file, 'rb'))")

with open(path, "w") as f:
    f.write(src)
print("refer.py fixed.")
