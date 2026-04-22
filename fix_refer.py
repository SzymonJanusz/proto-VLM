import re

path = "/net/tscratch/people/plgabedychaj/ctrl-o/refer/refer.py"
with open(path) as f:
    src = f.read()

src = re.sub(r'\bprint (.*)', r'print(\1)', src)
src = src.replace("import cPickle as pickle", "import pickle")
src = src.replace("from external import mask", "from pycocotools import mask")

with open(path, "w") as f:
    f.write(src)
print("refer.py fixed.")
