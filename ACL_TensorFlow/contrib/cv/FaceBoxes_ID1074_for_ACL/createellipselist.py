#-*-coding:utf-8-*-
import shutil
import sys
sys.path.append('.')
import os
import argparse





#TODO change parameters
ap = argparse.ArgumentParser()
ap.add_argument( "--split_dir", required=False,default='../FDDB/FDDB-folds/FDDB-folds/',help="dir to FDDB-folds")
ap.add_argument( "--result", required=False,default='../FDDBFile',help="dir to write result")
args = ap.parse_args()


ANNOTATIONS_PATH = args.split_dir
RESULT_DIR = args.result

if not os.access(RESULT_DIR,os.F_OK):
    os.mkdir(RESULT_DIR)

annotations = [s for s in os.listdir(ANNOTATIONS_PATH) if s.endswith('ellipseList.txt')]
image_lists = [s for s in os.listdir(ANNOTATIONS_PATH) if not s.endswith('ellipseList.txt')]
annotations = sorted(annotations)
image_lists = sorted(image_lists)

images_to_use = []
for n in image_lists:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        images_to_use.extend(f.readlines())

images_to_use = [s.strip() for s in images_to_use]
with open(os.path.join(RESULT_DIR, 'faceList.txt'), 'w') as f:
    for p in images_to_use:
        f.write(p + '\n')


ellipses = []
for n in annotations:
    with open(os.path.join(ANNOTATIONS_PATH, n)) as f:
        ellipses.extend(f.readlines())

i = 0
with open(os.path.join(RESULT_DIR, 'ellipseList.txt'), 'w') as f:
    for p in ellipses:

        # check image order
        if 'big/img' in p:
            assert images_to_use[i] in p
            i += 1

        f.write(p)



