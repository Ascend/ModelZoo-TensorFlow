# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
from rdkit.Chem import SaltRemover
from rdkit import Chem
import shutil
import csv
import pickle

FIRST_LINE = True
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--apply_file", type=str, default="./data/test.csv")
args = parser.parse_args()
APPLY_FILE = args.apply_file
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
g_chars = set(chars)
vocab_size = len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
CONV_OFFSET = 20
props = {}

class suppress_stderr(object):
   def __init__(self):
       self.null_fds = [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]
   def __enter__(self):
       os.dup2(self.null_fds[0],2)
   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)
       for fd in self.null_fds + self.save_fds:
          os.close(fd)


# input_p0.tofile("./bin_input/input_p0/{0:05d}.bin".format(0))
def gen_data(data, props):

    batch_size = len(data);

    #search for max lengths
    nl = len(data[0][0]);
    nl = vocab_size    #########test
    for i in range(1, batch_size, 1):
        nl_a = len(data[i][0]);
        if nl_a > nl:
            nl = nl_a;

    nl = nl + CONV_OFFSET;
    x = np.zeros((batch_size, nl), np.int8);
    mx = np.zeros((batch_size, nl), np.int8);

    z = [];
    ym = [];

    for i in range(len(props)):
       z.append(np.zeros((batch_size, 1), np.float32));
       ym.append(np.zeros((batch_size, 1), np.int8));

    for cnt in range(batch_size):

        n = len(data[cnt][0]);
        for i in range(n):
           x[cnt, i] = char_to_ix[ data[cnt][0][i]] ;
        mx[cnt, :i+1] = 1;

        for i in range(len(props)):
           z[i][cnt] = data[cnt][1][i];
           ym[i][cnt ] = data[cnt][2][i];

    d = [x, mx];

    for i in range(len(props)):
       d.extend([ym[i]]);

    return d, z;



def preprocess():

    first_row = FIRST_LINE;
    DS = [];
    props = pickle.load(open("model.pkl", "rb"));
    ind_mol = 0;

    if not os.path.exists("./input_bins"):
        os.makedirs("./input_bins")
    if not os.path.exists("./input_bins/input_x0"):
        os.makedirs("./input_bins/input_x0")
    if not os.path.exists("./input_bins/input_x1"):
        os.makedirs("./input_bins/input_x1")


    index = 0
    remover = SaltRemover.SaltRemover();
    for row in csv.reader(open(APPLY_FILE, "r")):
        if first_row:
            first_row = False;
            continue;

        mol = row[ind_mol];
        g_mol = set(mol);
        g_left = g_mol - g_chars;

        arr = [];
        if len(g_left) == 0:
            try:
                with suppress_stderr():
                    m = Chem.MolFromSmiles(mol);
                    m = remover.StripMol(m);
                    if m is not None and m.GetNumAtoms() > 0:
                        for step in range(10):
                            arr.append(Chem.MolToSmiles(m, rootedAtAtom=np.random.randint(0, m.GetNumAtoms()),
                                                        canonical=False));
                    else:
                        for step in range(10):
                            arr.append(mol);
            except:
                for step in range(10):
                    arr.append(mol);


        z = np.zeros(len(props), dtype=np.float32);
        ymask = np.ones(len(props), dtype=np.int8);

        d = [];
        for i in range(len(arr)):
            d.append([arr[i], z, ymask]);

        x, y = gen_data(d, props)
        x[0] = x[0].astype(np.float32)
        x[1] = x[1].astype(np.float32)
        x[0].tofile("./input_bins/input_x0/{0:03d}.bin".format(index))
        x[1].tofile("./input_bins/input_x1/{0:03d}.bin".format(index))
        index += 1
        print("Current iteration:{}".format(index))

    print("Relax!")


if __name__ == "__main__":
    preprocess()