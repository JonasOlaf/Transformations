import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
#import torch
import sys
sys.path.append('faceHE/')

# My files
from DataSerilization import save_db, get_db, list_dbs
from Transforms import compute_int_bins, transform_to_int, transform_to_bin, transform_to_int_single, transform_to_bin_single
from plots import dist_curve, DET_identification
from ANN_helpers import unbatch, transform_model, Model
from Helpers import unroll_db, inroll_db, identification
from Pca import PCA
import pdb
from sklearn.utils import shuffle

import bisect
from math import isclose
from random import shuffle, seed


db_train = get_db('vgg/2/b1')
db_test = get_db('vgg/2/b2')

pca = PCA(db_train, 32)
db_new = pca.transform(db_test)

DET_identification(db_new, 'example')


# #
