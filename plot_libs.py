import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
import math
import json 
import glob
import h5py
import time
import shutil
import pickle
import pandas as pd
from netCDF4 import Dataset, num2date, date2index, date2num
from scipy import interpolate
#import datetime

sys.path.append('/Users/xchen224/anaconda3/envs/xcpy/lib/python3.7/site-packages/mylib/')
#import tropomifile as fep
#import pytropomi as ptr
import scatter_plot as sca

