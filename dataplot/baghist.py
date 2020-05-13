"""This module present several functions histogram based in bag data."""

import logging
import os
import sys

import seaborn as sns
import numpy as np
import pandas as pd

from scipy import stats as scipystats
from scipy.stats import spearmanr
from sklearn.utils import resample
from quandarium.analy.aux import checkmissingkeys
from quandarium.analy.aux import to_list
from quandarium.analy.aux import to_nparray

import matplotlib
import matplotlib.pylab as plt
import matplotlib.lines as mlines
#from matplotlib import rc                         # For laxtex in ploting
from matplotlib.ticker import FormatStrFormatter  # For tick labels
from matplotlib.legend import Legend



#rc('text', usetex=True)
#rc('text.latex', preamble=r'\usepackage{mhchem} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{mathtools} \usepackage[T1]{fontenc} ')

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.WARNING)
logging.info('The logging level is INFO')

def histbag(figname, bag, grupbybag):

    print('Initializing histbag.')
    logging.info('Initializing histbag.')

    bag = to_list(bag)
    grupbybag = to_list(grupbybag)
    data = []
    for i, _ in enumerate(bag):
        data += to_list(bag[i])
    data = np.array(data)

    splitfeaturedata = []
    for i in range(len(bag)):
        splitfeaturedata += to_list(grupbybag[i])
    splitfeaturedata = np.array(splitfeaturedata)

    split_vals = np.unique(splitfeaturedata)

    plt.close('all')
    for val in split_vals:
        dataval = data[splitfeaturedata == val]
        g = sns.distplot(dataval, hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3},
                         label = val)
    g.get_figure().savefig(figname)
