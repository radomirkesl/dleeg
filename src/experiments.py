import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from tqdm import tqdm

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

all_data = loadmat('../data/S1_Session_9.mat', simplify_cells=True)['BCI']

print(type(all_data['chaninfo']['label']))
print(all_data['chaninfo']['label'])
print(np.where(all_data['chaninfo']['label'] == 'C3'))
print(all_data['chaninfo']['noisechan'])

