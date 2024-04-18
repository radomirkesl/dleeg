from scipy.io import loadmat
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

all_data = loadmat('../data/S1_Session_9.mat', simplify_cells=True)['BCI']

print(all_data['chaninfo']['label'])
print(np.where(all_data['chaninfo']['label'] == 'C3'))

