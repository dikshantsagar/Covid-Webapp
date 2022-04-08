
import pickle
import numpy as np
import pandas as pd
import json
import glob

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import shap
import joblib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from numba import cuda 
import gc

plt.style.use('ggplot')

# input format  : (samples, 12, 1000, 1)
# output format : (samples, 5)
def get_prediction(data):
  # load model
    model = load_model('static/data/models/ST-CNN-GAP-5.h5')
    print('Trainable Parameters:', np.sum([K.count_params(w) for w in model.trainable_weights]))

  # output prediction
    y_pred  = model.predict(data)[0]
    
    probs = y_pred.copy()
    print(probs)

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5]  = 0

  # multi-labelled data: 1 means positive and 0 means negative for each index
    return y_pred, probs
    

# input format  : (samples, 12, 1000, 1)
# output format : (num_classes, samples, 12, 1000)
def get_shap(data):
    # load model
    model = load_model('static/data/models/ST-CNN-GAP-5.h5')
    
    # find shap values
    x_train = np.load('static/data/x_train.npy')
    
    x_train = x_train.transpose(0, 2, 1)                    # transpose matrix
    x_train = np.expand_dims(x_train, axis=-1)              # Add another channel
    explainer = shap.GradientExplainer(model, x_train)

    #print('saving shap explainer')
    

    shap_vals = explainer.shap_values(data)
    
    
    
    # remove the extra dimension at the end
    shap_vals = np.squeeze(shap_vals, axis=-1)

    return shap_vals

def top500(vals):
  
  arr = np.copy(vals)
  #print(arr.shape)
  arr = arr.flatten()
  arr.sort()
  
  threshold = arr[-500]
  vals[ vals < threshold] = -1. 
  #print(np.count_nonzero(vals >= 0))
  return vals


def plot_shap(x, y, shap_val, lead_idx, ind):
  mn, mx = np.min(y), np.max(y)
  
  fig, axs = plt.subplots()

  # Create a set of line segments so that we can color them individually
  # This creates the points as a N x 1 x 2 array so that we can stack points
  # together easily to get the segments. The segments array for line collection
  # needs to be (numlines) x (points per line) x 2 (for x and y)
  leads  = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)

  cmap = ListedColormap(['blue', 'red'])
  norm = BoundaryNorm([-10, 0, 10], cmap.N)
  
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  lc.set_array(shap_val)
  lc.set_linewidth(0.8)
  line = axs.add_collection(lc)
  #fig.colorbar(line, ax=axs)
  limlow = mn if mn > -0.5 else  -0.6
  limmax = mx if mx > 0.5 else 0.6
  plt.xlim(-0.2, 10.2)
  plt.ylim(mn - 0.1, mx + 0.1)
  plt.xticks(np.arange(0, 10.04, step=0.04), labels=[]) 
  plt.yticks(np.arange(mn, mx, step=0.1), labels=[]) 

  fig.set_size_inches(35, 2)
  plt.title("Lead " + leads[lead_idx],fontdict = {'fontsize' : 25})
  #plt.xlabel("Time (sec)")
  #plt.ylabel("Voltage (mV)")
  path = 'static/data/plots/'+str(ind)+'/'+leads[lead_idx]+ '.png'
  plt.savefig(path, bbox_inches='tight', dpi=600)
  plt.close(fig)
  plt.figure().clear()
  plt.close('all')
  plt.cla()
  plt.clf()
  
  gc.collect()


  return path