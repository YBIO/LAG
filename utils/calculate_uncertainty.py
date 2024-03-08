
#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd


def brierss(brier_forecasts, brierss_ref = None, printres = True):

    brier = brier_forecasts.brier.mean()
    brierref = brier_forecasts.brier_ref.mean()

    res = 1 - brier/brierref
    if printres:
        print(f'Brier Skill Score (calculated) = {res:.4f}')
        if brierss_ref is not None:
            diff = (res - brierss_ref)/brierss_ref
            
            print(f'Brier Skill Score (reference) = {brierss_ref:.4f}')
            print(f'Difference (%): {diff:.1%}')

    return res

def brier_components(probs, groupby, f = 'probwin', o = 'probwin_outcome', outcome_classes = 'party', nbins=10):
    probs = probs.copy()

    probs['bin'] = pd.qcut(probs[f], q = nbins)
        
    _0, RES, UNC, _1 = brier_components_binary(probs, f = f, o = o)

    BS = probs.groupby(forecast, dropna=False)['brier'].sum().mean()
    CAL = BS + RES - UNC

    return CAL, RES, UNC, BS

def brier_components_binary(probs, f = 'probwin', o = 'probwin_outcome'):

    dfp = probs.copy()

    gb = dfp.groupby(by = 'bin')
    ok = gb[o].mean()
    ok.name = 'avg_outcome_bin'

    nk = dfp.bin.value_counts()
    nk.name = 'predictions_bin'

    fk = gb[f].mean()
    fk.name = 'avg_forecast_bin'

    N = nk.sum()

    # calibration - how accurate predicitions are
    CAL = (nk * (fk - ok)**2).sum() / N

    # resolution - how bold predictions are
    obar = (nk * ok).sum() / N
    RES = (nk * (ok - obar)**2).sum() / N

    # uncertainty - how random the predicted phenomenon is
    UNC = obar * (1 - obar)

    # Brier Score - a combination of those
    BS = CAL - RES + UNC

    return CAL, RES, UNC, BS



dfsen = pd.read_csv(filepath_or_buffer = r'./datasets/us_senate_elections.csv')

# per 538`s graph in https://projects.fivethirtyeight.com/checking-our-work/
brier_ss_538 = 0.7888
brier_unc_538 = 0.2451
brier_res_538 = 0.1940

#%%
forecast = ['year', 'state', 'special', 'forecast_date', 'forecast_type']


# transform
gb = dfsen.groupby(by = forecast, dropna = False)

# create identification for each opponent for each race
# party affiliation doesn`t do it: there are senate races with two democrats (CA), for instance
dfsen['candidate_id'] = gb['candidate'].cumcount()
candidate_forecast = forecast + ['candidate_id']

# calculate reference forecast (equal chance to all candidates)
dfsen['candidates'] = gb['candidate_id'].transform('count')
dfsen['probwin_ref'] = 1/dfsen['candidates']

# calculate brier reference score, considering each candidate had equal chance
dfsen['brier_ref'] = (dfsen['probwin_ref'] - dfsen['probwin_outcome'])**2

# calculate brier score for each prediction
dfsen['brier'] = (dfsen['probwin'] - dfsen['probwin_outcome'])**2
briercalc = dfsen.copy()

# %%
# calculate overall brier skill score

# first, calculate the brier score for each prediction
# for instance, if there are three candidates, sum (p-o)**2 across each candidate to obtain a brier score for each race
brier = gb[['brier', 'brier_ref']].sum()
N = brier.count()

# filtering
brierr = brier.reset_index()
brierr.index.name = 'race_prediction'
# brierf = brierr[brierr.forecast_type == 'deluxe']
brierf = brierr.copy()

brier_ss = brierss(brier_forecasts = brierf, brierss_ref = brier_ss_538)
CAL, RES, UNC, BS = brier_components(briercalc, groupby = forecast)
BS_REF = brierf.brier_ref.mean()

BS_SS = 1 - BS/BS_REF
