#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: reubendo
"""

from imblearn.metrics import macro_averaged_mean_absolute_error
import pandas as pd
import os
from natsort import natsorted
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

path_predictions = '/media/nas/crossmoda2022/predictions_class/'
path_gt = '/media/nas/crossmoda2022/official_data/testing/{}'
path_info = 'media/nas/crossmoda2022/data_all.csv'


df = pd.read_csv(path_info) # Private dataframe with all case info

teams = os.listdir(path_predictions)
cases = [k for k in os.listdir(path_gt.format(''))  if  'Label' in k]
cases = natsorted(cases)


# Only considering pre-operative cases
gt = []
preopcases = []
for case in cases:
    koos = df[df['crossmoda_name']==case.split('_Label')[0]]['koos'].values
    assert koos.shape == (1,)
    if not koos[0]=='post-operative-london':
        gt.append(int(koos[0]))
        preopcases.append(case)
print(f"Number of cases: {len(preopcases)} \n")

# Computing scores for each team
scores = dict()
scores_f1 = dict()
for team in teams:
    df_team = pd.read_csv(os.path.join(path_predictions,team,'predictions.csv'))
    pred = []
    for case in preopcases:
        koos = df_team[df_team['case']==case.split('_Label')[0]]['class'].values
        assert koos.shape == (1,)
        pred.append(int(koos[0])) 
    scores[team] =  macro_averaged_mean_absolute_error(gt, pred)
    scores_f1[team] = f1_score(gt, pred, average='macro')
    
    # Plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization ", None),
        ("Normalized confusion matrix " , "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_predictions(
            gt,
            pred,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title + team)
    
for team in teams:
    print(f'{team} - MAMAE: {scores[team]} - F1 {scores_f1[team]}') 