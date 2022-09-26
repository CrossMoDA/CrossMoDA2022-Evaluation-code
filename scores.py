#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: reubendo
"""

from medpy.metric import dc, assd
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
import nibabel as nib

MAX_VALUE_ASSD = 350 #mm
DEBBUGING =  True

# Path data
path_joint= '/media/nas/crossmoda2022/predictions_seg/{}/{}'
path_output = '/media/nas/crossmoda2022/results/score_test.csv'
path_gt = '/media/nas/crossmoda2022/official_data/testing/{}'

# List cases for inference
cases = [k for k in os.listdir(path_gt.format(''))  if  'Label' in k]
cases = natsorted(cases)
print(f"Number of cases: {len(cases)} \n")

# Metrics:
metrics = ['VS_Dice', 'VS_ASSD', 'Cochlea_Dice', 'Cochlea_ASSD']

# Teams
teams = os.listdir(path_joint.format('',''))

# Testing shape cases
if not DEBBUGING:
    error = []
    for team in teams:
        for case in tqdm(cases):
            gt_array = nib.load(path_gt.format(case))
            pred_array = nib.load(path_joint.format(team, case))
            if not pred_array.shape==gt_array.shape:
                error.append((team,case))
        if len(error)>0:
            print(f"errors with {team}: {error}")
    

if DEBBUGING:
    cases = cases[::5]

# Creating one csv per team with all scores per cases
list_df_team = []
for team in teams:
    print( '-'*5 + team + '-'*5 + '\n')
    df_metric = {
        'Case':[],
        'VS_Dice':[],
        'VS_ASSD':[],
        'Cochlea_Dice':[],
        'Cochlea_ASSD':[]
        }
    for case in tqdm(cases):
        assert os.path.exists(path_joint.format(team, case))
        
        df_metric['Case'].append(case)
        
        # Numpy arrays
        gt_array = nib.load(path_gt.format(case)).get_fdata()
        pred_array = nib.load(path_joint.format(team, case)).get_fdata()
        
        # Voxel spacing
        affine = nib.load(path_gt.format(case)).affine
        vxlspacing = [abs(affine[k,k]) for k in range(3)]
        
        # VS
        gt_VS = (gt_array==1).astype(int)
        pred_VS = (pred_array==1).astype(int)

        df_metric['VS_Dice'].append(dc(pred_VS, gt_VS))
        if DEBBUGING:
            df_metric['VS_ASSD'].append(10)
        else:
            if np.sum(pred_VS)>0:
                df_metric['VS_ASSD'].append(assd(pred_VS, gt_VS, voxelspacing=vxlspacing))
            else:
                df_metric['VS_ASSD'].append(MAX_VALUE_ASSD)

       	# Cochlea
        gt_cochlea = (gt_array==2).astype(int)
        pred_cochlea = (pred_array==2).astype(int)

        df_metric['Cochlea_Dice'].append(dc(pred_cochlea, gt_cochlea))
        if DEBBUGING:
            df_metric['Cochlea_ASSD'].append(10)
        else:
            if np.sum(pred_cochlea)>0:
                df_metric['Cochlea_ASSD'].append(assd(pred_cochlea, gt_cochlea, voxelspacing=vxlspacing))
            else:
                df_metric['Cochlea_ASSD'].append(MAX_VALUE_ASSD)
            
    for metric in ['VS_Dice', 'VS_ASSD', 'Cochlea_Dice', 'Cochlea_ASSD']:
        print(f'{metric}: {np.mean(df_metric[metric])}')
        
    df = pd.DataFrame(df_metric)
    df['Team'] = [str(team)] * len(cases)
    list_df_team.append(df)
    
# Create one dataframe for all
df_metrics = pd.concat(list_df_team).reset_index(drop=True)
df_metrics['Case'] = df_metrics['Case'].str.split('_Label').str[0]
df_metrics.to_csv(path_output, index=False)