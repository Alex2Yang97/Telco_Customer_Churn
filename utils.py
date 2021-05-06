# -*- coding:utf8 -*-
"""
@Author: Zhirui(Alex) Yang
@Date: 2021/5/1 下午11:36
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly
import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# import plotly.graph_objs as go

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve, auc


def read_from_csv(filename, dir_path='./', **kwargs):
    file_path = os.path.join(dir_path, filename)
    df = pd.read_csv(file_path, **kwargs)
    return df


def save_to_csv(df, filename, dir_path='./', index=False,
                  encoding='utf-8-sig', **kwargs):
    file_path = os.path.join(dir_path, filename)
    kwargs.update({'index': index, 'encoding': encoding})
    df.to_csv(file_path, **kwargs)


def cal_dis_cate_fea(df, cat_col):
    # calculate for churn data
    churn_dis = df[df['Churn'] == 1][cat_col].value_counts().reset_index(drop=False)
    churn_dis.columns = [cat_col, 'counts']
    churn_dis['Churn'] = 1

    # calculate for no churn data
    no_churn_dis = df[df['Churn'] == 0][cat_col].value_counts().reset_index(drop=False)
    no_churn_dis.columns = [cat_col, 'counts']
    no_churn_dis['Churn'] = 0
    cate_dis = pd.concat([churn_dis, no_churn_dis])

    # calculate for the whole data set
    all_dis = df[cat_col].value_counts().reset_index(drop=False)
    all_dis.columns = [cat_col, 'all_counts']

    # percent means that
    cate_dis = pd.merge(cate_dis, all_dis, on=cat_col, how='left')
    cate_dis['percent'] = cate_dis['counts'] / cate_dis['all_counts']
    cate_dis = cate_dis.sort_values(by=[cat_col, 'Churn'])

    return cate_dis.reset_index(drop=True)


def plot_for_cate(cate_dis, cat_col):
    fig = make_subplots(
        rows=1, cols=1,
        print_grid=True,
        horizontal_spacing=0.15,
        subplot_titles=(f"Distribution of {cat_col}")
    )

    churn_dis = cate_dis[cate_dis['Churn'] == 1]
    trace1 = go.Bar(
        x=churn_dis[cat_col],
        y=churn_dis['counts'],
        name='Churn', opacity=0.7,
        text=churn_dis['counts'], textposition='auto',
        marker=dict(color='seagreen'))

    no_churn_dis = cate_dis[cate_dis['Churn'] == 0]
    trace2 = go.Bar(
        x=no_churn_dis[cat_col],
        y=no_churn_dis['counts'],
        name='No Churn', opacity=0.7,
        text=no_churn_dis['counts'], textposition='auto',
        marker=dict(color='indianred'))

    # Calculate the percentage of every categorical value in Churn/No Churn to
    # every categorical value in the whole data set
    # This is use to check whether the feature has the ability to distinguish Churn/No Churn
    diff_dis = cate_dis[cate_dis['Churn'] == 1]
    trace3 = go.Scatter(
        x=diff_dis[cat_col],
        y=diff_dis['percent'],
        yaxis='y2',
        name='Ratio', opacity=0.6,
        marker=dict(color='black')
    )

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)

    # fig is an object like dict
    # Add another y axis for % Churn
    fig['data'][2].update(yaxis='y2')

    fig['layout']['xaxis'].update(
        autorange=True,
        tickfont=dict(size=10),
        title=f'{cat_col}',
        type='category'
    )

    fig['layout']['yaxis'].update(title='Count')
    fig['layout']['yaxis2']=dict(
        range=[0, 1], #right y-axis in subplot (1,1)
        overlaying='y',
        anchor='x',
        side='right',
        showgrid=False,
        title=f'{cat_col} in Churn Ratio'
    )

    # fig['layout']['title'] = f"{cat_col} Distributions"
    fig['layout']['annotations'][0]['text'] = f'{cat_col} Distribution'
    fig['layout']['height'] = 500
    fig['layout']['width'] = 1000

    fig.show()


def cal_charge_for_cat(df, cat_col, charge_type='MonthlyCharges'):
    df_tmp = round(df.groupby(['Churn', cat_col])[charge_type].mean().reset_index(), 2)
    tmp_churn = df_tmp[df_tmp['Churn'] == 1]
    tmp_no_churn = df_tmp[df_tmp['Churn'] == 0]

    charge_df = pd.merge(tmp_churn, tmp_no_churn, on=cat_col,
                         how='inner', suffixes=('_Churn', ''))

    charge_df[f'diff_{charge_type}'] = round(
        (charge_df[f'{charge_type}_Churn'] / charge_df[f'{charge_type}']) - 1, 2
    )

    return charge_df


def plot_charge_for_cat(charge_df, cat_col, charge_type='MonthlyCharges', size=10):
    fig = make_subplots(
        rows=1, cols=1,
        print_grid=True,
        horizontal_spacing=0.15,
        subplot_titles=(f"Distribution of {cat_col}")
    )

    trace1 = go.Bar(
        x=charge_df[cat_col],
        y=charge_df[f"{charge_type}_Churn"],
        showlegend=True,
        name='Churn', opacity=0.7,
        text=charge_df[f"{charge_type}_Churn"], textposition='auto',
        marker=dict(color='seagreen'))

    trace2 = go.Bar(
        x=charge_df[cat_col],
        y=charge_df[charge_type],
        showlegend=True,
        name='No Churn', opacity=0.7,
        text=charge_df[charge_type], textposition='auto',
        marker=dict(color='indianred'))

    trace3 = go.Scatter(
        x=charge_df[cat_col],
        y=charge_df[f'diff_{charge_type}'],
        yaxis='y2',
        name=f'Diff {charge_type} Churn/No Churn', opacity=0.6,
        # text=charge_df[f'diff_{charge_type}'], textposition='top center',
        marker=dict(color='black'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)

    fig['data'][2].update(yaxis='y2')

    fig['layout']['xaxis'].update(
        autorange=True,
        tickfont=dict(size=size),
        title=f'{cat_col}',
        type='category')
    fig['layout']['yaxis'].update(title=f'Mean {charge_type}')

    fig['layout']['yaxis2'] = dict(
        range=[-1, 1], #right y-axis in the subplot (1,2)
        # autorange=True,
        overlaying='y',
        anchor='x',
        side='right',
        showgrid=False,
        title=f'{charge_type} Difference'
                             )
    # fig['layout']['title'] = f"{cat_col} Distributions"
    fig['layout']['annotations'][0]['text'] = f'{charge_type} of {cat_col} Distribution'
    fig['layout']['height'] = 500
    fig['layout']['width'] = 1000

    # iplot(fig)
    fig.show()


def get_scores(true_y, pred_y, pred_prob):
    fpr, tpr, thresholds = roc_curve(true_y, pred_prob[:, 1].reshape(-1, 1))

    score_df = pd.DataFrame({
        # accuracy =（TP+TN）/ (P+N)
        'accuracy': [accuracy_score(true_y, pred_y.reshape(-1, 1))],
        # precision = TP / (TP+FP)
        'precision': [precision_score(true_y, pred_y.reshape(-1, 1))],
        # recall = TP / (TP+FN) = TP / P = sensitive
        'recall': [recall_score(true_y, pred_y.reshape(-1, 1))],
        'f1': [f1_score(true_y, pred_y.reshape(-1, 1))],
        'auc': [auc(fpr, tpr)]
    })

    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })

    return score_df, roc_df


def plot_roc(roc_df, model_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_df['fpr'],
            y=roc_df['tpr'],
            mode='lines',
            name='ROC'))
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            showlegend=False,
            line=dict(dash='dot')))

    #     fig['layout']['height'] = 600
    #     fig['layout']['width'] = 800

    fig.update_layout(
        height=600,
        width=800,
        title=f'{model_name} Roc Curve',
        xaxis_title='False positive Rate',
        yaxis_title='True Positive Rate')

    fig.show()