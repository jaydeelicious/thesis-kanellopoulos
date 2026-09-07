import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import datetime
from ..utils import misc

from tqdm import tqdm
from matplotlib.pyplot import figure
from torchvision import transforms
from icecream import ic
ic.disable()

from ..dataloader import dataset_csv
from . import tralzformer
from ..utils import misc

basedir="/home/kanellopoulos/thesis/tralzformer_baseline/"

fname = 'ModalityB_02_09_01_BaselineFinal_fold_1_Preds'
save_path = f'./model_predictions/'
dat_file = '/home/kanellopoulos/thesis/tralzformer_baseline/data/adni/ModalityB_Dataset_Test.csv'
cnf_file = f'{basedir}/data/toml/modality_b_conf_baseline.toml'
ckpt_path = '/home/kanellopoulos/thesis/tralzformer_baseline/ckpt/Modality_B_Exp_31_08_01_BaselineCausal_fold_5.pt'

dat_file = pd.read_csv(dat_file, sep=';')
print(dat_file)

if not os.path.exists(save_path):
    os.makedirs(save_path)

device = 'cuda:1'
mdl = tralzformer.TralzformerModel.from_ckpt(ckpt_path, device=device)
print("loaded")

from matplotlib import rc, rcParams

rc('axes', linewidth=1)
rc('font', size=18)
plt.rcParams['font.family'] = 'Arial'

def read_csv(filename):
    return pd.read_csv(filename)

def save_predictions(dat_tst, test_file, scores_proba, scores, save_path=None, filename=None, if_save=True):
    y_true = [{k: int(v) if v is not None else np.NaN for k, v in entry.items()} for entry in dat_tst.labels]
    mask = [{k: 1 if v is not None else 0 for k, v in entry.items()} for entry in dat_tst.labels]

    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in test_file.columns}
    scores_proba_ = {f'{k}_prob': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in test_file.columns}
    scores_ = {f'{k}_logit': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in test_file.columns}

    ids = test_file['PTID']

    y_true_df = pd.DataFrame(y_true_)
    scores_df = pd.DataFrame(scores_)
    scores_proba_df = pd.DataFrame(scores_proba_)

    id_df = pd.DataFrame(ids)

    df = pd.concat([id_df, y_true_df, scores_proba_df], axis=1)

    print(len(y_true_df), len(scores_df), len(scores_proba_df), len(id_df))

    if if_save:
        df.to_csv(save_path + filename, index=False)

    return df

def generate_performance_report(dat_tst, y_pred, scores_proba):
    y_true = [{k: int(v) if v is not None else 0 for k, v in entry.items()} for entry in dat_tst.labels]
    mask = [{k: 1 if v is not None else 0 for k, v in entry.items()} for entry in dat_tst.labels]

    y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
    y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
    scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
    mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

    met = {}

    for k in dat_tst.label_modalities:
        print('Performance metrics of {}'.format(k))
        metrics = misc.get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
        misc.print_metrics(metrics)

        met[k] = metrics
        met[k].pop('Confusion Matrix')

    return met

def generate_performance_report(dat_tst, y_pred, scores_proba):
    y_true = [{k: int(v) if v is not None else 0 for k, v in entry.items()} for entry in dat_tst.labels]
    mask = [{k: 1 if v is not None else 0 for k, v in entry.items()} for entry in dat_tst.labels]

    y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
    y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
    scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
    mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

    ordered_keys = y_true_dict.keys()

    y_true_arr = np.stack([y_true_dict[k] for k in ordered_keys], axis=1)  # [N, T]
    y_pred_arr = np.stack([y_pred_dict[k] for k in ordered_keys], axis=1)  # [N, T]
    scores_arr = np.stack([scores_proba_dict[k] for k in ordered_keys], axis=1)  # [N, T]
    mask_arr = np.stack([mask_dict[k] for k in ordered_keys], axis=1)   # [N, T]

    binary_y_true_test = (y_true_arr == 2).astype(np.int32)

    met_pred = misc.get_metrics_multitask(
        binary_y_true_test,
        y_pred_arr,
        scores_arr,
        mask_arr
    )

    return met_pred



def generate_predictions_for_data_file(dat_file):
    seed = 0
    print('Done.\nLoading testing dataset...')
    dat_tst = dataset_csv.CSVDataset(data_file=dat_file, cnf_file=cnf_file, mode=0)

    print('Done.')

    print('Generating model predictions')
    scores, scores_proba, y_pred = mdl.predict(x=dat_tst.features, _batch_size=64)
    print('Done.')

    print('Saving model predictions')
    df = save_predictions(dat_tst, dat_file, scores_proba, scores, save_path, f'{fname}_prob.csv', if_save=True)
    print('Done.')

    print('Generating performance reports')
    met = generate_performance_report(dat_tst, y_pred, scores_proba)
    print('Done.')
    return df, met


if __name__ == '__main__':
    df_pred, met = generate_predictions_for_data_file(dat_file)

    met_df = pd.DataFrame(met)
    met_df.to_csv(save_path + f'{fname}_performance_report.pdf', index=False)
    print(met_df.round(2))

