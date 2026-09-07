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

basedir="/home/kanellopoulos/thesis/thesis-ad-diagnosis/tralzformer/tralzformer_september/"

# fname = 'ModalityB_02_09_01_BaselineFinal_fold_1_Preds'
# save_path = f'./model_predictions/'
# dat_file = '/home/kanellopoulos/thesis/tralzformer_baseline_Final/data/adni/ModalityB_Dataset_Test.csv'
# cnf_file = f'{basedir}/data/toml/modality_b_conf_baseline.toml'
# ckpt_path = '/home/kanellopoulos/thesis/tralzformer_baseline_Final/ckpt/Modality_B_Exp_01_09_02_BaselineCausalNoDropout_fold_5.pt'

fname = 'Exp_03_09_DiagnosisPrediction_9V_fold_5_Preds'
save_path = '/home/kanellopoulos/thesis/thesis-ad-diagnosis/tralzformer/tralzformer_september/model_predictions/'
dat_file = '/home/kanellopoulos/thesis/thesis-ad-diagnosis/tralzformer/tralzformer_september/data/adni/FullCohort_9_Visits_Test.csv'
cnf_file = f'{basedir}/data/toml/diagnosis_prediction_9_visits.toml'
ckpt_path = '/home/kanellopoulos/thesis/thesis-ad-diagnosis/tralzformer/tralzformer_september/ckpt/Exp_03_09_DiagnosisPrediction_9V_fold_5.pt'

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

    # --- IG on the mean-pooling model (post-hoc, frozen checkpoint) ---
    from torch.utils.data import DataLoader
    from ..utils import transformer_dataset  # same module you use in predict_logits

    print("Computing Integrated Gradients...")

    # Build a DataLoader identical to prediction (no shuffle!)
    dat = transformer_dataset.TransformerTestingDataset(dat_tst.features, mdl.src_modalities)
    ldr = DataLoader(
        dataset=dat,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=transformer_dataset.TransformerTestingDataset.collate_fn
    )

    times_all = discover_times_from_labels(dat_tst)  # e.g., ["time0",...,"time6"]
    # Choose which time indices to attribute (fewer = faster). Example: all 0..6
    target_times = list(range(len(times_all)))       # or [0,3,6]

    ig_accum = {}   # {key -> concatenated tensor over all batches}
    with torch.no_grad():
        pass  # just to emphasize we'll re-enable grad per batch below

    for x_batch, mask_batch in tqdm(ldr, desc="IG"):
        # Send to device
        x_batch = {k: v.to(mdl.device) for k, v in x_batch.items()}
        mask_batch = {k: v.to(mdl.device) for k, v in mask_batch.items()}

        # IMPORTANT: enable grad inside IG function
        ig_dict = compute_ig_batch(
            mdl,
            x_batch,
            mask_batch,
            target_times=target_times,
            class_index=2,   # AD logit
            steps=64,
            use_logit=True
        )

        # concatenate per key over batches
        for k, v in ig_dict.items():
            if k not in ig_accum:
                ig_accum[k] = [v.detach().cpu()]
            else:
                ig_accum[k].append(v.detach().cpu())

    # stack per key
    for k in list(ig_accum.keys()):
        ig_accum[k] = torch.cat(ig_accum[k], dim=0)  # [N]

    # Convert to long + summary and save
    ig_long, ig_summary = ig_to_long_dataframe(ig_accum, dat_file, top_k_regions=30)
    ig_long_path, ig_sum_path = save_ig_outputs(ig_long, ig_summary, save_path, base_name=fname)
    print(f"IG saved:\n  {ig_long_path}\n  {ig_sum_path}")

    return df, met


import copy
import torch
import pandas as pd
import numpy as np
import os

# ---------- utilities to discover times & feature names ----------
def discover_times_from_labels(dat_tst):
    # Reuse your label order to get time keys like "time0", "time1", ...
    time_keys = list(dat_tst.label_modalities)  # e.g., ["time0_DX_binary", ...]
    times = [k.split("_")[0] for k in time_keys]
    return times

def discover_feature_names(src_modalities):
    """
    src_modalities is whatever you used to build the model (mapping of time->features).
    If that's not handy here, we infer from a sample batch dict.
    """
    # Fallback: infer non-Age feature names from a single sample dict shape
    # You can tighten this if you already have a canonical list.
    return None  # we'll infer from a batch below


# ---------- IG over raw scalar inputs (non-Age features) ----------
@torch.no_grad()
def _clone_batch_like(x_batch):
    """Deep(ish) copy of the input batch dict without grad; values detached."""
    out = {}
    for k, v in x_batch.items():
        out[k] = v.detach().clone()
    return out

def _set_requires_grad_on_inputs(x_batch, enable=True, include_key=lambda k: True):
    for k, v in x_batch.items():
        if include_key(k) and v.dtype.is_floating_point:
            v.requires_grad_(enable)

def _zero_baseline_like(x_batch, include_key=lambda k: True):
    base = {}
    for k, v in x_batch.items():
        if include_key(k) and v.dtype.is_floating_point:
            base[k] = torch.zeros_like(v)
        else:
            base[k] = v.detach().clone()
    return base

def _interpolate_batches(x_base, x_batch, alpha, include_key=lambda k: True):
    """
    Returns x_interp = x_base + alpha * (x_batch - x_base) for numeric keys,
    passes other keys through unchanged.
    """
    x_interp = {}
    for k in x_batch:
        xb, x = x_base[k], x_batch[k]
        if include_key(k) and x.dtype.is_floating_point:
            x_interp[k] = xb + alpha * (x - xb)
        else:
            x_interp[k] = x
    return x_interp


def compute_ig_batch(
    model,
    x_batch,
    mask_batch,
    target_times,          # list of int time indices, or "all"
    class_index=2,         # AD class index in logits
    steps=64,
    use_logit=True,
):
    """
    Computes IG attributions wrt raw scalar inputs for a batch.
    - model.net_ is your Transformer (already on device, eval mode).
    - x_batch, mask_batch are dicts from your DataLoader.
    Returns:
      ig_dict: { key -> Tensor[B] } for each input key (feature at a specific time),
               only for included (non-Age) keys; others omitted.
    """
    device = next(model.net_.parameters()).device
    net = model.net_
    net.eval()

    # Which input keys to include (non-age scalars only)
    def include_key(k):
        # include time-stamped scalar features but exclude Age and labels
        # adapt this predicate if your naming differs
        return (k.startswith("time") and ("Age" not in k) and ("DX" not in k))

    # Determine which time indices we care about
    # We use mask_batch to figure out time order; simplest is from label keys
    # But you can also parse from the keys present.
    # We'll map "time{t}_" prefixes to integer t
    def time_index_from_key(k):
        # expects k like "time3_H_MUSE_Volume_31"
        prefix = k.split("_")[0]  # "time3"
        return int(prefix.replace("time", ""))

    # Build a time filter for the keys we include
    if target_times == "all":
        def include_key_and_time(k):
            return include_key(k)
    else:
        target_times_set = set(int(t) for t in target_times)
        def include_key_and_time(k):
            return include_key(k) and (time_index_from_key(k) in target_times_set)

    # 1) Baseline (zeros) and prep
    x_base = _zero_baseline_like(x_batch, include_key=include_key_and_time)

    # We need grads, so turn off global no_grad
    torch.set_grad_enabled(True)

    # We'll accumulate path integral of gradients wrt inputs we care about
    # Initialize accumulators (same shapes as inputs)
    grad_accum = {k: torch.zeros_like(v, dtype=torch.float32, device=v.device)
                  for k, v in x_batch.items() if include_key_and_time(k) and v.dtype.is_floating_point}

    # 2) Line integral with Riemann sum
    for s in range(1, steps + 1):
        alpha = float(s) / steps
        x_interp = _interpolate_batches(x_base, x_batch, alpha, include_key=include_key_and_time)

        # Enable grad on the included inputs
        _set_requires_grad_on_inputs(x_interp, True, include_key=include_key_and_time)

        # Forward
        # Expect your net forward wrapper: model.net_(x, mask, skip_embedding=None, return_out_emb=False)
        out = net(x_interp, mask_batch)  # dict: {"time{t}_DX_binary": [B, 3]}
        # Build scalar target: sum of AD logit at desired times over batch (mask real visits)
        loss = 0.0
        for k_out, logits in out.items():
            # k_out e.g., "time3_DX_binary"
            t_idx = int(k_out.split("_")[0].replace("time", ""))
            if (target_times != "all") and (t_idx not in target_times_set):
                continue
            # logits: [B, num_classes]
            y = logits[..., class_index]  # AD logit
            # mask: use same key as labels
            if k_out in mask_batch:
                m = mask_batch[k_out].to(y.dtype)  # [B], 1 for present, 0 for pad
                if use_logit:
                    loss = loss + (y * m).sum()
                else:
                    # if you prefer probability, apply softmax then sum
                    p = torch.softmax(logits, dim=-1)[..., class_index]
                    loss = loss + (p * m).sum()
            else:
                loss = loss + y.sum()

        # Backprop to inputs
        for p in net.parameters():
            if p.grad is not None:
                p.grad = None
        if isinstance(loss, float):
            loss = torch.tensor(loss, device=device, dtype=torch.float32)
        loss.backward()

        # Accumulate gradients wrt the included inputs
        for k in grad_accum:
            g = x_interp[k].grad
            if g is not None:
                grad_accum[k] += g.detach()

    # 3) Average gradient along path and multiply by (x - x_base) (elementwise)
    ig_dict = {}
    for k in grad_accum:
        avg_grad = grad_accum[k] / steps
        delta = (x_batch[k].detach() - x_base[k].detach())
        ig = delta * avg_grad
        ig_dict[k] = ig.detach()

    # turn grads off again for safety
    torch.set_grad_enabled(False)
    return ig_dict


def ig_to_long_dataframe(ig_dict, test_df, top_k_regions=None):
    """
    Convert {input_key -> [B]} IG dict into long dataframe with PTID, time, region, ig.
    Assumes input_key like "time3_H_MUSE_Volume_31".
    """
    rows = []
    ptids = test_df["PTID"].values
    for k, v in ig_dict.items():
        # v shape: [B] (or [B, ...] if your inputs had extra dims)
        if v.ndim > 1:
            v = v.view(v.shape[0], -1).sum(dim=1)  # collapse feature dims if needed
        t = k.split("_")[0]               # "time3"
        region = "_".join(k.split("_")[1:])  # "H_MUSE_Volume_31"
        vals = v.cpu().numpy()
        for i in range(len(vals)):
            rows.append({"PTID": ptids[i], "time": t, "region": region, "ig": float(vals[i])})
    attn_long = pd.DataFrame(rows)

    if top_k_regions is not None and len(attn_long):
        keep = (attn_long.groupby("region")["ig"].apply(lambda s: s.abs().mean())
                .sort_values(ascending=False).head(top_k_regions).index)
        attn_long = attn_long[attn_long["region"].isin(keep)]

    # summary
    if len(attn_long):
        summary = (attn_long
                   .groupby(["time", "region"])["ig"]
                   .agg(mean="mean", std="std", count="count")
                   .reset_index())
        summary["abs_mean"] = (attn_long.groupby(["time", "region"])["ig"]
                               .apply(lambda s: s.abs().mean())).values
    else:
        summary = pd.DataFrame(columns=["time","region","mean","std","count","abs_mean"])
    return attn_long, summary


def save_ig_outputs(attn_long, summary, save_path, base_name):
    os.makedirs(save_path, exist_ok=True)
    long_path = os.path.join(save_path, f"{base_name}_ig_long.csv")
    sum_path  = os.path.join(save_path, f"{base_name}_ig_summary.csv")
    attn_long.to_csv(long_path, index=False)
    summary.to_csv(sum_path, index=False)
    return long_path, sum_path


if __name__ == '__main__':
    df_pred, met = generate_predictions_for_data_file(dat_file)

    met_df = pd.DataFrame(met)
    met_df.to_csv(save_path + f'{fname}_performance_report.pdf', index=False)
    print(met_df.round(2))

