import pandas as pd
import numpy as np
import toml
import argparse
import os
from icecream import ic, install
from sklearn.model_selection import StratifiedKFold

from ..dataloader import dataset_csv
from . import tralzformer

install()
ic.configureOutput(includeContext=True)
ic.disable()

def parser():
    parser = argparse.ArgumentParser("Transformer pipeline", add_help=False)

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--test_path', default='', type=str)
    parser.add_argument('--cnf_file', default='', type=str)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--load_from_ckpt', action="store_true", help="Set to True to load model from checkpoints.")
    parser.add_argument('--save_intermediate-ckpts', action="store_true")
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--balanced_sampling', action="store_true")
    parser.add_argument('--ranking_loss', action="store_true")
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--modality', default='', type=str)
    parser.add_argument('--d_model', default=64, type=int)
    parser.add_argument('--nhead', default=2, type=int)
    parser.add_argument('--num_epochs', default=256, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=2, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    

    args = parser.parse_args()
    return args

args = parser()

seed = 0
print("Loading training/validation dataset...")
data_trnvld = dataset_csv.CSVDataset(data_file=args.data_path, cnf_file=args.cnf_file, mode=0)
print("Done.\nLoading testing dataset...")
data_tst = dataset_csv.CSVDataset(data_file=args.test_path, cnf_file=args.cnf_file, mode=2)
print("Done.")

label_fractions = data_trnvld.label_fractions

train_df = pd.read_csv(args.data_path, sep=';')

label_distribution = {}

cnf = toml.load(args.cnf_file)
for label in list(cnf['label'].keys()):
    label_distribution[label] = dict(train_df[label].value_counts())
ckpt_path = args.ckpt_path

print(label_fractions)
print(label_distribution)

labels = data_trnvld.labels

#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=False)


# Get all label keys matching the pattern (e.g. timeX_DX_next)
label_keys = [key for key in data_trnvld.labels[0] if "DX_next" in key]

# Ensure correct temporal order
label_keys = sorted(label_keys, key=lambda x: int(x.split("_")[0][4:]))

# Build an array: shape = (num_patients, num_time_steps)
labels_array = np.array([
    [patient_labels[k] for k in label_keys]
    for patient_labels in data_trnvld.labels
])

labels_for_strat = labels_array.max(axis=1).astype(int)

for fold, (train_idx, val_idx) in enumerate(skf.split(data_trnvld.features, labels_for_strat)):
    print(f"\n Fold {fold + 1}/5")

    train_features = [data_trnvld.features[i] for i in train_idx]
    train_labels = [data_trnvld.labels[i] for i in train_idx]
    train_ids = [data_trnvld.ids[i] for i in train_idx]

    val_features = [data_trnvld.features[i] for i in val_idx]
    val_labels = [data_trnvld.labels[i] for i in val_idx]
    val_ids = [data_trnvld.ids[i] for i in val_idx]

    base, ext = os.path.splitext(ckpt_path)
    fold_ckpt_path = f"{base}_fold_{fold+1}{ext}"

    # initialize and save transformer
    model = tralzformer.TralzformerModel(
        src_features = data_trnvld.feature_modalities,
        tgt_labels = data_trnvld.label_modalities,
        label_fractions = label_fractions,
        d_model = args.d_model,
        nhead = args.nhead, 
        num_encoder_layers = 1,
        num_epochs = args.num_epochs, 
        batch_size = args.batch_size,
        batch_size_multiplier = 1,
        lr = args.lr,
        weight_decay = args.weight_decay,
        gamma = args.gamma,
        criterion = 'AUC (ROC)',
        device = 'cuda',
        cuda_devices = [0,1],
        ckpt_path = fold_ckpt_path,
        load_from_ckpt = args.load_from_ckpt,
        save_intermediate_ckpts = args.save_intermediate_ckpts,
        data_parallel = True,
        verbose = 4,
        wandb_ = args.wandb, 
        label_distribution = label_distribution,
        ranking_loss = args.ranking_loss,
        _amp_enabled = False,
        _dataloader_num_workers = 4,
        fold_index = fold + 1
    )

    model.fit(train_features, val_features, train_labels, val_labels, train_ids, val_ids)