import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing import Any, Self, Type
from contextlib import suppress
from copy import deepcopy
from scipy.special import softmax
from tqdm import tqdm
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]

from ..nn import transformer
from ..nn import focal_loss
from ..utils import transformer_dataset
from ..utils import misc

class TralzformerModel(BaseEstimator):
    def __init__(self,
                 src_features: dict[str, dict[str, Any]],
                 tgt_labels: dict[str, dict[str, Any]],
                 label_fractions: dict[str, float],
                 d_model: int = 32,
                 nhead: int = 1,
                 num_encoder_layers: int = 1,
                 num_epochs: int = 32,
                 batch_size: int = 8,
                 batch_size_multiplier: int = 1,
                 lr: float = 1e-2,
                 weight_decay: float = 0.0,
                 beta: float = 0.9999,
                 gamma: float = 2.0,
                 criterion: str | None = None,
                 device: str = 'cpu',
                 cuda_devices: list = [1],
                 ckpt_path: str = './ckpt/ckpt.pt',
                 load_from_ckpt: bool = False,
                 save_intermediate_ckpts: bool = False,
                 data_parallel: bool = False,
                 verbose: int = 0,
                 wandb_ = 0,
                 balanced_sampling: bool = False,
                 label_distribution: dict = {},
                 ranking_loss: bool = False,
                 _device_ids: list | None = None,

                 _dataloader_num_workers: int = 4,
                 _amp_enabled: bool = False,

                 fold_index = None
            ) -> None:
        
                self.src_features = src_features
                self.tgt_labels = tgt_labels

                # Training parameters
                self.label_fractions = label_fractions
                self.d_model = d_model
                self.nhead = nhead
                self.num_encoder_layers = num_encoder_layers
                self.num_epochs = num_epochs
                self.batch_size = batch_size
                self.batch_size_multiplier = batch_size_multiplier
                self.lr = lr
                self.weight_decay = weight_decay
                self.beta = beta
                self.gamma = gamma
                self.criterion = criterion
                self.device = device
                self.cuda_devices = cuda_devices
                self.ckpt_path = ckpt_path
                self.load_from_ckpt = load_from_ckpt
                self.save_intermediate_ckpts = save_intermediate_ckpts
                self.data_parallel = data_parallel
                self.verbose = verbose
                self.label_distribution = label_distribution
                self.wandb_ = wandb_
                self.balanced_sampling = balanced_sampling
                self.ranking_loss = ranking_loss
                self._device_ids = _device_ids
                self._dataloader_num_workers = _dataloader_num_workers
                self._amp_enabled = _amp_enabled
                self.fold_index = fold_index
                
    def fit(self, x_trn, x_vld, y_trn, y_vld, train_ids, val_ids) -> Self:
            
        if self.wandb_ == 1:
            run_name = f"Tralzformer_fold_{self.fold_index}" if hasattr(self, "fold_index") else "Tralzformer_run"
            if wandb.run is not None:
                wandb.finish()
            wandb.init(
                entity="georgekanel-national-technical-university-of-athens",
                project="Tralzformer_Final_Experiments",
                group="Exp_07_09_DiagnosisPrediction_7V_ADvsCN",
                name=run_name,

                # track hyperparameters and run metadata
                config={
                    "Loss": 'Focalloss',
                    "EMB": "ALL_SEQ",
                    "epochs": self.num_epochs,
                    "d_model": self.d_model,
                    # 'positional encoding': 'Diff PE',
                    'Balanced Sampling': self.balanced_sampling,
                    'Shared_CNN': 'Yes',
                }
            )
            
            wandb.run.log_code(".")
        else:
            wandb.init(mode="disabled")

        torch.set_num_threads(1)

        print(self.criterion)
        print(f"Ranking Loss: {self.ranking_loss}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batch size multiplier: {self.batch_size_multiplier}")

        self.__init_net()

        self.train_ids = train_ids
        self.val_ids = val_ids

        # Create training and validation data loaders
        trn_loader, vld_loader = self._init_dataloader(x_trn, x_vld, y_trn, y_vld)

        # Initialize the optimizer and the learning rate scheduler
        if not self.load_from_ckpt:
            self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer)

        # Gradient Scaler for Automatic Mixed Precision
        if self._amp_enabled:
                self.scaler = torch.cuda.amp.GradScaler('cuda')

        self.loss_fn = {}

        for k in self.tgt_labels:
            if self.label_fractions[k] >= 0.3:
                alpha = -1
            else:
                alpha = pow((1 - self.label_fractions[k]), 2)

            self.loss_fn[k] = focal_loss.SoftmaxFocalLoss(
                alpha = alpha,
                gamma = self.gamma,
                reduction = 'none'
            )
        
        # To record the best validation performance criterion
        if self.criterion is not None:
            best_crit = None
            best_crit_AUPR = None
        
        # progress bar for epoch loops
        if self.verbose == 1:
            with self._lock if self._lock is not None else suppress():
                pbr_epoch = tqdm.tqdm(
                    desc = 'Rank {:02d}'.format(self._rank),
                    total = self.num_epochs,
                    position = self._rank,
                    ascii = True,
                    leave = False,
                    bar_format = '{l_bar}{r_bar}'
                )

        self.skip_embedding = {}
        for k, info in self.src_features.items():
            self.skip_embedding[k] = False

        self.grad_list = []

        # Patience for early stopping
        patience = 20
        no_improve_epochs = 0

        # Training Loop
        for epoch in range(self.start_epoch, self.num_epochs):
            met_trn = self.train_one_epoch(trn_loader, epoch)
            met_vld = self.validate_one_epoch(vld_loader, epoch)

            print(self.ckpt_path.split('/')[-1])

            # Save the model if it has the best validation performance by far
            if self.criterion is None:
                continue

            # Is current criterion better than previous best?
            curr_crit = np.mean([met_vld[i][self.criterion] for i in range(len(self.tgt_labels))])
            curr_crit_AUPR = np.mean([met_vld[i]["AUC (PR)"] for i in range(len(self.tgt_labels))])

            # AUROC
            if best_crit is None or np.isnan(best_crit):
                is_better = True
            elif self.criterion == 'Loss' and best_crit >= curr_crit:
                is_better = True
            elif self.criterion != 'Loss' and best_crit <= curr_crit:
                is_better = True
            else:
                is_better = False
            
            # AUPR
            if best_crit_AUPR is None or np.isnan(best_crit_AUPR):
                is_better_AUPR = True
            elif best_crit_AUPR <= curr_crit_AUPR:
                is_better_AUPR = True
            else:
                is_better_AUPR = False
            
            # Update best criterion
            if is_better_AUPR:
                best_crit_AUPR = curr_crit_AUPR
                no_improve_epochs = 0
                if self.save_intermediate_ckpts:
                    print(f"Saving the model to {self.ckpt_path[:-3]}_AUPR.pt")
                    self.save(self.ckpt_path[:-3]+"_AUPR.pt", epoch)
                else:
                    no_improve_epochs += 1
                
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch} after {patience} epochs without improvement.")
                    break
                    
                if is_better:
                    best_crit = curr_crit
                    best_state_dict = deepcopy(self.net_.state_dict())
                    if self.save_intermediate_ckpts:
                        print(f"Saving the model for fold to {self.ckpt_path}...")
                        self.save(self.ckpt_path, epoch)
                
                if self.verbose > 2:
                    print('Best {}: {}'.format(self.criterion, best_crit))
                    print('Best {}: {}'.format('AUC (PR)', best_crit_AUPR))

                if self.verbose == 1:
                    with self._lock if self._lock is not None else suppress():
                        pbr_epoch.update(1)
                        pbr_epoch.refresh()
                
        if self.verbose == 1:
            with self._lock if self._lock is not None else suppress():
                pbr_epoch.close()

        wandb.finish()

        return self
            
    def train_one_epoch(self, ldr_trn, epoch):
        
        # Progress bar for batch loops
        if self.verbose > 1:
            pbr_batch = misc.ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch))

        # Set model to train mode
        torch.set_grad_enabled(True)
        self.net_.train()

        scores_trn, y_true_trn, y_mask_trn = [], [], []
        losses_trn = [[] for _ in self.tgt_labels]
        iters = len(ldr_trn)
        for n_iter, (x_batch, y_batch, mask, y_mask) in enumerate(ldr_trn):

            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}

            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled,
            ):

                outputs = self.net_(x_batch, mask)

                # calculate multitask loss
                loss = 0

                for i, k in enumerate(self.tgt_labels):
                    loss_task = self.loss_fn[k](outputs[k], y_batch[k])
                    # Compute valid mask again (same logic as inside your FocalLoss)
                    target = y_batch[k].view(-1)
                    C = outputs[k].shape[-1]
                    valid_mask = (target >= 0) & (target < C) & (target != 1)
                    y_mask_valid = y_mask[k][valid_mask]  # match shape to loss_task

                    msk_loss_task = loss_task * y_mask_valid
                    msk_loss_mean = msk_loss_task.sum() / y_mask_valid.sum()
                    loss += msk_loss_mean
                    losses_trn[i] += msk_loss_task.detach().cpu().numpy().tolist()


            # backward
            loss = loss / self.batch_size_multiplier
            if self._amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if len(self.grad_list) > 0:
                print(len(self.grad_list), len(self.grad_list[-1]))
                print(f"Gradient at {n_iter}: {self.grad_list[-1][0]}")

            # update parameters
            if n_iter != 0 and n_iter % self.batch_size_multiplier == 0:
                if self._amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # set self.scheduler
                self.scheduler.step(epoch + n_iter / iters)

            sorted_keys = sorted(outputs.keys(), key=lambda x: int(x.split('_')[0].replace('time','')))
            outputs = torch.stack([outputs[k] for k in sorted_keys], dim=1)
            y_batch = torch.stack([y_batch[k] for k in sorted_keys], dim=1)
            y_mask = torch.stack([y_mask[k] for k in sorted_keys], dim=1)

            # save outputs to evaluate performance later
            scores_trn.append(outputs.detach().to(torch.float).cpu())
            y_true_trn.append(y_batch.cpu())
            y_mask_trn.append(y_mask.cpu())

            # update progress bar
            if self.verbose > 1:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size, {})
                pbr_batch.refresh()

            # clear cuda cache
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1:
            pbr_batch.close()

        # calculate and print training performance metrics
        scores_trn = torch.cat(scores_trn)
        y_true_trn = torch.cat(y_true_trn)
        y_mask_trn = torch.cat(y_mask_trn)

        # AD vs CN: exclude MCI by masking it out
        y_true_trn = y_true_trn.squeeze(-1)
        adcn_mask = ((y_true_trn == 0) | (y_true_trn == 2))
        y_mask_adcn = (y_mask_trn.bool() & adcn_mask).to(y_mask_trn.dtype)

        binary_y_true = (y_true_trn == 2).int()
        
        y_prob_trn_cls = torch.softmax(scores_trn, dim=-1)[..., 2]  # class 2 prob only
        thr = 0.5                                       # or a per-time tensor of thresholds
        binary_y_pred_trn = (y_prob_trn_cls > thr).int()            # [N, 9]

        met_trn = misc.get_metrics_multitask(
        binary_y_true.numpy(),
        binary_y_pred_trn.numpy(),
        y_prob_trn_cls.numpy(),
        y_mask_adcn.numpy()
        )

        # add loss to metrics
        for i in range(len(self.tgt_labels)):
            met_trn[i]['Loss'] = np.mean(losses_trn[i])

        # log metrics to wandb
        wandb.log({f"Train loss {list(self.tgt_labels)[i]}": met_trn[i]['Loss']  for i in range(len(self.tgt_labels))}, step=epoch)
        wandb.log({f"Train Balanced Accuracy {list(self.tgt_labels)[i]}": met_trn[i]['Balanced Accuracy']  for i in range(len(self.tgt_labels))}, step=epoch)
        
        wandb.log({f"Train AUC (ROC) {list(self.tgt_labels)[i]}": met_trn[i]['AUC (ROC)']  for i in range(len(self.tgt_labels))}, step=epoch)
        wandb.log({f"Train AUPR {list(self.tgt_labels)[i]}": met_trn[i]['AUC (PR)']  for i in range(len(self.tgt_labels))}, step=epoch)

        if self.verbose > 2:
            misc.print_metrics_multitask(met_trn)

        return met_trn
    

    def validate_one_epoch(self, vld_loader, epoch):
        # # progress bar for validation
        if self.verbose > 1:
            pbr_batch = misc.ProgressBar(len(vld_loader.dataset), 'Epoch {:03d} (VLD)'.format(epoch))

        # set model to validation mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        scores_vld, y_true_vld, y_mask_vld = [], [], []
        losses_vld = [[] for _ in self.tgt_labels]
        for x_batch, y_batch, mask, y_mask in vld_loader:
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}

            # forward
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled
            ):
                
                outputs = self.net_(x_batch, mask)

                # calculate multitask loss
                for i, k in enumerate(self.tgt_labels):
                    loss_task = self.loss_fn[k](outputs[k], y_batch[k])
                    target = y_batch[k].view(-1)
                    C = outputs[k].shape[-1]
                    valid_mask = (target >= 0) & (target < C) & (target != 1)
                    y_mask_valid = y_mask[k][valid_mask]  # match shape to loss_task

                    msk_loss_task = loss_task * y_mask_valid
                    losses_vld[i] += msk_loss_task.detach().cpu().numpy().tolist()

            sorted_keys = sorted(outputs.keys(), key=lambda x: int(x.split('_')[0].replace('time','')))
            outputs = torch.stack([outputs[k] for k in sorted_keys], dim=1)
            y_batch = torch.stack([y_batch[k] for k in sorted_keys], dim=1)
            y_mask = torch.stack([y_mask[k] for k in sorted_keys], dim=1)

            # save outputs to evaluate performance later
            scores_vld.append(outputs.detach().to(torch.float).cpu())
            y_true_vld.append(y_batch.cpu())
            y_mask_vld.append(y_mask.cpu())

            # update progress bar
            if self.verbose > 1:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size, {})
                pbr_batch.refresh()

            # clear cuda cache
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            
        # for better tqdm progress bar display
        if self.verbose > 1: 
            pbr_batch.close()

        # calculate and print validation performance metrics 
        scores_vld = torch.cat(scores_vld) # [N, 9, 3]
        y_true_vld = torch.cat(y_true_vld) # [N, 9, 1] 
        y_mask_vld = torch.cat(y_mask_vld) # [N, 9]

        # AD vs CN
        y_true_vld = y_true_vld.squeeze(-1)
        adcn_mask = ((y_true_vld == 0) | (y_true_vld == 2))
        y_mask_adcn = (y_mask_vld.bool() & adcn_mask).to(y_mask_vld.dtype)
        binary_y_true_vld = (y_true_vld == 2).int()

        y_prob_vld_cls = torch.softmax(scores_vld, dim=-1)[..., 2]  # class 2 prob only
        thr = 0.5                                       # or a per-time tensor of thresholds
        binary_y_pred_vld = (y_prob_vld_cls > thr).int()            # [N, 9]   

        met_vld = misc.get_metrics_multitask(
            binary_y_true_vld.numpy(),
            binary_y_pred_vld.numpy(),
            y_prob_vld_cls.numpy(),
            y_mask_adcn.numpy()
        )

        # add loss to metrics
        for i in range(len(self.tgt_labels)):
            met_vld[i]['Loss'] = np.mean(losses_vld[i])

        wandb.log({f"Validation loss {list(self.tgt_labels)[i]}": met_vld[i]['Loss'] for i in range(len(self.tgt_labels))}, step=epoch)
        wandb.log({f"Validation Balanced Accuracy {list(self.tgt_labels)[i]}": met_vld[i]['Balanced Accuracy']  for i in range(len(self.tgt_labels))}, step=epoch)
        
        wandb.log({f"Validation AUC (ROC) {list(self.tgt_labels)[i]}": met_vld[i]['AUC (ROC)']  for i in range(len(self.tgt_labels))}, step=epoch)
        wandb.log({f"Validation AUPR {list(self.tgt_labels)[i]}": met_vld[i]['AUC (PR)']  for i in range(len(self.tgt_labels))}, step=epoch)

        if self.verbose > 2:
            misc.print_metrics_multitask(met_vld)

        return met_vld
    

    def predict_logits(self,
        x: list[dict[str, Any]],
        _batch_size: int | None = None,
        skip_embedding: dict | None = None
    ) -> list[dict[str, float]]:
        '''
        The input x can be a single sample or a list of samples.
        '''
        # input validation
        check_is_fitted(self)
        print(self.device)

        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # initialize dataset and dataloader object
        dat = transformer_dataset.TransformerTestingDataset(x, self.src_modalities)
        ldr = DataLoader(
            dataset = dat, 
            batch_size = _batch_size if _batch_size is not None else len(x),
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = transformer_dataset.TransformerTestingDataset.collate_fn
        )

        # run model and collect results
        logits: list[dict[str, float]] = []
        # embeddings: list[dict[str, float]] = []
        for x_batch, mask in tqdm(ldr):
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
            mask = {k: mask[k].to(self.device) for k in mask}

            # forward
            output = self.net_(x_batch, mask)

            # convert output from dict-of-list to list of dict, then append
            tmp = {k: output[k].tolist() for k in self.tgt_labels}
            tmp = [{k: tmp[k][i] for k in self.tgt_labels} for i in range(len(next(iter(tmp.values()))))]
            logits += tmp
        
        return logits
    
    def predict_proba(self,
        x: list[dict[str, Any]],
        skip_embedding: dict | None = None,
        temperature: float = 1.0,
        _batch_size: int | None = None
    ) -> list[dict[str, float]]:
        ''' ... '''
        logits = self.predict_logits(x=x, _batch_size=_batch_size, skip_embedding=skip_embedding)
        print("got logits")

        proba = []
        for smp in logits:  # for each sample
            smp_proba = {}
            for t, vals in smp.items():  # for each time step
                z = np.asarray(vals, dtype=float) / temperature # shape (3,)
                p = softmax(z)
                smp_proba[t] = p.tolist()
                #smp_proba[t] = [softmax(v / temperature) for v in vals]
            proba.append(smp_proba)

        return logits, proba
        # return logits, [{k: expit(smp[k] / temperature) for k in self.tgt_labels} for smp in logits], embeddings
    

    def predict(self,
        x: list[dict[str, Any]],
        skip_embedding: dict | None = None,
        fpr: dict[str, Any] | None = None,
        tpr: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        _batch_size: int | None = None
    ) -> list[dict[str, int]]:
        ''' ... '''
        if fpr is None or tpr is None or thresholds is None:
            logits, proba = self.predict_proba(x, _batch_size=_batch_size, skip_embedding=skip_embedding)
            print("got proba")

            # One-vs-Rest (AD class = 2)
            pred = []
            proba_pos = []
            
            for smp in proba:  # over samples
                smp_pred = {}
                smp_prob = {}

                for t, vals in smp.items():  # over time steps
                    smp_pred[t] = int(vals[2] > 0.5)  # binary OVR for class 2
                    smp_prob[t] = float(vals[2])

                pred.append(smp_pred)
                proba_pos.append(smp_prob)

            logits_pos = []

            for smp_logit in logits:
                smp_logit_pos = {}

                for t in smp_prob.keys():
                    smp_logit_pos[t] = float(smp_logit[t][2])
                
                logits_pos.append(smp_logit_pos)

            return logits_pos, proba_pos, pred
        else:
            logits, proba = self.predict_proba(x, _batch_size=_batch_size, skip_embedding=skip_embedding)
            print("got proba")
            youden_index = {}
            thr = {}
            for i, k in enumerate(self.tgt_labels):
                youden_index[k] = tpr[i] - fpr[i]
                thr[k] = thresholds[i][np.argmax(youden_index[k])]
            #     print(thr[k])
            # print(thr)
            
            return logits, proba, [{k: int(smp[k] > thr[k]) for k in self.tgt_labels} for smp in proba]
        
    def save(self, filepath: str, epoch: int) -> None:
        """Save the model to the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param epoch: _description_
        :type epoch: int
        """        
        check_is_fitted(self)
        if self.data_parallel:
            state_dict = self.net_.module.state_dict()
        else:
            state_dict = self.net_.state_dict()

        # attach model hyper parameters
        state_dict['src_features'] = self.src_features
        state_dict['tgt_labels'] = self.tgt_labels
        state_dict['d_model'] = self.d_model
        state_dict['nhead'] = self.nhead
        state_dict['num_encoder_layers'] = self.num_encoder_layers
        #state_dict['num_decoder_layers'] = self.num_decoder_layers
        state_dict['optimizer'] = self.optimizer
        state_dict['epoch'] = epoch

        if self.label_distribution:
            state_dict['label_distribution'] = self.label_distribution

        torch.save(state_dict, filepath)

    def load(self, filepath: str, map_location: str = 'cpu', img_dict=None) -> None:
        """Load a model from the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param map_location: _description_, defaults to 'cpu'
        :type map_location: str, optional
        :param img_dict: _description_, defaults to None
        :type img_dict: _type_, optional
        """        
        # load state_dict
        state_dict = torch.load(filepath, map_location=map_location, weights_only=False)

        # load data modalities
        self.src_modalities: dict[str, dict[str, Any]] = state_dict.pop('src_features')
        self.tgt_labels: dict[str, dict[str, Any]] = state_dict.pop('tgt_labels')
        if 'label_distribution' in state_dict:
            self.label_distribution: dict[str, dict[int, int]] = state_dict.pop('label_distribution')
        if 'optimizer' in state_dict:
            self.optimizer = state_dict.pop('optimizer')

        # initialize model
        self.d_model = state_dict.pop('d_model')
        self.nhead = state_dict.pop('nhead')
        self.num_encoder_layers = state_dict.pop('num_encoder_layers')
        #self.num_decoder_layers = state_dict.pop('num_decoder_layers')
        if 'epoch' in state_dict.keys():
            self.start_epoch = state_dict.pop('epoch')

        self.net_ = transformer.Transformer(self.src_modalities, self.tgt_labels, self.d_model, self.nhead, 0.3, self.num_encoder_layers, 1, 17, self.device, self.cuda_devices)
       
        if 'scaler' in state_dict and state_dict['scaler']:
            self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.net_.load_state_dict(state_dict)
        check_is_fitted(self)
        self.net_.to(self.device)

    def to(self, device: str) -> Self:
        """Mount the model to the given device. 

        :param device: _description_
        :type device: str
        :return: _description_
        :rtype: Self
        """        
        self.device = device
        if hasattr(self, 'model'): self.net_ = self.net_.to(device)
        return self
    
    @classmethod
    def from_ckpt(cls, filepath: str, device='cpu') -> Self:
        """Create a new ADRD model and load parameters from the checkpoint. 

        This is an alternative constructor.

        :param filepath: _description_
        :type filepath: str
        :param device: _description_, defaults to 'cpu'
        :type device: str, optional
        :return: _description_
        :rtype: Self
        """ 
        obj = cls(None, None, None,device=device)
        if device == 'cuda':
            obj.device = "{}:{}".format(obj.device, str(obj.cuda_devices[1]))
        print(obj.device)
        obj.load(filepath, map_location=obj.device)
        return obj
        

    def __init_net(self):
            
        if self.device == 'cuda':
            self.device = "{}:{}".format(self.device, str(self.cuda_devices[1]))
        print("Device: " + self.device)

        self.start_epoch = 0
        if self.load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                #self.load(self.ckpt_path, map_location=self.device)
            except:
                print("Cannot load form checkpoint. Initializing new model...")
                self.load_from_ckpt = False
        
        if not self.load_from_ckpt:
            self.net_ = transformer.Transformer(
                 src_features = self.src_features,
                 tgt_labels = self.tgt_labels,
                 d_model = self.d_model,
                 nhead = self.nhead,
                 num_encoder_layers = self.num_encoder_layers,
                 device = self.device,
                 cuda_devices = self.cuda_devices
            )

            for name, p in self.net_.named_parameters():
                 if p.dim() > 1:
                      torch.nn.init.xavier_uniform(p)
        
        # import gc

        # torch.cuda.empty_cache()
        # gc.collect()
        # print(torch.cuda.memory_summary())

        self.net_.to(self.device)

        if self.data_parallel and torch.cuda.device_count() > 1:
             print("Available", torch.cuda.device_count(), "GPUs!")
             print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

             self.net_ = torch.nn.DataParallel(self.net_, device_ids=[self.cuda_devices[1]])

    def _init_dataloader(self, x_trn, x_vld, y_trn, y_vld):
         
        # initialize dataset and dataloader
        if self.balanced_sampling:
            dat_trn = transformer_dataset.Transformer2ndOrderBalancedTrainingDataset(
                x_trn, y_trn,
                self.src_features,
                self.tgt_labels,
                dropout_rate = .5,
                dropout_strategy = 'permutation'
            )
        else:
            dat_trn = transformer_dataset.TransformerTrainingDataset(
                x_trn, y_trn,
                self.src_features,
                self.tgt_labels,
                dropout_rate = .5,
                dropout_strategy = 'permutation'
            )

        dat_vld = transformer_dataset.TransformerValidationDataset(
            x_vld, y_vld, 
            self.src_features,
            self.tgt_labels
        )

        trn_loader = DataLoader(
            dataset = dat_trn, 
            batch_size = self.batch_size,
            shuffle = False, # default value: True
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = transformer_dataset.TransformerTrainingDataset.collate_fn
        )

        vld_loader = DataLoader(
            dataset = dat_vld,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = transformer_dataset.TransformerValidationDataset.collate_fn
        )

        return trn_loader, vld_loader
    

    def _init_optimizer(self):
        """ ... """
        params = list(self.net_.parameters())
        return torch.optim.AdamW(
            params,
            lr = self.lr,
            betas = (0.9, 0.98),
            weight_decay = self.weight_decay
        )
    
    def _init_scheduler(self, optimizer):
        """ ... """       

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=64,
            T_mult=2,
            eta_min = 0,
            verbose=(self.verbose > 2)
        )
    
    def _proc_fit(self):
        """ ... """ 