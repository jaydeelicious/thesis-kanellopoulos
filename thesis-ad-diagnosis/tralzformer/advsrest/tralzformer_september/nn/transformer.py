import torch 
import numpy as np
from typing import Any, Type
import math
Tensor = Type[torch.Tensor]
from icecream import ic
ic.disable()
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe.shape ==> (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args: 
            x: Tensor of shape (S, N, D)
            S: sequence length
            N: batch size
            D: embedding size (d_model)

        Returns:
            x + positional encoding: shape (S, N, D)
        """
        
        seq_len = x.size(0)
        return x + self.pe[:seq_len]
    
class Transformer(torch.nn.Module):
    def __init__(
            self, 
            src_features: dict[str, dict[str, Any]],
            tgt_labels: dict[str, dict[str, Any]],
            d_model: int = 128,
            nhead: int = 4,
            dropout: float = 0.3,
            num_encoder_layers: int = 2,
            num_decoder_layers: int = 1,
            max_time_steps: int = 17,
            device: str = 'cpu',
            cuda_devices: list[int] | None = None,
    ) -> None:
            super().__init__()

            if cuda_devices is None:
                cuda_devices = []
            self.cuda_devices = cuda_devices
            
            self.src_features = src_features
            self.tgt_labels = tgt_labels
            self.d_model = d_model
            self.nhead = nhead
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers    
            self.max_time_steps = max_time_steps
            self.device = device

            # Embedding modules per source feature 
            self.modules_emb_src = torch.nn.ModuleDict()
            for name, info in src_features.items():
                if info['type'] == 'categorical':
                    emb = torch.nn.Embedding(info['num_categories'], d_model)
                    self.modules_emb_src[name] = emb
                elif info['type'] == 'numerical':
                    self.modules_emb_src[name] = torch.nn.Sequential(
                         torch.nn.BatchNorm1d(info['shape'][0]),
                         torch.nn.Linear(info['shape'][0], d_model)
                    )
                else:
                    raise ValueError(f"Unsupported src ")
                
            self.modality_type_emb = torch.nn.ModuleDict({
                'Age': torch.nn.Embedding(1, self.d_model),
                'Volume': torch.nn.Embedding(1, self.d_model),
            })
            
            self.w_age = torch.nn.Parameter(torch.tensor(0.5))

            # Positional Encoding for sequences up to max_time_steps
            self.pe = PositionalEncoding(d_model, max_len=max_time_steps)

            # Encoder layers
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=False,
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            # Prediction heads for each target
            # Each head outputs per timestep prediction: input shape (batch, time, d_model) -> output (batch, time, classes)
            self.target_preds = torch.nn.ModuleDict()
            for k, info in tgt_labels.items():
                if info['type'] == 'categorical' and info['num_categories'] == 3:
                    self.target_preds[k] = torch.nn.Linear(d_model, 3)
                else:
                    raise ValueError(f"Unsupported target type or class count for {k}")
                
            self.emb_aux = torch.nn.Parameter(torch.randn(len(self.target_preds), 1, d_model))

    def forward(self,
            x: dict[str, Tensor],
            mask: dict[str, Tensor],
            return_out_emb: bool = False
        ) -> dict[str, Tensor] | tuple[torch.Tensor, dict[str, Tensor]]:

            out_emb = self.forward_emb(x)

            out_trf = self.forward_trf(out_emb, mask)

            # Transpose to [B, T, D]
            out_cls_vec = out_trf.permute(1, 0, 2)

            # Pass each target vector to its classifier
            out_cls = {}

            for i, k in enumerate(self.target_preds.keys()):
                # Parse the time index from the key
                time_idx = int(k.split('_')[0].replace('time', ''))

                # Extract the time-specific latent vector: [B, H] from [B, T, H]
                time_feat = out_cls_vec[:, time_idx, :]
                
                # Pass it through the corresponding classifier head
                out_cls[k] = self.target_preds[k](time_feat)  # [B, C]

            return (out_trf, out_cls) if return_out_emb else out_cls
    
    def forward_emb(self,
    x: dict[str, Tensor]
    ) -> Tensor:

        out_emb = dict()  # time_idx -> {feature_name: emb}

        for k in self.modules_emb_src.keys():
            emb = self.modules_emb_src[k](x[k])  # [B, D]

            # Infer time step (e.g., from 'time0_H_MUSE_Volume_3')
            if k.startswith("time"):
                time_part = k.split("_")[0]  # 'time0'
                time_idx = int(time_part.replace("time", ""))
                feature_name = "_".join(k.split("_")[1:])  # 'H_MUSE_Volume_3'
            else:
                time_idx = 0
                feature_name = k

            # Add modality type embedding
            if "Age" in k:
                mod_type = "Age"
            elif "Volume" in k:
                mod_type = "Volume"
            else:
                mod_type = "Other"
            if mod_type in self.modality_type_emb:
                emb += self.modality_type_emb[mod_type](torch.zeros(1, dtype=torch.long, device=emb.device))

            if time_idx not in out_emb:
                out_emb[time_idx] = dict()
            out_emb[time_idx][feature_name] = emb

        return out_emb  # dict[int, dict[str, Tensor[B, D]]]

    def forward_trf(self, out_emb, mask):

        # stack only non-Age features to (B, T, F, D)
        emb_tensor = torch.stack([
            torch.stack([out_emb[t][f] for f in sorted(out_emb[t].keys()) if "Age" not in f], dim=1)
            for t in sorted(out_emb.keys())
        ], dim=1)  # [B, T, F, D] without Age
 
        # combine features per time step (e.g., by averaging or a small linear layer)
        # This creates a single vector for each time step
        combined_emb = emb_tensor.mean(dim=2) # [B, T, D]

        ts = sorted(out_emb.keys())
        
        # Add Age per time step (robust if missing)
        def get_age_emb(t):
            return next((v for k, v in out_emb[t].items() if "Age" in k),
                        torch.zeros_like(next(iter(out_emb[t].values()))))
        age_stack = torch.stack([get_age_emb(t) for t in ts], dim=1)  # [B, T, D]

        combined_emb = combined_emb + self.w_age * age_stack                  # [B, T, D]

        # Reshape for Transformer and apply Positional Encoding
        # Transformer expects [T, B, D]
        combined_emb = combined_emb.permute(1, 0, 2) # [T, B, D]
        combined_emb = self.pe(combined_emb)

        # Build masks
        # time_mask from dict: assume 1/True = real visit, 0/False = PAD
        
        time_mask = torch.stack([mask[f'time{t}_H_MUSE_Volume_4'] for t in ts], dim=1)  # [B, T]
        time_present = time_mask.to(dtype=torch.bool)
        src_key_padding_mask = (~time_present).to(dtype=torch.bool, device=combined_emb.device)  # True=PAD

        # causal attn mask: [T, T], blocks attending to future
        T = combined_emb.size(0)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=combined_emb.device),
            diagonal=1
        )  # [T, T], True above diagonal = block future


        # Feed to the Transformer
        # Uncomment for causality!
        out_trf = self.transformer_encoder(
            src=combined_emb, # [T, B, D]
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask, # [B, T]
            is_causal=True
        )

        # Not Causal
        # out_trf = self.transformer_encoder(
        #     src=combined_emb, # [T, B, D]
        #     src_key_padding_mask=src_key_padding_mask, # [B, T]
        # )
        return out_trf # [T, B, D]
    