"""
Load and inspect a PyTorch Lightning checkpoint with OmegaConf compatibility.

This script safely loads a PyTorch Lightning checkpoint file (.ckpt) that includes
OmegaConf-based configuration objects (DictConfig, ListConfig). It prints out the
top-level keys in the checkpoint, displays stored hyperparameters, and lists a
sample of parameter names from the model's state dictionary.

Usage:
    - Set `ckpt_path` to the path of your checkpoint file.
    - The script will print:
        • All top-level keys in the checkpoint.
        • The hyperparameter dictionary (if available).
        • The number of tensors and first 20 parameter names in `state_dict`.

Notes:
    - Uses `torch.serialization.add_safe_globals()` to register OmegaConf types,
      ensuring compatibility with Lightning's checkpoint serialization.
    - Uses `weights_only=False` to fully load metadata, scheduler states, etc.
    - If no `state_dict` is present, a warning is printed.

Example:
    $ python inspect_ckpt.py
"""

import torch
import omegaconf
import pprint

ckpt_path = "logs/pretrain/runs/2025-09-26_21-15-38/checkpoints/last.ckpt"

#  Allow OmegaConf globals (required for Lightning checkpoints)
torch.serialization.add_safe_globals(
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ]
)

#  Force unsafe load explicitly (trusted file, your own)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("\n=== TOP-LEVEL KEYS ===")
print(list(ckpt.keys()))

print("\n=== HYPERPARAMETERS ===")
pprint.pprint(ckpt.get("hyper_parameters", {}))

if "state_dict" in ckpt:
    print(f"\n=== STATE_DICT: {len(ckpt['state_dict'])} tensors ===")
    for k in list(ckpt["state_dict"].keys())[:20]:
        print(k)
else:
    print("⚠️ No state_dict found — this is not a model checkpoint.")
