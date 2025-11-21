"""
Hydra-based Embedding Extractor (Direct JSON)
=============================================
This version skips the HDF5/DataModule pipeline and directly
builds graph features from JSON discussion trees for embedding extraction.

Usage:
------
python src/test_embedding.py \
  --json src/test_data.json \
  --ckpt src/logs/pretrain/runs/2025-10-23_22-41-28/checkpoints/last.ckpt \
  --device cuda \

"""

# the batch size command is removed as it actually valids the parameters,
# need to see why

import os
import sys
import json
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
from data.types import GraphFeatures, TextFeatures, ImageFeatures
from data import collator_utils
from tasks import dataset_utils as dut

# ---- Pathing ----
import rootutils 
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 


#  Convert a tree jsn to feature dicts, the graph
#  watch: Similar to data.dataset_utils.tree_to_graph_features but simplified
def tree_to_graph_features(tree, text_tokenizer, image_processor, root_dir="."):
    """Flatten a discussion tree into tensors for model input."""
    dut.compute_relative_distance(tree)

    flat = {
        "id": [],
        "parent_id": [],
        "text": [],
        "images": [],
        "rotary_position": [],
    }

    def traverse(node, parent_id=None):
        if parent_id is None:
            parent_id = node["id"]
        flat["id"].append(node["id"])
        flat["parent_id"].append(parent_id)
        flat["text"].append(dut.clean_text(node.get("body", "")))
        flat["images"].append(
            node.get("images", [None])[0] if node.get("images") else None
        )
        flat["rotary_position"].append(node.get("rotary_position", [0, 0]))
        for c in node.get("tree", []):
            traverse(c, node["id"])

    traverse(tree)

    # --- Build graph tensors ---
    n = len(flat["id"])
    id_map = {nid: i for i, nid in enumerate(flat["id"])}
    edges = torch.tensor(
        [list(id_map.values()), list(id_map.values())], dtype=torch.long
    )
    # in_degree = torch.zeros(n) I just set both to
    # long tensors but this needs to be verified with liam
    # in later meetings.
    # out_degree = torch.zeros(n)
    in_degree = torch.zeros(n, dtype=torch.long)
    out_degree = torch.zeros(n, dtype=torch.long)
    image_mask = torch.tensor(
        [img is not None for img in flat["images"]], dtype=torch.bool
    )

    graph_features = {
        "edge_index": edges,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "attn_bias": torch.zeros((n, n)),
        "distance": torch.zeros((n, n, 2)),
        "distance_index": torch.zeros((n, n), dtype=torch.int16),
        "image_mask": image_mask,
        "rotary_position": torch.tensor(
            flat["rotary_position"], dtype=torch.float32
        ),
    }

    # --- Text ---
    tokenized_text = text_tokenizer(
        flat["text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=256,
    )
    text_features = {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "token_type_ids": tokenized_text.get(
            "token_type_ids", torch.zeros_like(tokenized_text["input_ids"])
        ),
    }

    # --- Images ---
    valid_imgs = []
    for img in flat["images"]:
        if img:
            try:
                valid_imgs.append(
                    Image.open(os.path.join(root_dir, img)).convert("RGB")
                )
            except Exception:
                valid_imgs.append(None)
        else:
            valid_imgs.append(None)

    tokenized_images = (
        image_processor([img for img in valid_imgs if img], return_tensors="pt")
        if any(valid_imgs)
        else None
    )
    image_features = {
        "pixel_values": (
            tokenized_images["pixel_values"]
            if tokenized_images
            else torch.zeros((0, 3, 224, 224))
        )
    }

    return graph_features, text_features, image_features


# ---- Embedding Generator ----
class EmbeddingGenerator:

    def _to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_device(v) for v in obj]
        else:
            return obj

    def __init__(self, ckpt_path, device="cuda"):
        self.ckpt_path = ckpt_path
        self.device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = None

    def load_model(self):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(self.ckpt_path)),
            ".hydra",
            "config.yaml",
        )
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
        model = instantiate(cfg.model)
        print(f"[Info] Model instantiated: {model.__class__.__name__}")

        ckpt = torch.load(
            self.ckpt_path, map_location="cpu", weights_only=False
        )
        state = ckpt.get("state_dict", ckpt)
        new_state = {
            (
                k[len("model.encoder.") :]
                if k.startswith("model.encoder.")
                else k
            ): v
            for k, v in state.items()
        }
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(
            f"[Info] Checkpoint loaded (missing={len(missing)},"
            f" unexpected={len(unexpected)})"
        )

        self.model = model.to(self.device).eval()
        return self.model

    def generate(self, json_path):
        model = self.load_model()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        with open(json_path, "r") as f:
            samples = json.load(f)

        if isinstance(samples, dict):
            samples = [samples]

        embeddings = []
        with torch.no_grad():
            for tree in samples:
                # g, t, i = tree_to_graph_features(tree,
                # tokenizer, image_processor)
                # batch = collator_utils.generic_collator(
                #     graph_features=[g], text_features=[t], image_features=[i]
                # )
                g, t, i = tree_to_graph_features(
                    tree, tokenizer, image_processor
                )

                graph_features_batch = {
                    GraphFeatures.Edges: [g["edge_index"]],
                    GraphFeatures.OutDegree: [g["out_degree"]],
                    GraphFeatures.ImageMask: [g["image_mask"]],
                    GraphFeatures.Distance: [g["distance"]],
                    GraphFeatures.RotaryPos: [g["rotary_position"]],
                    "in_degree": [
                        g["in_degree"]
                    ],  # not in enum, added manually
                }

                text_features_batch = {
                    TextFeatures.InputIds: [t["input_ids"]],
                    TextFeatures.AttentionMask: [t["attention_mask"]],
                    TextFeatures.TokenTypeIds: [t["token_type_ids"]],
                }

                image_features_batch = {
                    ImageFeatures.PixelValues: [i["pixel_values"]],
                }

                batch = collator_utils.generic_collator(
                    graph_features=graph_features_batch,
                    text_features=text_features_batch,
                    image_features=image_features_batch,
                )
                # batch = {
                #     k: (v.to(self.device) if torch.is_tensor(v) else v)
                #     for k, v in batch.items()
                #     }
                batch = self._to_device(batch)
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[1]
                embeddings.append(out.detach().cpu())

        arr = torch.cat(embeddings, dim=0).numpy()
        print(f"[Info] Final embedding shape: {arr.shape}")
        return arr


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    eg = EmbeddingGenerator(args.ckpt, args.device)
    embs = eg.generate(args.json)

    out_path = args.out or (os.path.splitext(args.json)[0] + ".embeddings.npy")
    np.save(out_path, embs)
    print(f"✅ Saved embeddings → {out_path}, shape={embs.shape}")


if __name__ == "__main__":
    main()
