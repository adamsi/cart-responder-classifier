"""Shared helpers: config loading, seeding, and turning a cell matrix into patient bags."""
import argparse
import os
import random

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


def abspath(rel: str) -> str:
    return os.path.join(ROOT, rel)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_bags(obs, Z):
    """Group cell embeddings by patient.

    obs: DataFrame with Patient_ID, y, annotated_clusters (one row per cell, aligned with Z)
    Z:   (n_cells, n_pcs) float32
    Returns patients (sorted array), bags {patient: (N_i, n_pcs)}, y (n_patients,), ctype {patient: (N_i,)}
    """
    pat = obs["Patient_ID"].values
    patients = np.unique(pat)
    bags = {p: Z[pat == p] for p in patients}
    ctype = {p: obs.loc[pat == p, "annotated_clusters"].values for p in patients}
    y = np.array([int(obs.loc[pat == p, "y"].iloc[0]) for p in patients])
    return patients, bags, y, ctype
