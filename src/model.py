"""Attention-based multiple-instance learning model for CAR-T response.

A patient is a bag of cells. Each cell is a 50-dim PCA vector. The model
scores every cell, pools them with softmax attention into one patient vector,
and outputs the probability of response. Same class must be used in Colab
for training and here for inference.
"""
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, d_in: int = 50, d: int = 128, d_att: int = 64, p_drop: float = 0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.V = nn.Linear(d, d_att)
        self.U = nn.Linear(d, d_att)
        self.w = nn.Linear(d_att, 1)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        """x: (N cells, d_in). Returns (probability scalar, attention weights of shape (N,))."""
        h = self.enc(x)
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(scores, dim=0)
        z = (a.unsqueeze(-1) * h).sum(0)
        return torch.sigmoid(self.head(z)).squeeze(), a
