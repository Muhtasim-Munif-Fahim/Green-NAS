"""
Green-NAS Model Definitions

This module contains the Green-NAS discovered architectures.
These models were found by NSGA-II search and represent points
on the Pareto front of accuracy vs. efficiency.
"""

import torch
import torch.nn as nn


class GreenNASModel(nn.Module):
    """
    Green-NAS 1.0: Hybrid CNN-LSTM Architecture.
    A proven, stable architecture for spatiotemporal weather forecasting.

    Components:
        1. 1D CNN: Extracts local temporal features (short-term dependencies).
        2. LSTM: Captures long-term dependencies.
        3. Linear Head: Standard regression output.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GreenNASModel, self).__init__()

        # 1. Feature Extractor (CNN)
        self.cnn = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim,
            kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

        # 2. Recurrent Core (LSTM)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

        # 3. Output Head
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [Batch, Seq, Feat]

        # CNN expects [Batch, Feat, Seq]
        x = x.permute(0, 2, 1)
        x = self.relu(self.cnn(x))

        # Back to [Batch, Seq, Feat] for LSTM
        x = x.permute(0, 2, 1)

        # LSTM Processing
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state for prediction
        last_hidden = out[:, -1, :]

        return self.head(last_hidden)


def get_synflow_score(model, input_data):
    """
    Computes the SynFlow score (gradient flow at initialization).
    Used as a training-free proxy for model quality.
    """
    @torch.no_grad()
    def linearize(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.LSTM)):
            for p in m.parameters():
                p.abs_()

    model.apply(linearize)

    model.zero_grad()
    input_data = torch.ones_like(input_data) * 0.1
    output = model(input_data)

    torch.sum(output).backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += torch.sum(torch.abs(p * p.grad)).item()

    return score
