"""
PP4CKD / PPB3 PyTorch Deep Neural Network Architecture.

Strictly adheres to Reymond group's Polypharmacology Browser (PPB3) specification:
  - Input Layer: Fingerprint dimension (4096 or 8192 bits)
  - Hidden Layer 1: Linear(in_features, 1000) -> ReLU -> Dropout(0.2)
  - Hidden Layer 2: Linear(1000, 500) -> ReLU -> Dropout(0.2)
  - Output Layer: Linear(500, out_features)
"""

import os
from typing import Optional, Union
import torch
import torch.nn as nn


class PPB3Net(nn.Module):
    """
    PPB3 Multi-Target Classification Neural Network.

    Parameters
    ----------
    input_dim : int, default=4096
        Input fingerprint feature dimension (e.g., 4096 for ECFP4/MHFP6, 8192 for Fused).
    output_dim : int, default=7676
        Number of target classes (default: 7,676 ChEMBL 36 targets).
    dropout : float, default=0.2
        Dropout probability between dense layers.
    """

    def __init__(self, input_dim: int = 4096, output_dim: int = 7676, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input bit fingerprint tensor.

        Returns
        -------
        torch.Tensor of shape (batch_size, output_dim)
            Unnormalized logits for BCEWithLogitsLoss.
        """
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning sigmoid probabilities.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
            Input bit fingerprint tensor.

        Returns
        -------
        torch.Tensor of shape (batch_size, output_dim)
            Sigmoid output probabilities in [0.0, 1.0].
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, os.PathLike],
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "PPB3Net":
        """
        Instantiate model and load weights from a PyTorch .pt checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to .pt weight file.
        input_dim : int, optional
            Input dimension. If None, inferred from state_dict weight shape.
        output_dim : int, optional
            Output dimension. If None, inferred from state_dict weight shape.
        device : str or torch.device, default="cpu"
            Device to map loaded weights.

        Returns
        -------
        PPB3Net
            Loaded evaluation-mode model.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Infer dimensions if not specified
        if input_dim is None:
            # First linear weight is net.0.weight shape: (1000, input_dim)
            input_dim = state_dict["net.0.weight"].shape[1]
        if output_dim is None:
            # Last linear weight is net.6.weight shape: (output_dim, 500)
            output_dim = state_dict["net.6.weight"].shape[0]

        model = cls(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
