import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim: int, embed_dim: int):
        r"""
        Initialization
        Args:
            query_dim:  the dimension of the additive attention query vectors.
            embed_dim: the dimension of the ``embeddings``.
        """
        super().__init__()
        self.projection = nn.Linear(in_features=embed_dim, out_features=query_dim)
        self.query_vector = nn.Parameter(nn.init.xavier_uniform_(torch.empty(query_dim, 1),
                                                                 gain=nn.init.calculate_gain('tanh')).squeeze())

    def forward(self, embeddings: Tensor, mask: Tensor):
        r"""
        Forward propagation
        Args:
            embeddings: tensor of shape ``(batch_size, seq_length, embed_dim)``.
            mask: tensor of shape ``(batch_size, seq_length)``. Positions with True are allowed to attend.
        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        attn_weight = torch.matmul(torch.tanh(self.projection(embeddings)), self.query_vector)
        attn_weight.masked_fill_(~mask, 1e-30)
        attn_weight = torch_f.softmax(attn_weight, dim=1)
        seq_repr = torch.bmm(attn_weight.unsqueeze(dim=1), embeddings).squeeze(dim=1)

        return seq_repr
