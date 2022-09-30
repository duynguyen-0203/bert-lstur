from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as torch_f

from src.model.news_encoder import NewsEncoder
from src.model.user_encoder import UserEncoder


class LSTUR(nn.Module):
    r"""
    Implementation of Multi-interest matching network for news recommendation. Please see the paper in
    https://aclanthology.org/P19-1033.pdf.
    """
    def __init__(self, news_encoder: NewsEncoder, user_encoder: UserEncoder):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder

    def forward(self, user_encoding: Tensor, title: Tensor, title_mask: Tensor, his_title: Tensor,
                his_title_mask: Tensor, his_mask: Tensor, sapo: Union[Tensor, None] = None,
                sapo_mask: Union[Tensor, None] = None, his_sapo: Union[Tensor, None] = None,
                his_sapo_mask: Union[Tensor, None] = None, category: Union[Tensor, None] = None,
                his_category: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            user_encoding: tensor of shape ``(batch_size)``.
            title: tensor of shape ``(batch_size, num_candidates, title_length)``.
            title_mask: tensor of shape ``(batch_size, num_candidates, title_length)``.
            his_title: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_title_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            his_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            sapo_mask: tensor of shape ``(batch_size, num_candidates, sapo_length)``.
            his_sapo: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            his_sapo_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category: tensor of shape ``(batch_size, num_candidates)``.
            his_category: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tensor of shape ``(batch_size, num_candidates)``
        """
        batch_size = title.shape[0]
        # Representation of the candidate news
        num_candidates = title.shape[1]
        title = title.view(batch_size * num_candidates, -1)
        title_mask = title_mask.view(batch_size * num_candidates, -1)
        sapo = sapo.view(batch_size * num_candidates, -1)
        sapo_mask = sapo_mask.view(batch_size * num_candidates, -1)
        category = category.view(batch_size * num_candidates)
        candidate_news_repr = self.get_news_repr(title_encoding=title, title_attn_mask=title_mask, sapo_encoding=sapo,
                                                 sapo_attn_mask=sapo_mask, category_encoding=category)
        candidate_news_repr = candidate_news_repr.view(batch_size, num_candidates, -1)

        # Representation of the users
        user_repr = self.get_user_repr(user_encoding=user_encoding, title_encoding=his_title,
                                       title_attn_mask=his_title_mask, history_mask=his_mask,
                                       sapo_encoding=his_sapo, sapo_attn_mask=his_sapo_mask,
                                       category_encoding=his_category)

        # Click predictor
        logits = torch.bmm(candidate_news_repr, user_repr.unsqueeze(dim=2)).squeeze(dim=2)

        return logits

    def get_user_repr(self, user_encoding: Tensor, title_encoding: Tensor, title_attn_mask: Tensor,
                      history_mask: Tensor, sapo_encoding: Tensor, sapo_attn_mask: Tensor,
                      category_encoding: Tensor):
        r"""
        Get the user's representation

        Args:
            user_encoding: tensor of shape ``(batch_size)``.
            title_encoding: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            history_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo_encoding: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category_encoding: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        user_repr = self.user_encoder(user_encoding=user_encoding, title_encoding=title_encoding,
                                      title_attn_mask=title_attn_mask, history_mask=history_mask,
                                      sapo_encoding=sapo_encoding, sapo_attn_mask=sapo_attn_mask,
                                      category_encoding=category_encoding)

        return user_repr

    def get_news_repr(self, title_encoding: Tensor, title_attn_mask: Tensor, sapo_encoding: Tensor,
                      sapo_attn_mask: Tensor, category_encoding: Tensor):
        r"""
        Get the news's representation

        Args:
            title_encoding: tensor of shape ``(batch_size, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, title_length)``.
            sapo_encoding: tensor of shape ``(batch_size, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, sapo_length)``.
            category_encoding: tensor of shape ``(batch_size)``.

        Returns:
            Tensor of shape ``(batch_size, embed_dim)``
        """
        news_repr = self.news_encoder(title_encoding=title_encoding, title_attn_mask=title_attn_mask,
                                      sapo_encoding=sapo_encoding, sapo_attn_mask=sapo_attn_mask,
                                      category_encoding=category_encoding)

        return news_repr

    @staticmethod
    def compute_score(user_repr: Tensor, news_repr: Tensor):
        r"""


        Args:
            user_repr: tensor of shape ``(batch_size, embed_dim)``
            news_repr: tensor of shape ``(batch_size, num_candidates, embed_dim)``

        Returns:

        """
        logits = torch.bmm(news_repr, user_repr.unsqueeze(dim=2)).squeeze(dim=2)
        if news_repr.shape[1] == 1:
            scores = torch.sigmoid(logits)
        else:
            scores = torch_f.softmax(logits, dim=1)

        return scores
