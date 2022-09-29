from typing import Union

import torch
from torch import Tensor
import torch.nn as nn

from src.model.news_encoder import NewsEncoder


class UserEncoder(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, num_users: int, pad_user_id: int, num_gru_layers: int,
                 gru_dropout: float, long_term_dropout: float, combine_type: str):
        r"""
        Initialization

        Args:
            news_encoder: a NewsEncoder object.
            num_users: number of training user.
            pad_user_id: ID of pad value in user dictionary.
            num_gru_layers: number of recurrent layers in GRU netowrk.
            gru_dropout: dropout value in GRU network.
            long_term_dropout: dropout value in Long-Term User Representations
            combine_type: method to combine the long-term and short-term user presentations.
        """
        super().__init__()
        self.news_encoder = news_encoder
        self.long_term_embed = nn.Embedding(num_embeddings=num_users, embedding_dim=self.news_encoder.embed_dim,
                                            padding_idx=pad_user_id)
        self.long_term_dropout = nn.Dropout(long_term_dropout)
        self.num_gru_layers = num_gru_layers
        self.short_term_gru = nn.GRU(input_size=self.news_encoder.embed_dim, hidden_size=self.news_encoder.embed_dim,
                                     num_layers=num_gru_layers, batch_first=True, dropout=gru_dropout,
                                     bidirectional=False)
        self.combine_type = combine_type
        if self.combine_type == 'con':
            self.reduce_user_dim = nn.Linear(in_features=self.news_encoder.embed_dim * 2,
                                             out_features=self.news_encoder.embed_dim)

    def forward(self, user_encoding: Tensor, title_encoding: Tensor, title_attn_mask: Tensor, history_mask: Tensor,
                sapo_encoding: Union[Tensor, None] = None, sapo_attn_mask: Union[Tensor, None] = None,
                category_encoding: Union[Tensor, None] = None):
        r"""
        Forward propagation

        Args:
            user_encoding: tensor of shape ``(batch_size)``.
            title_encoding: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            title_attn_mask: tensor of shape ``(batch_size, num_clicked_news, title_length)``.
            history_mask: tensor of shape ``(batch_size, num_clicked_news)``.
            sapo_encoding: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            sapo_attn_mask: tensor of shape ``(batch_size, num_clicked_news, sapo_length)``.
            category_encoding: tensor of shape ``(batch_size, num_clicked_news)``.

        Returns:
            tensor of shape ``(batch_size, embed_dim)``
        """
        batch_size = title_encoding.shape[0]
        num_clicked_news = title_encoding.shape[1]
        title_encoding = title_encoding.view(batch_size * num_clicked_news, -1)
        title_attn_mask = title_attn_mask.view(batch_size * num_clicked_news, -1)
        sapo_encoding = sapo_encoding.view(batch_size * num_clicked_news, -1)
        sapo_attn_mask = sapo_attn_mask.view(batch_size * num_clicked_news, -1)
        category_encoding = category_encoding.view(batch_size * num_clicked_news)

        news_repr = self.news_encoder(title_encoding=title_encoding, title_attn_mask=title_attn_mask,
                                      sapo_encoding=sapo_encoding, sapo_attn_mask=sapo_attn_mask,
                                      category_encoding=category_encoding)
        news_repr = news_repr.view(batch_size, num_clicked_news, -1)
        history_count = history_mask.long().sum(-1, keepdims=False)
        news_repr_packed = nn.utils.rnn.pack_padded_sequence(input=news_repr, lengths=history_count.cpu(),
                                                             batch_first=True, enforce_sorted=False)
        long_term_repr = self.long_term_embed(user_encoding)
        long_term_repr = self.long_term_dropout(long_term_repr)
        if self.combine_type == 'ini':
            init_hidden_state = torch.stack([long_term_repr] * self.num_gru_layers, dim=0)
            _, last_hidden_state = self.short_term_gru(news_repr_packed, init_hidden_state)
            user_repr = last_hidden_state[-1]
        elif self.combine_type == 'con':
            _, last_hidden_state = self.short_term_gru(news_repr_packed)
            short_term_repr = last_hidden_state[-1]
            user_repr = torch.concat([short_term_repr, long_term_repr], dim=1)
            user_repr = self.reduce_user_dim(user_repr)
        else:
            raise ValueError('Invalid method to combine the long-term and short-term user presentations')

        return user_repr
