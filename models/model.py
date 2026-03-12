import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta


torch.set_printoptions(precision=4)



class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(
                -1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1),
                       y.size(-1))  # (timesteps, samples, output_size)

        return y


class StaticFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(StaticFeatureEncoder, self).__init__()

        self.batchnorm = nn.BatchNorm1d(hidden_dim * 4)

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.meta_linear = nn.Linear(51, hidden_dim)

        self.image_ln = nn.LayerNorm(512)
        self.text_ln = nn.LayerNorm(512)
        self.temp_ln = nn.LayerNorm(512)
        self.meta_ln = nn.LayerNorm(512)
        self.feature_ln = nn.LayerNorm(512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, img_encoding, text_encoding, temporal_encoding=None, meta_data=None):
        image_embedding = self.dropout(self.image_ln(img_encoding))
        text_embedding = self.dropout(self.text_ln(text_encoding))
        temporal_embedding = self.dropout(self.temp_ln(temporal_encoding))
        meta_embedding = self.dropout(self.meta_ln(self.meta_linear(meta_data)))

        features = self.activation(torch.cat([image_embedding, text_embedding, temporal_embedding, meta_embedding], dim=1))
        features = self.batchnorm(features)
        features = self.feature_fusion(features)
        features = self.feature_ln(self.activation(features))

        return features


# class StaticFeatureEncoder(nn.Module):
#     def __init__(self, hidden_dim, dropout=0.2):
#         super(StaticFeatureEncoder, self).__init__()
#         # BatchNorm1d를 LayerNorm으로 변경
#         self.norm = nn.LayerNorm(hidden_dim * 4)
        
#         self.feature_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim * 2, bias=False),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )

#         self.meta_linear = nn.Linear(51, hidden_dim)
#         self.image_ln = nn.LayerNorm(512)
#         self.text_ln = nn.LayerNorm(512)
#         self.temp_ln = nn.LayerNorm(512)
#         self.meta_ln = nn.LayerNorm(512)
#         self.feature_ln = nn.LayerNorm(512)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, img_encoding, text_encoding, temporal_encoding=None, meta_data=None):
#         image_embedding = self.dropout(self.image_ln(img_encoding))
#         text_embedding = self.dropout(self.text_ln(text_encoding))
#         temporal_embedding = self.dropout(self.temp_ln(temporal_encoding))
#         meta_embedding = self.dropout(self.meta_ln(self.meta_linear(meta_data)))
        
#         features = self.activation(torch.cat([image_embedding, text_embedding, temporal_embedding, meta_embedding], dim=1))
#         features = self.norm(features)  # BatchNorm1d를 LayerNorm으로 변경한 부분
#         features = self.feature_fusion(features)
#         features = self.feature_ln(self.activation(features))
        
#         return features



# class StaticFeatureEncoder(nn.Module):
#     def __init__(self, hidden_dim, dropout=0.2):
#         super(StaticFeatureEncoder, self).__init__()

#         self.batchnorm = nn.BatchNorm1d(hidden_dim * 4)

#         self.feature_fusion = nn.Sequential(
#             nn.Linear(hidden_dim * 4, hidden_dim * 2, bias=False),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )

#         self.meta_linear = nn.Linear(51, hidden_dim)

#         self.image_ln = nn.LayerNorm(512)
#         self.text_ln = nn.LayerNorm(512)
#         self.temp_ln = nn.LayerNorm(512)
#         self.meta_ln = nn.LayerNorm(512)
#         self.feature_ln = nn.LayerNorm(512)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, img_encoding, text_encoding, temporal_encoding=None, meta_data=None):
#         image_embedding = self.dropout(self.image_ln(img_encoding))
#         text_embedding = self.dropout(self.text_ln(text_encoding))
#         temporal_embedding = self.dropout(self.temp_ln(temporal_encoding))
#         meta_embedding = self.dropout(self.meta_ln(self.meta_linear(meta_data)))

#         features = self.activation(torch.cat([image_embedding, text_embedding, temporal_embedding, meta_embedding], dim=1))
#         features = self.batchnorm(features)
#         features = self.feature_fusion(features)
#         features = self.feature_ln(self.activation(features))

#         return features




class K_item_sales_Embedder(nn.Module):
    def __init__(self, embedding_dim, trend_len, num_trends):
        """
        embedding_dim (int): 임베딩 차원.
        trend_len (int): 트렌드 데이터의 길이.
        num_trends (int): Google Trends 데이터의 개수
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.trend_len = trend_len

        # Input Linear Layers
        self.input_linear_pop = nn.Linear(1, embedding_dim)  # Pop signal
        self.k_item_sales_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())  # ERP distance

        # Gating Layers for ntrends
        self.gtrend_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, gtrends, k_item_sales=None, mask=None, pop_signal=None):
        """
        Forward method for K_item_sales_Embedder.
        
        Args:
            gtrends (Tensor): Trend data of shape (batch_size, hidden_dim, trend_len) - 메타데이터조합의 12주간 mean 추이 데이터.
            k_item_sales (Tensor or None): ERP distance data of shape (batch_size, seq_len_k_items, hidden_dim).
            mask (Tensor or None): Attention mask for the encoder.
            pop_signal (Tensor or None): Popularity signal of shape (batch_size, trend_len).
        
        Returns:
            Tensor: Encoded representation from the Transformer Encoder.
        """
        
        # Process ntrends
        gtrend_emb = self.gtrend_gate(gtrends.transpose(1, 2)) * gtrends.transpose(1, 2)
        gtrend_emb = self.dropout(gtrend_emb)

        combined_list = [gtrend_emb]
        seq_len = gtrend_emb.size(1)

        # Process k_item_sales if provided
        if k_item_sales is not None:
            k_item_sales_emb = self.k_item_sales_gate(k_item_sales) * k_item_sales
            k_item_sales_emb = self.dropout(k_item_sales_emb)
            combined_list.append(k_item_sales_emb)
            seq_len += k_item_sales_emb.size(1)

        # Process pop_signal if provided
        if pop_signal is not None:
            pop_emb = self.input_linear_pop(pop_signal.unsqueeze(2))
            pop_emb = self.gtrend_gate(pop_emb) * pop_emb
            pop_emb = self.dropout(pop_emb)
            combined_list.append(pop_emb)
            seq_len += pop_emb.size(1)

        combined = torch.cat(combined_list, axis=1).permute(1, 0, 2)

        # Generate mask only if needed
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.size(1) != seq_len or mask.size(2) != seq_len:
                mask = None  # Discard the mask if it doesn't match the sequence length

        context_emb = self.encoder(combined, mask=mask)
        return context_emb
    
    # def forward(self, gtrends, k_item_sales, mask, pop_signal=None):
    #     batch_size = gtrends.size(0)
    
    #     # gtrends shape: (batch_size, hidden_dim, trend_len)
    #     gtrend_emb = self.gtrend_gate(gtrends.transpose(1, 2)) * gtrends.transpose(1, 2)
    #     gtrend_emb = self.dropout(gtrend_emb)

    #     k_item_sales_emb = self.k_item_sales_gate(k_item_sales) * k_item_sales
    #     k_item_sales_emb = self.dropout(k_item_sales_emb)
    #     # k_item_sales_emb = self.dropout(k_item_sales)

    #     pop_emb = self.input_linear_pop(pop_signal.unsqueeze(2))
    #     pop_emb = self.pop_gate(pop_emb) * pop_emb
    #     pop_emb = self.dropout(pop_emb)

    #     # Combine embeddings
    #     combined = torch.cat([gtrend_emb, pop_emb, k_item_sales_emb], axis=1)
    #     # combined = torch.cat([gtrend_emb, k_item_sales_emb], axis=1)

    #     # Transformer Encoder에 입력하기 전에 차원 변환
    #     combined = combined.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

    #     # Transformer Encoder에 입력하기 전에 차원 변환
    #     combined = combined.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        
    #     context_emb = self.encoder(combined, mask=mask)
    #     return context_emb

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (week, month, year)
        w, m, y = temporal_features[:, 0].unsqueeze(2), temporal_features[:, 1].unsqueeze(2), temporal_features[:, 2].unsqueeze(2)
        w_emb, m_emb, y_emb = self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([w_emb, m_emb, y_emb], dim=2))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings


class TransformerDecoderLayer_first(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer_first, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer_first, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Given_0_nonauto_linear(nn.Module):
    def __init__(self, hidden_dim, trend_len, output_len):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, output_len * hidden_dim, bias=True)

        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(0.2)

    def forward(self, memory_of_decoder_se):
        out = self.activation(self.linear1(memory_of_decoder_se))
        return out


class GTM(pl.LightningModule):
    def __init__(self, hidden_dim, output_dim, num_heads,
                 num_layers, trend_len, num_trends, gpu_num, lr,
                 batch_size, sales_transform,
                 n_neighbors, ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_len = output_dim

        self.gpu_num = gpu_num
        self.save_hyperparameters()
        self.trend_len = trend_len
        self.lr = lr
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors

        self.sales_transform = sales_transform

        # Encoder
        self.dummy_encoder = DummyEmbedder(hidden_dim)
        self.k_item_sales_encoder = K_item_sales_Embedder(hidden_dim, trend_len, num_trends)
        self.static_feature_encoder = StaticFeatureEncoder(hidden_dim)

        """decoder first 구성"""
        decoder_layer_first = TransformerDecoderLayer_first(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * 4,
                                                            dropout=0.1)
        self.decoder_first = nn.TransformerDecoder(decoder_layer_first, num_layers)
        self.given_0_nonauto_linear = Given_0_nonauto_linear(hidden_dim, trend_len, output_dim)

    def _generate_deocder_fisrt_mask(self):
        mask = (torch.triu(torch.ones(self.trend_len, self.trend_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))
        return mask

    def _generate_k_item_sales_mask(self):
        mask = (torch.triu(torch.ones(self.output_len, self.output_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))
        mask_list = []
        for i in range(self.n_neighbors):
            mask_list.append(mask)
        column_mask = torch.stack(mask_list).reshape(self.output_len * self.n_neighbors, self.output_len)
        column_mask_list = []
        for i in range(self.n_neighbors):
            column_mask_list.append(column_mask)

        mask = torch.stack(column_mask_list, axis=1).reshape(self.output_len * self.n_neighbors, self.output_len * self.n_neighbors)
        return mask

    
    def forward(self, item_sales, temporal_features, ntrends, images, texts, meta_data,
                k_item_sales=None, pop_signal=None):
        # 기존 전처리 부분
        ntrends = ntrends.squeeze(1)
        temporal_encoding = self.dummy_encoder(temporal_features)

        # encoder 부분
        img_encoding = images
        text_encoding = texts
        static_feature_fusion = self.static_feature_encoder(img_encoding,
                                                            text_encoding,
                                                            temporal_encoding=temporal_encoding[:, 52],
                                                            meta_data=meta_data)

        # 시퀀스 길이 계산
        batch_size = ntrends.size(0)
        
        # 마스크 생성 (필요한 경우만 사용)
        if k_item_sales is not None or pop_signal is not None:
            gtrend_len = self.trend_len
            pop_len = self.trend_len
            k_item_len = self.n_neighbors * 12  # n_neighbors * 12
            total_seq_len = gtrend_len + pop_len + k_item_len

            device = f'cuda:{self.gpu_num}'
            mask_0 = torch.zeros(gtrend_len, gtrend_len).to(device)
            mask_1 = torch.zeros(gtrend_len, k_item_len).to(device)

            ntrends_mask = torch.cat([
                self._generate_deocder_fisrt_mask(),
                mask_0,
                mask_1
            ], axis=1)

            pop_mask = torch.cat([
                mask_0,
                self._generate_deocder_fisrt_mask(),
                mask_1
            ], axis=1)

            k_item_mask = torch.cat([
                mask_1.transpose(1, 0),
                mask_1.transpose(1, 0),
                self._generate_k_item_sales_mask()
            ], axis=1)

            combined_mask = torch.cat([ntrends_mask, pop_mask, k_item_mask], axis=0)
        else:
            combined_mask = None

        # Encoder 실행 (ntrends만 입력 가능하도록 수정)
        k_item_sales_emb = self.k_item_sales_encoder(
            gtrends=ntrends.permute(0, 2, 1),  # (batch_size, hidden_dim, trend_len)
            k_item_sales=k_item_sales,
            mask=combined_mask, #combined_mask
            pop_signal=pop_signal
        )

        # Decoder 실행
        memory_of_decoder_se = self.decoder_first(
            tgt=static_feature_fusion.unsqueeze(0),
            memory=k_item_sales_emb
        )

        # 최종 예측
        forecast = self.given_0_nonauto_linear(memory_of_decoder_se).reshape(-1, self.output_len, 512)

        return forecast
    
    
    # def forward(self, item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal):
    #     # 기존 전처리 부분
    #     ntrends = ntrends.squeeze(1)
    #     temporal_encoding = self.dummy_encoder(temporal_features)
        
    #     # encoder 부분
    #     img_encoding = images
    #     text_encoding = texts
    #     static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, 
    #                                                     temporal_encoding=temporal_encoding[:, 52],
    #                                                     meta_data=meta_data)

    #     # 시퀀스 길이 계산
    #     batch_size = ntrends.size(0)
    #     gtrend_len = self.trend_len  # 12
    #     pop_len = self.trend_len     # 12
    #     k_item_len = self.n_neighbors * 12  # n_neighbors * 12
    #     total_seq_len = gtrend_len + pop_len + k_item_len
        
    #     # 기본 마스크 생성
    #     device = f'cuda:{self.gpu_num}'
    #     mask_0 = torch.zeros(gtrend_len, gtrend_len).to(device)
    #     mask_1 = torch.zeros(gtrend_len, k_item_len).to(device)
        
    #     # 각 컴포넌트별 마스크 생성
    #     ntrends_mask = torch.cat([
    #         self._generate_deocder_fisrt_mask(),  # 트렌드 자기 주의력 마스크
    #         mask_0,                               # 트렌드-팝 크로스 어텐션 마스크
    #         mask_1                                # 트렌드-아이템 크로스 어텐션 마스크
    #     ], axis=1)
        
    #     pop_mask = torch.cat([
    #         mask_0,                               # 팝-트렌드 크로스 어텐션 마스크
    #         self._generate_deocder_fisrt_mask(),  # 팝 자기 주의력 마스크
    #         mask_1                                # 팝-아이템 크로스 어텐션 마스크
    #     ], axis=1)
        
    #     k_item_mask = torch.cat([
    #         mask_1.transpose(1, 0),               # 아이템-트렌드 크로스 어텐션 마스크
    #         mask_1.transpose(1, 0),               # 아이템-팝 크로스 어텐션 마스크
    #         self._generate_k_item_sales_mask()    # 아이템 자기 주의력 마스크
    #     ], axis=1)
        
    #     # 최종 마스크 생성
    #     combined_mask = torch.cat([ntrends_mask, pop_mask, k_item_mask], axis=0)
        
    #     # 디버깅을 위한 shape 출력
    #     # print(f"Mask shapes - ntrends: {ntrends_mask.shape}, pop: {pop_mask.shape}, k_item: {k_item_mask.shape}")
    #     # print(f"Combined mask shape: {combined_mask.shape}")
        
    #     # Encoder 실행
    #     k_item_sales_emb = self.k_item_sales_encoder(
    #         ntrends.permute(0, 2, 1),  # (batch_size, hidden_dim, trend_len)
    #         k_item_sales,              # (batch_size, n_neighbors * 12, hidden_dim)
    #         combined_mask,             # (total_seq_len, total_seq_len)
    #         pop_signal=pop_signal
    #     )
        
    #     # Decoder 실행
    #     memory_of_decoder_se = self.decoder_first(
    #         tgt=static_feature_fusion.unsqueeze(0),
    #         memory=k_item_sales_emb
    #     )
        
    #     # 최종 예측
    #     forecast = self.given_0_nonauto_linear(memory_of_decoder_se).reshape(-1, self.output_len, 512)
        
    #     return forecast

    # def forward(self, item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal):
        
    #     # 차원 확인용 print문 추가
    #     ntrends = ntrends.squeeze(1)  # (4, 1, 12, 512) -> (4, 12, 512)
    #     print("ntrends shape before processing::", ntrends.shape) # (batch_size, trend_len=12, hidden_dim=D)
    #     print("k_item_sales shape:", k_item_sales.shape)
        
    #     temporal_encoding = self.dummy_encoder(temporal_features)

    #     """encoder"""
    #     img_encoding = images
    #     text_encoding = texts

    #     static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, temporal_encoding=temporal_encoding[:, 52],
    #                                                         meta_data=meta_data)

    #     mask_0 = torch.full((self.trend_len, self.trend_len), float(0.0)).to(f'cuda:{self.gpu_num}')
    #     mask_1 = torch.full((self.trend_len, self.n_neighbors * 12), float(0.0)).to(f'cuda:{self.gpu_num}')
    #     ntrends_mask = torch.cat([self._generate_deocder_fisrt_mask(), mask_0, mask_1], axis=1)
    #     # ntrends_mask = torch.cat([self._generate_deocder_fisrt_mask(), mask_1], axis=1)
    #     pop_mask = torch.cat([mask_0, self._generate_deocder_fisrt_mask(), mask_1], axis=1)
    #     k_item_mask = torch.cat([mask_1.transpose(1, 0), mask_1.transpose(1, 0), self._generate_k_item_sales_mask()], axis=1)
    #     # k_item_mask = torch.cat([mask_1.transpose(1, 0), self._generate_k_item_sales_mask()], axis=1)


    #     # nn.Module이 __call__ 메소드를 통해 forward 메소드를 자동으로 호출하도록 구현되어 있기 때문에
    #     # self.k_item_sales_encoder 인스턴스를 함수처럼 호출하면 내부적으로 forward 메소드가 실행됨
    #     # ntrends permute 전에 차원 확인
    #     ntrends_permuted = ntrends.permute(0, 2, 1) ## (batch_size, D, trend_len=12)
    #     print("ntrends_permuted shape:", ntrends_permuted.shape)
    #     k_item_sales_emb = self.k_item_sales_encoder(ntrends.permute(0, 2, 1), k_item_sales, torch.cat([ntrends_mask, pop_mask, k_item_mask], axis=0),
    #                                                  pop_signal=pop_signal)
    #     # k_item_sales_emb = self.k_item_sales_encoder(ntrends.permute(0, 2, 1), k_item_sales, torch.cat([ntrends_mask, k_item_mask], axis=0))

    #     memory_of_decoder_se = self.decoder_first(tgt=static_feature_fusion.unsqueeze(0), memory=k_item_sales_emb)
    #     forecast = self.given_0_nonauto_linear(memory_of_decoder_se).reshape(-1, self.output_len, 512)

        

    #     return forecast

    def configure_optimizers(self):
        import torch_optimizer as optim
        from torch_optimizer import QHAdam, LARS
        from torch_optimizer import Lookahead
        from torch.optim import AdamW
        from torch_optimizer import DiffGrad
        from lion_pytorch import Lion
        from adabelief_pytorch import AdaBelief
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #팀장님 원본

#         optimizer = AdaBelief(             # 팀장님 것보다 더 성능 좋아진 옵티마이저
#     self.parameters(), 
#     lr=0.0001, # 기존 학습률 유지
#     eps=1e-16, # 분모가 0이 되는 것을 방지
#     betas=(0.9, 0.999), # Adam과 동일한 설정
#     weight_decay=1e-5, # 가중치 감소 적용
#     rectify=False # 필요에 따라 True로 설정
# )


        # 학습률 스케줄러
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 최소값을 기준으로 학습률 조정
            factor=0.5,  # 학습률 감소 비율 /팀장님 0.5
            patience=5,  # 5번의 에포크 동안 개선이 없으면 학습률 감소 /팀장님 5
            verbose=True  # 학습률 변화 로그 출력
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_week_given0_ad_smape_gs',  # 정확한 메트릭 이름 입력
                'interval': 'epoch',  # 에폭 단위로 LR 조정
                'frequency': 1,
            }
        }

    def training_step(self, train_batch, batch_idx):
        item_sales, temporal_features, ntrends, images, texts, \
            real_value_sales, release_dates, \
            meta_data, k_item_sales, pop_signal = train_batch

        forecasted_sales = self.forward(item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal)
        loss = F.mse_loss(item_sales, forecasted_sales)

        self.log('train_loss_total', loss)

        with torch.no_grad():
            batch_size = self.batch_size
            plot_idx = [batch_idx * batch_size + i for i in range(len(item_sales)) if (batch_idx * batch_size + i) % 1000 == 0]

            forecasted_sales = self.forward(item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal)

            unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

            gt = real_value_sales
            pred = unscaled_forecasted_sales

            r2score = R2Score()
            r2_score_gs_stack = torch.stack([r2score(pred[i], gt.detach().cpu()[i]) for i in range(len(gt))])
            r2_score_gs = torch.mean(r2_score_gs_stack)

            ad_smape = SymmetricMeanAbsolutePercentageError()
            smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
            smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

            mse_gs_stack = torch.mean(F.mse_loss(pred, gt.detach().cpu(), reduction='none'), axis=1)
            mse_gs = torch.mean(mse_gs_stack)

            smape_adjust_accum_stack = torch.stack(
                [ad_smape(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) * 0.5 for i in range(len(gt))])
            smape_adjust_accum = torch.mean(smape_adjust_accum_stack)

            mse_accum_stack = torch.stack([F.mse_loss(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) for i in range(len(gt))])
            mse_accum = torch.mean(mse_accum_stack)

            wape_stack = torch.sum(torch.abs(gt.detach().cpu() - pred), axis=-1) / torch.sum(gt.detach().cpu(), axis=-1)
            wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

            mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
            mae = torch.mean(mae_stack)

            self.log('train_week_given0_r2score_gs', r2_score_gs)
            self.log('train_week_given0_ad_smape_gs', smape_adjust_gs)
            self.log('train_week_given0_mse_gs', mse_gs)
            self.log('train_week_given0_ad_smape_accum_12', smape_adjust_accum)
            self.log('train_week_given0_mse_accum_12', mse_accum)
            self.log('train_week_wape', wape)
            self.log('train_week_mae', mae)

            print('train_week_given0_r2score_gs', r2_score_gs,
                  'train_week_given0_ad_smape_gs', smape_adjust_gs,
                  'train_week_given0_mse_gs', mse_gs,
                  'train_week_given0_ad_smape_accum_12', smape_adjust_accum,
                  'train_week_given0_mse_accum_12', mse_accum,
                  'train_week_wape', wape,
                  'train_week_mae', mae,
                  'LR:', self.optimizers().param_groups[0]['lr'],
                  )

            for idx in plot_idx:
                inner_idx = idx - batch_idx * batch_size
                release_dates_p = release_dates.detach().cpu()[inner_idx].int()
                release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
                date_range = np.array(pd.date_range(release_dates_p, release_dates_p + relativedelta(weeks=self.output_len - 1), freq='7d'))

                plt.plot(date_range, gt[inner_idx].detach().cpu(), color='r')
                plt.plot(date_range, pred[inner_idx], color='g')
                plt.title(f'train_week_given0_unscaled')
                plt.xlabel(f'r2_score:{np.format_float_positional(r2_score_gs_stack[inner_idx], precision=4)}, '
                           f'ad_smape:{np.format_float_positional(smape_adjust_gs_stack[inner_idx], precision=4)}, '
                           f'mse:{np.format_float_positional(mse_gs_stack[inner_idx], precision=0)}, '
                           f'ad_smape_accum:{np.format_float_positional(smape_adjust_accum_stack[inner_idx], precision=4)}'
                           f'wape:{np.format_float_positional(wape_stack[inner_idx], precision=4)},'
                           f'mae:{np.format_float_positional(mae_stack[inner_idx], precision=0)},')
                plt.legend(['gt', 'pred'])
                self.logger.log_image(key=f'train_week_given0', images=[plt])
                plt.show()
                plt.clf()

        return loss

    def validation_step(self, test_batch, batch_idx):
        item_sales, temporal_features, ntrends, images, texts, \
            real_value_sales, release_dates, \
            meta_data, k_item_sales, pop_signal = test_batch

        batch_size = self.batch_size
        plot_idx = [batch_idx * batch_size + i for i in range(len(item_sales)) if (batch_idx * batch_size + i) % 100 == 0]

        forecasted_sales = self.forward(item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        gt = real_value_sales
        pred = unscaled_forecasted_sales

        r2score = R2Score()
        r2_score_gs_stack = torch.stack([r2score(pred[i], gt.detach().cpu()[i]) for i in range(len(gt))])
        r2_score_gs = torch.mean(r2_score_gs_stack)

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

        mse_gs_stack = torch.mean(F.mse_loss(pred, gt.detach().cpu(), reduction='none'), axis=1)
        mse_gs = torch.mean(mse_gs_stack)

        smape_adjust_accum_stack = torch.stack(
            [ad_smape(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_accum = torch.mean(smape_adjust_accum_stack)

        mse_accum_stack = torch.stack([F.mse_loss(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) for i in range(len(gt))])
        mse_accum = torch.mean(mse_accum_stack)

        wape_stack = torch.sum(torch.abs(gt.detach().cpu() - pred), axis=-1) / torch.sum(gt.detach().cpu(), axis=-1)
        wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

        mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
        mae = torch.mean(mae_stack)

        self.log('val_week_given0_r2score_gs', r2_score_gs)
        self.log('val_week_given0_ad_smape_gs', smape_adjust_gs)
        self.log('val_week_given0_mse_gs', mse_gs)
        self.log('val_week_given0_ad_smape_accum_12', smape_adjust_accum)
        self.log('val_week_given0_mse_accum_12', mse_accum)
        self.log('val_week_wape', wape)
        self.log('val_week_mae', mae)

        print('val_week_given0_r2score_gs', r2_score_gs,
              'val_week_given0_ad_smape_gs', smape_adjust_gs,
              'val_week_given0_mse_gs', mse_gs,
              'val_week_given0_ad_smape_accum_12', smape_adjust_accum,
              'val_week_given0_mse_accum_12', mse_accum,
              'val_week_wape', wape,
              'val_week_mae', mae,
              )

        for idx in plot_idx:
            inner_idx = idx - batch_idx * batch_size
            release_dates_p = release_dates.detach().cpu()[inner_idx].int()
            release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
            date_range = np.array(pd.date_range(release_dates_p, release_dates_p + relativedelta(weeks=self.output_len - 1), freq='7d'))

            plt.plot(date_range, gt[inner_idx].detach().cpu(), color='r')
            plt.plot(date_range, pred[inner_idx], color='g')
            item_number = idx
            plt.title(f'val_week_given0_unscaled_{item_number}')
            plt.xlabel(f'r2_score:{np.format_float_positional(r2_score_gs_stack[inner_idx], precision=4)}, '
                       f'ad_smape:{np.format_float_positional(smape_adjust_gs_stack[inner_idx], precision=4)}, '
                       f'mse:{np.format_float_positional(mse_gs_stack[inner_idx], precision=0)}, '
                       f'ad_smape_accum:{np.format_float_positional(smape_adjust_accum_stack[inner_idx], precision=4)}, '
                       f'wape:{np.format_float_positional(wape_stack[inner_idx], precision=4)}, '
                       f'mae:{np.format_float_positional(mae_stack[inner_idx], precision=0)}')
            plt.legend(['gt', 'pred'])
            self.logger.log_image(key=f'val_week_given0_{item_number}', images=[plt])
            plt.show()
            plt.clf()

    def test_step(self, test_batch, batch_idx):
        item_sales, temporal_features, ntrends, images, texts, \
            real_value_sales, release_dates, \
            meta_data, k_item_sales, pop_signal = test_batch

        batch_size = self.batch_size
        plot_idx = [batch_idx * batch_size + i for i in range(len(item_sales)) if (batch_idx * batch_size + i) % 100 == 0]

        forecasted_sales = self.forward(item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        gt = real_value_sales
        pred = unscaled_forecasted_sales

        r2score = R2Score()
        r2_score_gs_stack = torch.stack([r2score(pred[i], gt.detach().cpu()[i]) for i in range(len(gt))])
        r2_score_gs = torch.mean(r2_score_gs_stack)

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

        mse_gs_stack = torch.mean(F.mse_loss(pred, gt.detach().cpu(), reduction='none'), axis=1)
        mse_gs = torch.mean(mse_gs_stack)

        smape_adjust_accum_stack = torch.stack(
            [ad_smape(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_accum = torch.mean(smape_adjust_accum_stack)

        mse_accum_stack = torch.stack([F.mse_loss(torch.sum(pred, dim=-1)[i], torch.sum(gt, dim=-1).detach().cpu()[i]) for i in range(len(gt))])
        mse_accum = torch.mean(mse_accum_stack)

        wape_stack = torch.sum(torch.abs(gt.detach().cpu() - pred), axis=-1) / torch.sum(gt.detach().cpu(), axis=-1)
        wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

        mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
        mae = torch.mean(mae_stack)

        self.log('test_week_given0_r2score_gs', r2_score_gs)
        self.log('test_week_given0_ad_smape_gs', smape_adjust_gs)
        self.log('test_week_given0_mse_gs', mse_gs)
        self.log('test_week_given0_ad_smape_accum_12', smape_adjust_accum)
        self.log('test_week_given0_mse_accum_12', mse_accum)
        self.log('test_week_wape', wape)
        self.log('test_week_mae', mae)

        print('test_week_given0_r2score_gs', r2_score_gs,
              'test_week_given0_ad_smape_gs', smape_adjust_gs,
              'test_week_given0_mse_gs', mse_gs,
              'test_week_given0_ad_smape_accum_12', smape_adjust_accum,
              'test_week_given0_mse_accum_12', mse_accum,
              'test_week_wape', wape,
              'test_week_mae', mae,
              )

        for idx in plot_idx:
            inner_idx = idx - batch_idx * batch_size
            release_dates_p = release_dates.detach().cpu()[inner_idx].int()
            release_dates_p = datetime.date(release_dates_p[0], release_dates_p[1], release_dates_p[2])
            date_range = np.array(pd.date_range(release_dates_p, release_dates_p + relativedelta(weeks=self.output_len - 1), freq='7d'))

            plt.plot(date_range, gt[inner_idx].detach().cpu(), color='r')
            plt.plot(date_range, pred[inner_idx], color='g')
            item_number = idx
            plt.title(f'test_week_given0_unscaled_{item_number}')
            plt.xlabel(f'r2_score:{np.format_float_positional(r2_score_gs_stack[inner_idx], precision=4)}, '
                       f'ad_smape:{np.format_float_positional(smape_adjust_gs_stack[inner_idx], precision=4)}, '
                       f'mse:{np.format_float_positional(mse_gs_stack[inner_idx], precision=0)},'
                       f'ad_smape_accum:{np.format_float_positional(smape_adjust_accum_stack[inner_idx], precision=4)}'
                       f'wape:{np.format_float_positional(wape_stack[inner_idx], precision=4)}, '
                       f'mae:{np.format_float_positional(mae_stack[inner_idx], precision=0)}')
            plt.legend(['gt', 'pred'])
            self.logger.log_image(key=f'test_week_given0_{item_number}', images=[plt])
            plt.show()
            plt.clf()

    def predict_step(self, test_batch, batch_idx):
        item_sales, temporal_features, ntrends, images, texts, \
            real_value_sales, release_dates, \
            meta_data, k_item_sales, pop_signal = test_batch

        forecasted_sales = self.forward(item_sales, temporal_features, ntrends, images, texts, meta_data, k_item_sales, pop_signal)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        gt = real_value_sales
        pred = unscaled_forecasted_sales

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])

        r2score = R2Score()
        r2_score_gs_stack = torch.stack([r2score(pred[i], gt.detach().cpu()[i]) for i in range(len(gt))])

        return smape_adjust_gs_stack
        # return r2_score_gs_stack
