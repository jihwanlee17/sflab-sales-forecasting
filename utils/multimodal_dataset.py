import os
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import pipeline
from utils.timefeatures import time_features
import pickle
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy.stats import boxcox

import torch.nn as nn
from torchvision import models


class ZeroShotDataset():
    def __init__(self,
                 sales_df_root,
                 n_trends_root, #수정된 부분 - 네이버 트렌드가 아니라 메타데이터 조합별 주차별 median 추이
                 meta_df_root, 
                 item_number_list,
                 sales_total_len, 
                 trend_len,
                 sales_transform,
                 n_neighbors,
                 img_emb_root, 
                 text_emb_root,
                 distance_sorted_root,
                 local_savepath,
                 pred_sampling=None,
                 pop_root=None,
                 ):
        """
                ZeroShotDataset 초기화 메서드.
                Args:
                    sales_df_root (str): 판매 데이터 경로.
                    n_trends_root (str): ntrends 데이터 경로.
                    meta_df_root (str): 메타데이터 경로.
                    item_number_list (list): 아이템 번호 리스트.
                    sales_total_len (int): 판매 데이터 총 길이.
                    trend_len (int): 트렌드 데이터 길이.
                    sales_transform (callable): 판매 데이터 변환 함수.
                    n_neighbors (int): k-nearest neighbors 개수.
                    img_emb_root (str): 이미지 임베딩 경로.
                    text_emb_root (str): 텍스트 임베딩 경로.
                    distance_sorted_root (str or None): ERP distance 데이터 경로. 없으면 None.
                    local_savepath (str or None): 저장 경로. 없으면 None.
                    pred_sampling (callable or None): 예측 샘플링 함수. 없으면 None.
                    pop_root (str or None): Pop signal 데이터 경로. 없으면 None.
                """

        self.sales_df_root = sales_df_root
        self.n_trends_root = n_trends_root #수정된 부분 - 기존에는 3개를 받았으나 이젠 1개만 받음
        self.meta_df_root = meta_df_root
        self.item_number_list = item_number_list

        self.sales_total_len = sales_total_len
        self.trend_len = trend_len


        self.sales_transform = sales_transform

        self.past_trend_len = trend_len - sales_total_len

        self.img_root = "/home/sflab/SFLAB/DATA/mind_br_data_full_240227/images"
        self.img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.prepo_data_folder = "/home/sflab/SFLAB/DATA/mind_br_data_240916/"
        self.data_folder = "/home/sflab/SFLAB/DATA/mind_br_data_full_240227/"

        self.n_neighbors = n_neighbors

        self.img_emb_root = img_emb_root
        self.text_emb_root = text_emb_root

        self.distance_sorted_root = distance_sorted_root

        self.local_savepath = local_savepath # 저장 경로
        self.pred_sampling = pred_sampling 
        self.pop_root = pop_root # Pop signal 경로
        
        # 데이터 로드 메서드 호출
        self.__read_data__() 

    def __read_data__(self): #데이터 읽기 메서드. 판매 데이터, 메타데이터, 이미지/텍스트 임베딩 등을 로드함
        
        # 판매 데이터 로드 및 필터링
        sales_df = pd.read_csv(os.path.join(self.prepo_data_folder, self.sales_df_root), index_col="item_number_color")  # 21659개 품번
        self.sales_df = sales_df[(pd.Series(sales_df.index).isin(self.item_number_list)).values]

        #수정된 부분 - median 12주 추이 불러오기(1개 데이터)
        self.n_trends = pickle.load(open(os.path.join(self.prepo_data_folder, self.n_trends_root), 'rb'))
        print("Loaded n_trends structure:", {k: np.array(v).shape for k, v in list(self.n_trends.items())[:5]})

        # 메타데이터 로드 및 필터링
        meta_df = pd.read_csv(os.path.join(self.prepo_data_folder, self.meta_df_root), index_col='item_number_color')
        if 'sales_mean' in meta_df.columns:
            meta_df = meta_df.drop(columns=['sales_mean', 'sales_std', 'sales_total', 'item_number'])
        if 'main_color_nan' in meta_df.columns:
            meta_df = meta_df.drop(columns=['main_color_nan'])
        meta_df['brand'] = [0 if idx[0] == 'J' else 1 for idx in meta_df.index]
        self.meta_df = meta_df[(pd.Series(meta_df.index).isin(self.item_number_list)).values]
        self.meta_df = self.meta_df.loc[~self.meta_df.index.duplicated(keep='first')]

        # 이미지 및 텍스트 임베딩 로드
        self.img_emb = pickle.load(open(os.path.join(self.prepo_data_folder, self.img_emb_root), 'rb'))
        self.text_emb = pickle.load(open(os.path.join(self.prepo_data_folder, self.text_emb_root), 'rb'))

        # Pop signal 로드 (옵션)
        if self.pop_root:
            pop_signal_files = [pickle.load(open(os.path.join(self.pop_root, f"v{i}/pop_idx.pickle"), 'rb')) for i in range(1, 5)]
            self.pop_signal = {k: v for d in pop_signal_files for k, v in d.items()}
        else:
            self.pop_signal = None
        # self.pop_signal = {k: v for d in [pickle.load(open(os.path.join(self.pop_root, f"v{i}/pop_idx.pickle"), 'rb')) for i in range(1,5)] for k, v in d.items()}
        
        # ERP distance 행렬 로드 (옵션)
        if self.distance_sorted_root:
            self.distance_sorted_mat = np.load(os.path.join('/home/sflab/SFLAB/sanguk/mind_br_data_prepro_full/240731_prepro/', self.distance_sorted_root))
        else:
            self.distance_sorted_mat = None
        # self.distance_sorted_mat = np.load(os.path.join('/home/sflab/SFLAB/sanguk/mind_br_data_prepro_full/240731_prepro/', self.distance_sorted_root))
        

        train_list = pickle.load(open(os.path.join(self.prepo_data_folder, "train_list_240916.pkl"), 'rb'))
        self.memory_sales_df = sales_df.loc[train_list]
        self.loaded_data = self.get_loader_shuffle()


    def preprocess_data(self):
        item_sales_list, temporal_features_list, ntrends_list, images_list, texts_list, \
            real_value_sales_list, release_dates_list, \
            meta_data_list, k_item_sales_list_list, pop_signal_list = [], [], [], [], [], [], [], [], [], []

        #변경된 부분 - 단일 데이터로 대체
        for idx in tqdm(range(len(self.sales_df))):
            
            item_number_color = self.sales_df.iloc[idx]._name

            # n_trends 데이터 로드 및 처리
            if item_number_color in self.n_trends.keys():
                ntrend_raw = np.array(self.n_trends[item_number_color]).reshape(-1, 1)  # 12x1 데이터
                # print("Original ntrend_raw shape:", ntrend_raw.shape)
                # 정수형 텐서로 변환 (스케일링 적용 안함)
                ntrend = torch.tensor(ntrend_raw, dtype=torch.int64)
                # print("Sales values before Fourier transform:", ntrend)
                # Fourier Mapping 적용
                ntrend = self.sales_transform.transform(ntrend.squeeze().long())  # (12,) -> Fourier mapped (12, D)
                # print("After Fourier transform shape:", ntrend.shape)
                
            else:
                ntrend = torch.zeros(12, self.sales_transform.f_pe.shape[1])  # 결측값 처리 - (12, D)

            # Batch 차원 추가
            ntrend = ntrend.unsqueeze(0)  # (1, 12, D)
            # print("Final ntrend shape after unsqueeze:", ntrend.shape)

            # 이미지 임베딩 처리
            if item_number_color in self.img_emb.keys():
                images = torch.FloatTensor(self.img_emb[item_number_color])
            else:
                images = torch.zeros(512)

            # 텍스트 임베딩 처리
            if item_number_color[:-2] in self.text_emb.keys():
                texts = torch.FloatTensor(self.text_emb[item_number_color[:-2]])
            else:
                texts = torch.zeros(512)

            # 판매 데이터 처리
            row = self.sales_df.iloc[idx].iloc[1:].values
            real_value_sales = torch.FloatTensor(np.array(row, dtype='int'))
            
            try:
                item_sales = self.sales_transform.transform(torch.tensor(np.array(row, dtype='int'), dtype=torch.int64))
            except Exception as e:
                print(f"Error processing sales data for {item_number_color}: {e}")
                item_sales = torch.zeros_like(real_value_sales)

            # 출시일 데이터 처리
            release_date = pd.to_datetime(self.sales_df.iloc[idx]['release_date'])
            release_dates_tensor = torch.FloatTensor([release_date.year, release_date.month, release_date.day])

            # 메타데이터 처리
            meta_data = torch.FloatTensor(self.meta_df.loc[item_number_color].values)

            # Temporal Features 생성 (주별, 월별, 연도별)
            time_feature_range = pd.date_range(release_date - relativedelta(weeks=52), release_date, freq='7d')
            temporal_features = [time_features(time_feature_range, freq='w')[0].tolist(),
                                time_features(time_feature_range, freq='m')[0].tolist(),
                                time_features(time_feature_range, freq='y')[0].tolist()]
            temporal_features_tensor = torch.FloatTensor(temporal_features)

            
            # K-Nearest Item Sales 데이터 처리
            if self.distance_sorted_mat is not None:
                k_nearest_idx = self.distance_sorted_mat[idx][:self.n_neighbors]
                k_item_sales_list = []
                for k_item in k_nearest_idx:
                    row = self.memory_sales_df.iloc[k_item]
                    k_item_sales_list.append(
                        self.sales_transform.transform(torch.tensor(np.array(row.iloc[1:].values, dtype='int'), dtype=torch.int64))
                    )
                k_item_sales = torch.stack(k_item_sales_list).reshape(-1, 512)
            else:
                k_item_sales = torch.zeros(self.n_neighbors * 12, 512)
            # k_nearest_idx = self.distance_sorted_mat[idx][:self.n_neighbors]
            # k_item_sales_list = []
            # for k_item in k_nearest_idx:
            #     row = self.memory_sales_df.iloc[k_item]
            #     k_item_sales_list.append(
            #         self.sales_transform.transform(torch.tensor(np.array(row.iloc[1:].values, dtype='int'), dtype=torch.int64))
            #     )
            # k_item_sales = torch.stack(k_item_sales_list).reshape(-1, 512)

            # Pop Signal 처리
            if self.pop_signal and item_number_color in self.pop_signal:
                pop_signal = torch.FloatTensor(self.pop_signal[item_number_color])
            else:
                pop_signal = torch.zeros(52)
            # if item_number_color in self.pop_signal.keys():
            #     pop_signal = torch.FloatTensor(self.pop_signal[item_number_color])
            # else:
            #     pop_signal = torch.zeros(52)

            # 리스트에 추가
            item_sales_list.append(item_sales)
            temporal_features_list.append(temporal_features_tensor)
            ntrends_list.append(ntrend)
            images_list.append(images)
            texts_list.append(texts)
            real_value_sales_list.append(real_value_sales)
            release_dates_list.append(release_dates_tensor)
            meta_data_list.append(meta_data)
            k_item_sales_list_list.append(k_item_sales)
            pop_signal_list.append(pop_signal)


        # TensorDataset 생성 직전에 디버깅 print 추가
        # print("Final ntrends shapes:", [t.shape for t in ntrends_list[:5]])
        # print("Stacked ntrends shape:", torch.stack(ntrends_list).shape)
        return TensorDataset(
            torch.stack(item_sales_list),
            torch.stack(temporal_features_list),
            torch.stack(ntrends_list),
            torch.stack(images_list),
            torch.stack(texts_list),
            torch.stack(real_value_sales_list),
            torch.stack(release_dates_list),
            torch.stack(meta_data_list),
            torch.stack(k_item_sales_list_list),
            torch.stack(pop_signal_list)
        )

    # def __getitem__(self, idx):
    #     idx_data = list(self.loaded_data[idx])
    #     k_item_sales = idx_data[-1]
    #     k_item_sales_orig = k_item_sales.reshape(10, 12, 512)
    #     idx_data[-1] = torch.index_select(k_item_sales_orig, dim=0, index=torch.randperm(10)).reshape(-1, 512)
    #     return tuple(idx_data)

    def __getitem__(self, idx):
        idx_data = self.loaded_data[idx]
        
        if self.distance_sorted_mat is not None:
            k_nearest_idx = self.distance_sorted_mat[idx][:self.n_neighbors]
            k_item_sales_list = []
            for k_item in k_nearest_idx:
                row = self.memory_sales_df.iloc[k_item]
                row = row.iloc[1:].values
                k_item_sales_list.append(self.sales_transform.transform(torch.tensor(np.array(row, dtype='int'), dtype=torch.int64)))
            k_item_sales = torch.stack(k_item_sales_list).reshape(-1, 512)
            
            idx_data = list(idx_data)
            idx_data[-2] = k_item_sales
            idx_data = tuple(idx_data)
        
        return idx_data

        # k_nearest_idx = self.distance_sorted_mat[idx][:self.n_neighbors]
        # k_item_sales_list = []
        # for k_item in k_nearest_idx:
        #     # row = self.train_sales_df.loc[k_item]
        #     row = self.memory_sales_df.iloc[k_item]
        #     row = row.iloc[1:].values
        #     k_item_sales_list.append(self.sales_transform.transform(torch.tensor(np.array(row, dtype='int'), dtype=torch.int64)))
        #     # k_item_sales_list.append(torch.FloatTensor(StandardScaler().fit_transform(np.array(row, dtype='int').reshape(-1, 1))).reshape(12))
        # k_item_sales = torch.stack(k_item_sales_list).reshape(-1, 512)
        # idx_data = list(idx_data)

        # idx_data[-2] = k_item_sales


        # idx_data = tuple(idx_data)
        # return idx_data

    def get_loader_shuffle(self):
        # data_with_gtrends = self.preprocess_data()

        print("Loading dataset...")
        if os.path.isfile(self.local_savepath):
            data_with_gtrends = torch.load(self.local_savepath)  # load dataset directly from saved files
        else:
            print('Starting dataset creation process...')
            data_with_gtrends = self.preprocess_data()
            torch.save(data_with_gtrends, self.local_savepath)
        print("loading dataset...Done.")

        return data_with_gtrends

    # def get_loader(self, batch_size, shuffle, num_workers):
    #     # data_with_gtrends = self.preprocess_data()
    #
    #     print("Loading dataset...")
    #     if os.path.isfile(self.local_savepath):
    #         data_with_gtrends = torch.load(self.local_savepath)  # load dataset directly from saved files
    #     else:
    #         print('Starting dataset creation process...')
    #         data_with_gtrends = self.preprocess_data()
    #         torch.save(data_with_gtrends, self.local_savepath)
    #     print("loading dataset...Done.")
    #
    #     data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #
    #     return data_loader

    def __len__(self):
        return len(self.sales_df)


