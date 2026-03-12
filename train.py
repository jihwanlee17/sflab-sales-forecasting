import os
import argparse
import wandb
import torch
import pandas as pd
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from models.ANTM_all_forecast_fourier_nt_erp_pop_jihwan import GTM
from utils.data_multitrends_all_forecast_fourier_nt_erp_pop_new_split_jihwan import ZeroShotDataset
from torch.utils.data import DataLoader
from utils.scaling_method import Fourier_transform_pos

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['CURL_CA_BUNDLE'] = ''

wandb.login(key='65f58319d77e59bf2f13edce1380a3a6390dbfee', relogin=True, force=True) #이지환의 wandb api 키
wandb.init(
    project="SFLab_jihwan",  # 사용할 프로젝트명
    entity="ywl9845-yonsei-university",  # 팀 이름 (조직 이름)
    name="experiment_01"  # 원하는 실험 이름
)
torch.autograd.set_detect_anomaly(True)


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    train_list = pickle.load(open(os.path.join(args.prepo_data_folder, "train_list_240916.pkl"), 'rb'))
    val_list = pickle.load(open(os.path.join(args.prepo_data_folder, "val_list_240916.pkl"), 'rb'))
    test_list = pickle.load(open(os.path.join(args.prepo_data_folder, "test_list_240916.pkl"), 'rb'))

    sales_transform = Fourier_transform_pos(args.f_max_len, args.hidden_dim)

    train_dataset = ZeroShotDataset(
                                    sales_df_root=args.sales_df_root,
                                    n_trends_root=args.n_trends_root,
                                    meta_df_root=args.meta_df_root,
                                    item_number_list=train_list,
                                    sales_total_len=args.sales_total_len,
                                    trend_len=args.trend_len,
                                    sales_transform=sales_transform,
                                    n_neighbors=0,  # k-nearest neighbors 사용하지 않음
                                    img_emb_root=args.imb_emb_root,
                                    text_emb_root=args.text_emb_root,
                                    distance_sorted_root=None,  # ERP distance 데이터 제거
                                    local_savepath=args.train_pt,
                                    pop_root=None  # Pop Signal 데이터 제거
                                )
    # train_loader = train_dataset.get_loader(batch_size=args.batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = ZeroShotDataset(
                            sales_df_root=args.sales_df_root,
                            n_trends_root=args.n_trends_root,
                            meta_df_root=args.meta_df_root,
                            item_number_list=val_list,
                            sales_total_len=args.sales_total_len,
                            trend_len=args.trend_len,
                            sales_transform=sales_transform,
                            n_neighbors=0,  # k-nearest neighbors 사용하지 않음
                            img_emb_root=args.imb_emb_root,
                            text_emb_root=args.text_emb_root,
                            distance_sorted_root=None,  # ERP distance 데이터 제거
                            local_savepath=args.val_pt,
                            pop_root=None  # Pop Signal 데이터 제거
                        )
    # val_loader = val_dataset.get_loader(batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_dataset = ZeroShotDataset(
                                sales_df_root=args.sales_df_root,
                                n_trends_root=args.n_trends_root,
                                meta_df_root=args.meta_df_root,
                                item_number_list=test_list,
                                sales_total_len=args.sales_total_len,
                                trend_len=args.trend_len,
                                sales_transform=sales_transform,
                                n_neighbors=0,  # k-nearest neighbors 사용하지 않음
                                img_emb_root=args.imb_emb_root,
                                text_emb_root=args.text_emb_root,
                                distance_sorted_root=None,  # ERP distance 데이터 제거
                                local_savepath=args.test_pt,
                                pop_root=None  # Pop Signal 데이터 제거
                            )
    # test_loader = test_dataset.get_loader(batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = GTM(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        gpu_num=args.gpu_num,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        sales_transform=sales_transform,
        n_neighbors=args.n_neighbors,
    )

    dt_string = datetime.now().strftime("%Y%m%d-%H%M")[2:]
    model_savename = dt_string + '_' + args.model_type
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/' + model_savename,
        filename='---{epoch}---',
        monitor='val_week_given0_ad_smape_gs',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    wandb_logger = WandbLogger(name=model_savename)

    # #기존 팀장님 코드에서 Earlystopping 방법 추가함
    # early_stop_callback = pl.callbacks.EarlyStopping(
    # monitor='val_week_given0_ad_smape_gs',
    # patience=5,
    # mode='min'
    # )

    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs, check_val_every_n_epoch=1,
        logger=wandb_logger, callbacks=[checkpoint_callback], # , early_stop_callback 기존 팀장님 코드에서 Earlystopping 방법 추가함
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f'best_model___{checkpoint_callback.best_model_path}')
    print(f'last_model___{checkpoint_callback.last_model_path}')
    best_model_weight = torch.load(checkpoint_callback.best_model_path)
    last_model_weight = torch.load(checkpoint_callback.last_model_path)

    model.load_state_dict(best_model_weight['state_dict'])
    model.eval()
    trainer.test(model, dataloaders=test_loader, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='/home/sflab/SFLAB/jihwan_folder/log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM',
                        help='Choose between GTM or FCN')
    parser.add_argument('--trend_len', type=int, default=12) #수정함
    parser.add_argument('--num_trends', type=int, default=1) #수정함
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--num_attn_heads', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=2)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='ywl9845')
    parser.add_argument('--wandb_proj', type=str, default='SFLab_jihwan')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    parser.add_argument('--sales_total_len', type=int, default=12)

    parser.add_argument('--n_neighbors', type=int, default=10)

    args = parser.parse_args()

    args.prepo_data_folder = "/home/sflab/SFLAB/DATA/mind_br_data_240916/"
    args.data_folder = "/home/sflab/SFLAB/DATA/mind_br_data_full_240227/"
    args.learning_rate = 0.0001  # 0.001
    args.f_max_len = 1820 + 1

    args.sales_df_root = "total_r_s.csv"
    args.n_trends_root = "all_included(brand&category)3.pkl" # 수정 - 아이템별로 각 주차별로 자신이 해당하는 메타데이터조합의 평균 median값 넣기.
    # args.cat_trend_root = "total_cat_trend_240916.pkl" #네이버 트렌드 관련
    # args.fab_trend_root = "total_fab_trend_240916.pkl" #네이버 트렌드 관련
    # args.col_trend_root = "total_col_trend_240916.pkl" #네이버 트렌드 관련
    args.meta_df_root = "total_meta_data_240916.csv"
    args.imb_emb_root = "total_img_emb_240916.pkl"
    args.text_emb_root = "total_text_emb_240916.pkl"

    # args.train_distance_sorted = 'erp_realvalue_train_distance_sorted.npy'
    # args.val_distance_sorted = 'erp_realvalue_val_distance_sorted.npy'
    # args.test_distance_sorted = 'erp_realvalue_test_distance_sorted.npy'
    
    ######## ERP distance 데이터 ########## - 기존대로 인코더에 데이터 입력하고 싶으면 아래 세 라인의 주석을 제거하시오
    # args.train_distance_sorted = 'erp_pred_dist_argsort/241213-0844_erp_neighbor_cl_maxplus1_groupsize2048_norm2_10epoch_brand_10neighbors_new_split/train_pred_dist_argsort.npy'
    # args.val_distance_sorted = 'erp_pred_dist_argsort/241213-0844_erp_neighbor_cl_maxplus1_groupsize2048_norm2_10epoch_brand_10neighbors_new_split/val_pred_dist_argsort.npy'
    # args.test_distance_sorted = 'erp_pred_dist_argsort/241213-0844_erp_neighbor_cl_maxplus1_groupsize2048_norm2_10epoch_brand_10neighbors_new_split/test_pred_dist_argsort.npy'
##########
    # args.train_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/train_10neighbors_nt_erp_pop_new_split.pt"
    # args.val_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/val_10neighbors_nt_erp_pop_new_split_gt_erp.pt"
    # args.test_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/test_10neighbors_nt_erp_pop_new_split_gt_erp.pt"
    # args.train_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/train_10neighbors_nt_erp_pop_new_split.pt"
    # args.val_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/val_10neighbors_nt_erp_pop_new_split.pt"
    # args.test_pt = "/home/sflab/SFLAB/su_GTM_t/ANTM_MB/dataset_tensor/test_10neighbors_nt_erp_pop_new_split.pt"
    args.train_pt = "/home/sflab/SFLAB/jihwan_folder/ANTM_MB/dataset_tensor/train_bc3300_250210.pt" #데이터셋 생성 후 캐싱(저장)되는 경로
    args.val_pt = "/home/sflab/SFLAB/jihwan_folder/ANTM_MB/dataset_tensor/val_bc3300_250210.pt"     #데이터셋 생성 후 캐싱(저장)되는 경로
    args.test_pt = "/home/sflab/SFLAB/jihwan_folder/ANTM_MB/dataset_tensor/test_bc3300_250210.pt"   #데이터셋 생성 후 캐싱(저장)되는 경로

    ######## Pop Signal 데이터 ##########  - 기존대로 인코더에 데이터 입력하고 싶으면 아래 라인의 주석을 제거하시오
    # args.pop_root = '/home/sflab/SFLAB/yoonjung/mindbridge/pop03/result'
########
    args.model_type = "ANTM_fourier_realvalue_dtw_10neighbors_masking_brand_nt_erp_pop_continue_pop_crossrefer_new_split_dropout_gating_10epoch_median_OnlyTrendEncoder"
    args.epochs = 10 # 300  # 10
    args.batch_size = 128  # 4 / 128
    args.gpu_num = 1

    run(args)
