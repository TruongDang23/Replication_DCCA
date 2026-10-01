# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:30:50 2020

@author: chunmanzuo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scanpy as sc
import os
import time
import random
import argparse
import scipy as sp
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import optim

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import mixture
from scipy.io import mmread
import operator

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.model_selection import train_test_split

def setup_seed( seed ):
    # Fix all RNG sources: python, numpy, torch CPU/GPU; force deterministic cuDNN
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    try:
        # requires CUBLAS_WORKSPACE_CONFIG=:4096:8 to be set in the environment
        torch.use_deterministic_algorithms( True, warn_only = True )
    except ( AttributeError, TypeError ):
        pass

def parameter_setting():
    
    parser = argparse.ArgumentParser(description='Single cell Multi-omics data analysis')
    
    outPath = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/new_test/'
    
    parser.add_argument('--File1', '-F1', type=str, default = 'scRNA_seq_SNARE.tsv',    help='input file name1')
    parser.add_argument('--File2', '-F2', type=str, default = 'scATAC_seq_SNARE.txt', help='input file name2')
    parser.add_argument('--File2_1', '-F2_1', type=str, default = 'scATAC_seq_SNARE.txt', help='input file name2_1')

    parser.add_argument('--File3', '-F3', type=str, default = '5-cellinfo-RNA.tsv',  help='input meta file')
    parser.add_argument('--File_combine', '-F_com', type=str, default = 'Gene_chromatin_order_combine.tsv',    help='input combine file name')
    
    parser.add_argument('--workdir', '-wk', type=str, default = outPath, help='work path')
    parser.add_argument('--outdir', '-od', type=str, default = outPath, help='Output path')
    
    parser.add_argument('--lr1', type=float, default = 0.01, help='Learning rate1')
    parser.add_argument('--flr1', type=float, default = 0.001, help='Final learning rate1')
    parser.add_argument('--lr2', type=float, default = 0.002, help='Learning rate2')
    parser.add_argument('--flr2', type=float, default = 0.0002, help='Final learning rate2')
    parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default = 0.01, help='eps')

    parser.add_argument('--sf1', type=float, default = 2.0, help='scale_factor_1 for supervision signal from scRNA-seq')
    parser.add_argument('--sf2', type=float, default = 1.0, help='scale_factor_2 for supervision signal from scEpigenomics')
    parser.add_argument('--cluster1', '-clu1', type=int, default=2, help='predefined cluster for scRNA')
    parser.add_argument('--cluster2', '-clu2', type=int, default=2, help='predefined cluster for other epigenomics')
    parser.add_argument('--geneClu', '-gClu', type=list, default = None, help='predefined gene cluster for scRNA')
    
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
    
    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
    parser.add_argument('--max_epoch', '-me', type=int, default=500, help='Max epoches')
    parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
    parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default=10, help='Epoch per test')
    parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')
    
    return parser

def read_dataset( File_RNA = None, File_ATAC = None, test_size_prop = 0.15, state = 0, 
                  top_n_rna = 1000, top_n_atac = 1000 ):
    # read single-cell multi-omics data together
    """
    Đọc dữ liệu multimodal từ file .h5ad (đã preprocessing, dạng cells x features).
    
    Parameters
    ----------
    File_RNA  : str - đường dẫn tới file RNA.h5ad
    File_ATAC : str - đường dẫn tới file ATAC.h5ad
    test_size_prop : float - tỉ lệ tập test (mặc định 0.15)

    Returns
    -------
    adata, adata1, adata2, train_idx, test_idx, label_ground_truth
    (adata2 = None vì chỉ có 2 modality)
    """

    # ── 1. Đọc file .h5ad (đã là cells × features, không cần transpose) ──────
    adata  = sc.read(File_RNA)   # RNA
    adata1 = sc.read(File_ATAC)  # ATAC
    print(f"Đọc xong — RNA: {adata.shape} | ATAC: {adata1.shape}")

    # Feature selection RNA: HVG (seurat_v3) ────────────────────────────
    print(f"\n[RNA] Chọn top {top_n_rna} HVGs (seurat_v3) — shape ban đầu: {adata.shape}")

    # ── 2. Bộ lọc Đặc trưng cho scRNA-seq (HVG) ──────────────────────────────
    if 'highly_variable' in adata.var.columns:
        print(f"[RNA] Tìm thấy cột highly_variable sẵn có.")
        if adata.var['highly_variable'].sum() > top_n_rna and 'dispersions_norm' in adata.var.columns:
            hvg_subset = adata.var[adata.var['highly_variable'] == True]
            top_genes = hvg_subset.nlargest(top_n_rna, 'dispersions_norm').index
            adata = adata[:, top_genes].copy()
        else:
            # Nếu không có cột dispersion hoặc số lượng vừa bằng, lấy các gene được đánh dấu True
            # Nếu số lượng gene lớn hơn top_n_rna mà không có dispersion, lấy top_n_rna gene đầu tiên của tập True
            true_genes = adata.var_names[adata.var['highly_variable'] == True]
            adata = adata[:, true_genes[:top_n_rna]].copy()
    else:
        print(f"[Warning] Không thấy nhãn highly_variable trong RNA, tự động lấy {top_n_rna} gen đầu.")
        adata = adata[:, :top_n_rna].copy()
    
    print(f"[RNA] Sau khi lọc lấy đặc trưng biến thiên: {adata.shape}")

    # Feature selection ATAC: top peaks by total count ──────────────────
    print(f"\n[ATAC] Chọn top {top_n_atac} peaks by total count — shape ban đầu: {adata1.shape}")
    if 'highly_variable' in adata1.var.columns:
        print(f"[ATAC] Tìm thấy cột highly_variable từ bước xử lý phương sai nhị phân.")
        
        # Sắp xếp và lọc lấy đúng top_n_atac dựa trên cột phương sai thủ công 'binary_variance' mà ta đã lưu
        if 'binary_variance' in adata1.var.columns:
            hvp_subset = adata1.var[adata1.var['highly_variable'] == True]
            top_peaks = hvp_subset.nlargest(top_n_atac, 'binary_variance').index
            adata1 = adata1[:, top_peaks].copy()
        else:
            # Cắt lấy đúng số lượng cột yêu cầu để tránh lệch chiều mạng Neural
            true_peaks = adata1.var_names[adata1.var['highly_variable'] == True]
            adata1 = adata1[:, true_peaks[:top_n_atac]].copy()
    else:
        # Nhánh dự phòng an toàn bằng phương sai nhị phân (chứ không dùng tổng sum nữa)
        print(f"[Warning] Không tìm thấy cột highly_variable trong ATAC. Tính lại phương sai nhị phân...")
        X_binary = (adata1.X > 0).astype(np.float32)
        p = np.array(X_binary.mean(axis=0)).flatten()
        variances = p * (1.0 - p)
        top_indices = np.argpartition(variances, -top_n_atac)[-top_n_atac:]
        adata1 = adata1[:, top_indices].copy()

    # ── 2. Lấy ground truth từ cột cell_type ─────────────────────────────────
    if 'cell_type' in adata.obs.columns:
        label_ground_truth = adata.obs['cell_type'].values.tolist()
    else:
        print("[Warning] Không tìm thấy cột 'cell_type' trong adata.obs → gán nhãn 1 cho tất cả.")
        label_ground_truth = np.ones(adata.n_obs)

    # ── 3. Train / Test split ─────────────────────────────────────────────────
    if test_size_prop > 0:
        train_idx, test_idx = train_test_split(
            np.arange(adata.n_obs),
            test_size=test_size_prop,
            random_state=200
        )
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
    else:
        train_idx = list(range(adata.n_obs))
        test_idx  = list(range(adata.n_obs))
        spl = pd.Series(['train'] * adata.n_obs)

    # ── 4. Gán metadata split + Group vào obs ─────────────────────────────────
    for ad in [adata, adata1]:
        ad.obs['split'] = spl.values
        ad.obs['split'] = ad.obs['split'].astype('category')
        ad.obs['Group'] = label_ground_truth   # giữ tên 'Group' để tương thích với pipeline gốc
        ad.obs['Group'] = ad.obs['Group'].astype('category')
	
    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    
    ### here, adata with cells * features
    return adata, adata1, train_idx, test_idx, label_ground_truth, None

def normalize( adata, filter_min_counts=True, size_factors=True, 
               normalize_input=False, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if logtrans_input:
        sc.pp.log1p(adata)

    if size_factors:
        #adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        adata.obs['size_factors'] = np.log( np.sum( adata.X, axis = 1 ) )
    else:
        adata.obs['size_factors'] = 1.0

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def calculate_log_library_size( Dataset ):
    ### Dataset is raw read counts, and should be cells * features
    Nsamples     =  np.shape(Dataset)[0]
    library_sum  =  np.log( np.sum( Dataset, axis = 1 ) )
    
    lib_mean     =  np.full( (Nsamples, 1), np.mean(library_sum) )
    lib_var      =  np.full( (Nsamples, 1), np.var(library_sum) ) 
    
    return lib_mean, lib_var

def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

    lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr   


def save_checkpoint(model, filename='model_best.pth.tar' ):

    torch.save( model.state_dict(), filename )


def load_checkpoint(file_path, model, use_cuda=False):

    if use_cuda:
        device = torch.device( "cuda" )
        model.load_state_dict( torch.load(file_path) )
        model.to(device)
        
    else:
        device = torch.device('cpu')
        model.load_state_dict( torch.load(file_path, map_location=device) )
    model.eval()

    return model

def Normalized_0_1 (Data):
    ## here ,Data is cell * genes
    adata = sc.AnnData( Data )
    sc.pp.normalize_per_cell( adata, counts_per_cell_after=1,
                              key_n_counts='n_counts2' ) 
    return adata

def estimate_cluster_numbers(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (n,p) matrix, n is feature, p is sample
    """
    n, p = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()

    x             = scale(data) # normalization for each sample
    muTW          = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW       = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd    = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k

def pearson(data):
    print('Start for pearson similarity ')
    df    = pd.DataFrame(data.T)
    pear_ = df.corr(method='pearson')
    return np.where(pear_>=0, pear_, np.zeros(shape=(pear_.shape)))

def LaplacianScore(x, w):
    # x in (samples, features)
    n_samples, n_feat = x.shape[0], x.shape[1]

    if w.shape[0] != n_samples:
        raise Exception("W.shape not match X.shape")

    D = np.diag(np.sum(w, axis=1)) # (n_samples,)
    D2 = np.sum(w, axis=1) # (n_samples,)
    L = w

    tmp1 = (D2.T).dot(x)
    DPrime = np.sum((x.T.dot(D)).T * x, axis=0) - tmp1 * tmp1/np.sum(D2)
    LPrime = np.sum((x.T.dot(L)).T * x, axis=0) - tmp1 * tmp1/np.sum(D2)

    DPrime = eps2C(DPrime, c=10000)
    a1=np.sum(D)
    a2=np.sum((x.T.dot(D)).T * x, axis=0)
    a3=tmp1 * tmp1/np.sum(D)
    a4=(x.T.dot(D)).T * x
    a7=((x.T).dot(D)).T * x
    a5=tmp1 * tmp1
    a6=x.T.dot(D)
    a9=np.dot(x.T,D)

    Y = LPrime / DPrime
    #Y = Y.T#lzl edit
    return Y

def Feature_selection_by_Laplace( fileName, outName):
    ### here, fileName is sample * feature
    print('For similarity pearson')

    Data         = pd.read_table( fileName, header=0, index_col=0 )
    pear_sim0    = pearson(Data)
    pear_score   = LaplacianScore(Data, pear_sim0)

    return pear_score
