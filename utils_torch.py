
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.stats import rankdata
import scipy.sparse as sp
from sparsesvd import sparsesvd


def calculate_row_correlations(matrix1, matrix2):
    base_value = 1 # 이거하고 랭크정규화 하는게 성능이 가장 좋네 230905 (0으로놓으면 폭망)
    
    num_rows = matrix1.shape[0]
    correlations = np.zeros(num_rows)
    
    for row in range(num_rows):
        nz_indices1 = matrix1.indices[matrix1.indptr[row]:matrix1.indptr[row+1]]
        nz_indices2 = matrix2.indices[matrix2.indptr[row]:matrix2.indptr[row+1]]
        
        common_indices = np.intersect1d(nz_indices1, nz_indices2)
        
        nz_values1 = matrix1.data[matrix1.indptr[row]:matrix1.indptr[row+1]][np.searchsorted(nz_indices1, common_indices)]
        nz_values2 = matrix2.data[matrix2.indptr[row]:matrix2.indptr[row+1]][np.searchsorted(nz_indices2, common_indices)]
        
        if len(common_indices) > 0:
            correlation = np.corrcoef(nz_values1, nz_values2)[0, 1]
            correlations[row] = correlation + base_value 
    
    return correlations



# 정규화 
def corr_normalizer(corr_arr, version = 0):

    # 랭크정규화
    if version == 0:
        ranks = np.apply_along_axis(rankdata, axis=0, arr=corr_arr)
        normalized_corr_arr =  (corr_arr.shape[0] - ranks - 1) / (corr_arr.shape[0] - 1)
    
    # Min-Max 정규화
    elif version == 1:
        # 각 행의 최솟값과 최댓값을 계산합니다.
        min_vals = np.min(corr_arr, axis=1, keepdims=True)
        max_vals = np.max(corr_arr, axis=1, keepdims=True)        
        normalized_corr_arr = (corr_arr - min_vals) / (max_vals - min_vals)
        normalized_corr_arr[np.isnan(normalized_corr_arr)] = 0
        
    # 총합정규화
    # 각 행을 총합으로 나누어 총합이 1이 되도록 정규화합니다.
    elif version == 2:
        row_sums = np.sum(corr_arr, axis=1, keepdims=True)
        normalized_corr_arr = corr_arr / row_sums

    return normalized_corr_arr

def graph_construction(R_tr, n_cri, version = 0):

    # Single graph w/ overall ratings
    if version == 0:
        MCEG = R_tr[0]
    
    # MC Expansion graph construction (ablation with uniform)
    elif version == 1:
        R_tr_dense = []
        R_tr_dense.append(R_tr[0].to_dense())
        for i in range(n_cri-1):
            R_tr[i+1] = R_tr[i+1].to_dense()
            R_tr_dense.append(R_tr[i+1])
        MCEG = torch.hstack(R_tr_dense)
        MCEG = MCEG.to_sparse_csr()
    return MCEG


def freq_filter(s_values, mode = 1, alpha = 0.9, start = 0):
    '''
    input:
    - s_values: singular (eigen) values, list form

    output:
    - filterd_s_values
    '''
    if mode == 0:
        filtered_s_values = s_values
    elif mode == 1:
        filtered_s_values = [(lambda x: 1 / (1 - alpha * x))(v) for v in s_values]
    elif mode ==2:
        filtered_s_values = [(lambda x: 1 / (alpha *x))(v) for v in s_values]
    elif mode ==3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode ==3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == 'band_pass':
        end = start+5
        filtered_s_values = [0] * int(start) + [1] * int(end-start) + [0] * int(len(s_values) - end)
        
    return np.diag(filtered_s_values)



def get_norm_adj(alpha, adj_mat):
    
    # Calculate rowsum and columnsum using PyTorch operations
    rowsum = torch.sum(adj_mat, dim=1)
    colsum = torch.sum(adj_mat, dim=0)
    
    # Calculate d_inv for rows and columns
    d_inv_rows = torch.pow(rowsum, -alpha).flatten()
    d_inv_rows[torch.isinf(d_inv_rows)] = 0.
    d_mat_rows = torch.diag(d_inv_rows)
    
    d_inv_cols = torch.pow(colsum, alpha-1).flatten()
    d_inv_cols[torch.isinf(d_inv_cols)] = 0.
    d_mat_cols = torch.diag(d_inv_cols)
    d_mat_i_inv_cols = torch.diag(1/d_inv_cols)
    
    # Normalize adjacency matrix
    norm_adj = adj_mat.mm(d_mat_rows).mm(adj_mat).mm(d_mat_cols)
    norm_adj = norm_adj.to_sparse()  # Convert to sparse tensor
    
    # Convert d_mat_rows, d_mat_i_inv_cols to sparse tensors
    d_mat_rows_sparse = d_mat_rows.to_sparse()
    d_mat_i_inv_cols_sparse = d_mat_i_inv_cols.to_sparse()
    
    return norm_adj, d_mat_rows_sparse, d_mat_i_inv_cols_sparse

# Example usage
# alpha = ...
# adj_mat = ...
# norm_adj, d_mat_rows, d_mat_i_inv_cols = get_norm_adj(alpha, adj_mat)


# Evaluation functions

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:, :k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx, elm] = 1

    return result, top
    
def precision_k(topk, gt, k):
    '''
    topk, gt: (UXI) array
    k: @k measurement
    '''
    return np.multiply(topk, gt).sum() / (k * len(gt))


def recall_k(topk, gt, k):
    '''
    topk, gt: (UXI) array
    k: @k measurement
    '''
    return np.multiply(topk, gt).sum() / gt.sum()

def ndcg_k(rels, rels_ideal, gt):
    '''
    rels: sorted top-k arr
    rels_ideal: sorted top-k ideal arr
    '''
    k = rels.shape[1]
    n = rels.shape[0]
    dcg =0; idcg = 0
    for row in range(n):
        for col in range(k):
            if gt[row,rels[row, col]] == 1:
                if col==0:
                    dcg += 1
                else:
                    dcg += 1/np.log2(col+1)
            if gt[row,rels_ideal[row, col]] == 1:
                if col==0:
                    idcg += 1
                else:
                    idcg += 1/np.log2(col+1)
    return dcg/idcg

