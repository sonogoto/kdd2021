#!/usr/bin/env python3

import pandas as pd
import torch
import dgl
import numpy as np
from scipy.sparse import csr_matrix
import pickle as pk


def build_graph(remove_nodes=False, return_incidence=False):
    g = dgl.DGLGraph()
    h, r, t = np.load("/data/wikikg90m_kddcup2021/processed/train_hrt.npy").T
    g.add_edges(
        torch.from_numpy(h),
        torch.from_numpy(t),
        data={"type": torch.from_numpy(r)})
    if remove_nodes:
        with open("/home/liusx/wiki90/nodes2rm.pkl", "rb") as fr:
            nodes2rm = pk.load(fr)
            g.remove_nodes(nodes2rm)
    g = dgl.add_self_loop(g)
    if return_incidence:
        nnz = h.shape[0]
        data = np.ones(nnz, dtype=np.int16)
        in_inc = csr_matrix((data, (h, r)), shape=(87143637, 1315), dtype=np.int16)
        # in_inc_norm = np.nan_to_num(1./in_inc.sum(axis=1), posinf=0.)
        out_inc = csr_matrix((data, (t, r)), shape=(87143637, 1315), dtype=np.int16)
        # out_inc_norm = np.nan_to_num(1./out_inc.sum(axis=1), posinf=0.)
        return g, in_inc, out_inc
    return g


def build_inc():
    df = pd.read_csv("/data/wikikg90m_kddcup2021/processed/train_hrt.txt",
                     header=None, sep=" ", names=["h","r","t"])
    h, r, t = df.h.to_numpy(np.int64), df.r.to_numpy(np.int64), df.t.to_numpy(np.int64)
    nnz = h.shape[0]
    data = np.ones(nnz, dtype=np.int16)
    in_inc = csr_matrix((data, (h, r)), shape=(87143637, 1315), dtype=np.int16)
    out_inc = csr_matrix((data, (t, r)), shape=(87143637, 1315), dtype=np.int16)
    return in_inc, out_inc


def load_norm_inc():
    with open("/home/liusx/wiki90/in_inc.pkl", "rb") as fr:
        in_inc = pk.load(fr)
    with open("/home/liusx/wiki90/out_inc.pkl", "rb") as fr:
        out_inc = pk.load(fr)
    return in_inc, out_inc


def build_predict_graph(max_batch, skip_rows=0):
    h, r = np.loadtxt("/data/wikikg90m_kddcup2021/processed/val_hr.txt",
                      delimiter=" ",
                      dtype=np.int64,
                      max_rows=max_batch,
                      skiprows=skip_rows
                      ).T
    repeat = np.ones(h.shape[0], dtype=np.int32) + 1000
    h = np.repeat(h, repeat)
    r = np.repeat(r, repeat)
    t_candidate = np.load(
        "/data/wikikg90m_kddcup2021/processed/val_t_candidate.npy"
    )[skip_rows:skip_rows+max_batch].reshape(-1)
    g = dgl.DGLGraph()
    g.add_edges(
        torch.from_numpy(h),
        torch.from_numpy(t_candidate),
        data={"type": torch.from_numpy(r)})
    return g


class Identical(torch.nn.Module):

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            _, feat_dst = dgl.utils.expand_as_pair(feat, graph)
            return feat_dst
