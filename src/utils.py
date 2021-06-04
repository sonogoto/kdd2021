#!/usr/bin/env python3

import gc
import pandas as pd
import torch
import dgl
import numpy as np
import pickle as pk


def build_graph(remove_nodes=False):
    g = dgl.DGLGraph()
    df = pd.read_csv("/data/wikikg90m_kddcup2021/processed/train_hrt.txt",
                     header=None, sep=" ", names=["h","r","t"])
    g.add_edges(
        torch.from_numpy(df.h.to_numpy(np.int64)),
        torch.from_numpy(df.t.to_numpy(np.int64)),
        data={"type": torch.from_numpy(df.r.to_numpy())})
    del df
    gc.collect()
    if remove_nodes:
        with open("/home/liusx/wiki90/nodes2rm.pkl", "rb") as fr:
            nodes2rm = pk.load(fr)
            g.remove_nodes(nodes2rm)
    return dgl.add_self_loop(g)


def build_predict_graph(max_batch):
    h, r = np.loadtxt("/data/wikikg90m_kddcup2021/processed/val_hr.txt",
                      delimiter=" ",
                      dtype=np.int64,
                      max_rows=max_batch
                      ).T
    repeat = np.ones(h.shape[0], dtype=np.int32) + 1000
    h = np.repeat(h, repeat)
    r = np.repeat(r, repeat)
    t_candidate = np.load("/data/wikikg90m_kddcup2021/processed/val_t_candidate.npy")[:max_batch].reshape(-1)
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
