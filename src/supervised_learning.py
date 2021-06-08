#!/usr/bin/env python3

import dgl
import numpy as np
import time
import torch
from utils import build_graph, build_predict_graph, Identical
from dgl.nn.pytorch.conv import GraphConv
from ogb.lsc import WikiKG90MEvaluator


def concat_src_dst(edges):
    return {"edge_emb": torch.cat((edges.src['x'], edges.dst['x']), dim=1)}


def u_sub_v(edges):
    return {"edge_emb": edges.src['x'] - edges.dst['x']}


class GCNNet(torch.nn.Module):

    def __init__(self, num_rels, hidden_dims, out_dim, rel_emb_dim=32, device='cpu', feat_dim=0):
        super(GCNNet, self).__init__()
        self.device = device
        self.rel_emb = torch.nn.Embedding(num_rels, rel_emb_dim).to(device)
        self.out_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.in_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.gcn_layers = torch.nn.ModuleList(
            [GraphConv(rel_emb_dim*2, hidden_dims[0]).to(device)]
        )
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self.gcn_layers.append(
                GraphConv(size, hidden_dims[layer_idx+1]).to(device)
            )
        self.fc = torch.nn.Linear(in_features=hidden_dims[-1]+feat_dim, out_features=out_dim).to(device)
        self.ident = Identical().to(device)

    def forward(self, data, with_node_feat=False):
        input_nodes, input_feat, edge_subg, blocks = data
        out_inc = blocks[0].inc('out').to(self.device)
        in_inc = blocks[0].inc('in').to(self.device)
        out_edge_emb = torch.sparse.mm(
            out_inc,
            self.out_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        ).div(torch.sparse.sum(out_inc, dim=1).to_dense().unsqueeze(-1) + 1e-8)
        in_edge_emb = torch.sparse.mm(
            in_inc,
            self.in_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        ).div(torch.sparse.sum(in_inc, dim=1).to_dense().unsqueeze(-1) + 1e-8)
        out_edge_emb = torch.nn.functional.pad(
            out_edge_emb,
            (0, 0, 0, input_nodes.size(0)-out_edge_emb.size(0))
        )
        in_edge_emb = torch.nn.functional.pad(
            in_edge_emb,
            (0, 0, 0, input_nodes.size(0)-in_edge_emb.size(0))
        )
        x = torch.cat(
            (out_edge_emb, in_edge_emb),
            dim=1
        )
        x = self.ident(blocks[0], x)
        for layer, block in zip(self.gcn_layers, blocks[1:]):
            x = torch.relu(layer(block, x))
        if with_node_feat:
            x = torch.cat((x, input_feat), dim=1)
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(u_sub_v)
            return self.fc(edge_subg.edata['edge_emb'])


def train(g, train_eid=None, fanouts=(6, 6, 6), hidden_dims=(64, 32), rel_emb_dim=32, batch_size=256, device='cpu'):
    if train_eid is None:
        train_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, sampler, device=device,
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        hidden_dims=hidden_dims,
        out_dim=1315,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    opt = torch.optim.Adam(gcn.parameters())
    batch_idx = 0
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        labels = edge_subg.edata['type']
        preds = gcn((input_nodes, input_nodes, edge_subg, blocks))
        loss = torch.nn.functional.cross_entropy(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if batch_idx % 500 == 0:
            acc = (preds.argmax(dim=1) == labels).sum() / preds.size(0)
            print(time.ctime(), "%d-th mini batch, loss: %s, accuracy: %s" % (batch_idx, loss.item(), acc.item()))
            torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised/gcn_weight.pth.%d"%batch_idx)
    torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised/gcn_weight.pth.done")


def train_with_node_feat(g, train_eid=None, fanouts=(6, 6, 6), hidden_dims=(128, 64), rel_emb_dim=32, batch_size=256, device='cpu'):
    if train_eid is None:
        train_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, sampler, device=device,
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        hidden_dims=hidden_dims,
        out_dim=1315,
        rel_emb_dim=rel_emb_dim,
        device=device,
        feat_dim=768
    )
    opt = torch.optim.Adam(gcn.parameters())
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    batch_idx = 0
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        labels = edge_subg.edata['type']
        input_feat = node_feat[blocks[-1].dstdata[dgl.NID].cpu()].type(torch.float32).to(device)
        preds = gcn((input_nodes, input_feat, edge_subg, blocks), with_node_feat=True)
        loss = torch.nn.functional.cross_entropy(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if batch_idx % 500 == 0:
            acc = (preds.argmax(dim=1) == labels).sum() / preds.size(0)
            print(time.ctime(), "%d-th mini batch, loss: %s, accuracy: %s" % (batch_idx, loss.item(), acc.item()))
            torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised_with_node_feat/gcn_weight.pth.%d"%batch_idx)
    torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised_with_node_feat/gcn_weight.pth.done")


@torch.no_grad()
def predict(model_path,
            fanouts=(6, 6, 6),
            hidden_dims=(64, 32),
            rel_emb_dim=32,
            batch_size=1001,
            max_batch=10000,
            skip_rows=0,
            device='cpu'):
    g = build_predict_graph(max_batch=max_batch, skip_rows=skip_rows)
    pred_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, pred_eid, sampler, device=device, g_sampling=build_graph(),
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        hidden_dims=hidden_dims,
        out_dim=1315,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    gcn.load_state_dict(torch.load(model_path, map_location=device))
    probs = []
    batch_idx = 0
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        prob = torch.softmax(
            gcn((input_nodes, input_nodes, edge_subg, blocks)),
            dim=1
        )[(torch.arange(edge_subg.num_edges()), edge_subg.edata['type'])]
        probs.append(prob)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
    preds = torch.cat(probs, dim=0).reshape(-1, batch_size).cpu().numpy()
    t_pred_top10 = np.argsort(-preds, axis=1)[:,:10]
    t_correct_index = np.load(
        "/data/wikikg90m_kddcup2021/processed/val_t_correct_index.npy")[skip_rows:skip_rows+max_batch]
    evaluator = WikiKG90MEvaluator()
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}
    return evaluator.eval(input_dict)


@torch.no_grad()
def predict_with_node_feat(model_path,
            fanouts=(6, 6, 6),
            hidden_dims=(128, 64),
            rel_emb_dim=32,
            batch_size=1001,
            max_batch=10000,
            skip_rows=0,
            device='cpu'):
    g = build_predict_graph(max_batch=max_batch, skip_rows=skip_rows)
    pred_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, pred_eid, sampler, device=device, g_sampling=build_graph(),
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        hidden_dims=hidden_dims,
        out_dim=1315,
        rel_emb_dim=rel_emb_dim,
        device=device,
        feat_dim=768
    )
    gcn.load_state_dict(torch.load(model_path, map_location=device))
    probs = []
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    batch_idx = 0
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        input_feat = node_feat[blocks[-1].dstdata[dgl.NID].cpu()].type(torch.float32).to(device)
        prob = torch.softmax(
            gcn((input_nodes, input_feat, edge_subg, blocks), with_node_feat=True),
            dim=1
        )[(torch.arange(edge_subg.num_edges()), edge_subg.edata['type'])]
        probs.append(prob)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
    preds = torch.cat(probs, dim=0).reshape(-1, batch_size).cpu().numpy()
    t_pred_top10 = np.argsort(-preds, axis=1)[:,:10]
    t_correct_index = np.load(
        "/data/wikikg90m_kddcup2021/processed/val_t_correct_index.npy")[skip_rows:skip_rows+max_batch]
    evaluator = WikiKG90MEvaluator()
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}
    return evaluator.eval(input_dict)


if __name__ == "__main__":
    train(build_graph(), batch_size=1024, device='cuda:7')
    # train_with_node_feat(build_graph(), batch_size=1024, device='cuda:6')