#!/usr/bin/env python3
import torch
from torch.nn.functional import normalize
from dgl.nn.pytorch.conv import GraphConv
import dgl
import time
import numpy as np
from utils import build_graph, build_predict_graph


def _apply_fn(edges):
    return {
        "src": normalize(edges.src['x'], dim=1),
        "dst": normalize(edges.dst['x'], dim=1)
    }


class GCNNet(torch.nn.Module):

    def __init__(self, num_rels, feat_dim, hidden_dims, rel_emb_dim=32, device='cpu'):
        super(GCNNet, self).__init__()
        self.device = device
        self.rel_emb = torch.nn.Embedding(num_rels, rel_emb_dim, max_norm=1, norm_type=2.).to(device)
        self.gcn_layers = torch.nn.ModuleList(
            [GraphConv(feat_dim, hidden_dims[0]).to(device)]
        )
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self.gcn_layers.append(
                GraphConv(size, hidden_dims[layer_idx+1]).to(device)
            )
        self.linear = torch.nn.Linear(hidden_dims[-1]+feat_dim, rel_emb_dim, bias=False).to(device)

    def forward(self, data):
        input_feat, pos_edge_subg, neg_edge_subg, blocks = data
        x = input_feat
        for layer, block in zip(self.gcn_layers, blocks):
            x = torch.tanh(layer(block, x))
        x = self.linear(torch.cat((x, input_feat[pos_edge_subg.nodes('_U')]), dim=1))
        with pos_edge_subg.local_scope():
            pos_edge_subg.ndata['x'] = x
            pos_edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = pos_edge_subg.edata['src'], pos_edge_subg.edata['dst']
            r_emb = self.rel_emb(pos_edge_subg.edata['type'])
            h_r_emb = h_emb + r_emb
            pos_score = torch.norm((h_r_emb - t_emb), dim=1, keepdim=False)
        with neg_edge_subg.local_scope():
            neg_edge_subg.ndata['x'] = x
            neg_edge_subg.apply_edges(_apply_fn)
            t_emb = neg_edge_subg.edata['dst']
            neg_score = torch.norm((h_r_emb - t_emb), dim=1, keepdim=False)
        return pos_score, neg_score

    @torch.no_grad()
    def predict(self, data):
        input_feat, edge_subg, blocks = data
        x = input_feat
        for layer, block in zip(self.gcn_layers, blocks):
            x = torch.tanh(layer(block, x))
        x = self.linear(torch.cat((x, input_feat[edge_subg.nodes('_U')]), dim=1))
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = edge_subg.edata['src'], edge_subg.edata['dst']
            r_emb = self.rel_emb(edge_subg.edata['type'])
            score = torch.norm((h_emb+r_emb-t_emb), dim=1, keepdim=False)
        return score

    @torch.no_grad()
    def embed(self, data):
        input_feat, edge_subg, blocks = data
        x = input_feat
        for layer, block in zip(self.gcn_layers, blocks):
            x = torch.tanh(layer(block, x))
        x = self.linear(torch.cat((x, input_feat[blocks[-1].edges('uv')[1]]), dim=1))
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = edge_subg.edata['src'], edge_subg.edata['dst']
        return h_emb, t_emb


def train(g, train_eid=None, fanouts=(8,8), hidden_dims=(128, 128), rel_emb_dim=128, batch_size=256, device='cpu', model_path=None):
    if train_eid is None:
        train_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, sampler, device=device,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        feat_dim=100,#TODO,
        hidden_dims=hidden_dims,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    if model_path is not None:
        gcn.load_state_dict(torch.load(model_path, map_location=device))
    opt = torch.optim.Adam(gcn.parameters())
    # TODO node feature(transE embeddings)
    print(time.ctime(), "loading node features ......")
    node_feat = torch.from_numpy(np.load("/home/liusx/wiki90/wikikg90m_PairRE_entity.npy"))
    print(time.ctime(), "loading node done")
    batch_idx = 0
    for input_nodes, pos_edge_subg, neg_edge_subg, blocks in dataloader:
        batch_idx += 1
        if batch_idx == 1:
            print("start training ......")
        input_feat = node_feat[input_nodes.to('cpu')].type(torch.float32).to(gcn.device)
        pos_score, neg_score = gcn((input_feat, pos_edge_subg, neg_edge_subg, blocks))
        loss = (pos_score - neg_score + 1).clamp(min=0).mean()
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch, loss: %s" % (batch_idx, loss.item()))
            torch.save(gcn.state_dict(), "/home/liusx/torch_models/gcn_transE_feat/gcn_weight.pth.%d"%batch_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(gcn.state_dict(), "/home/liusx/torch_models/gcn_transE_feat/gcn_weight.pth.done")


def predict(model_path,
            fanouts=(8,8),
            hidden_dims=(128, 128),
            rel_emb_dim=128,
            batch_size=1001,
            max_batch=10000,
            device='cpu'):
    g = build_predict_graph(max_batch)
    pred_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, pred_eid, sampler, device=device, g_sampling=build_graph(),
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        feat_dim=100,
        hidden_dims=hidden_dims,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    gcn.load_state_dict(torch.load(model_path, map_location=device))
    node_feat = torch.from_numpy(np.load("/home/liusx/wiki90/wikikg90m_PairRE_entity.npy"))
    scores = []
    batch_idx = 0
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        input_feat = node_feat[input_nodes.to('cpu')].type(torch.float32).to(gcn.device)
        score = gcn.predict((input_feat, edge_subg, blocks))
        scores.append(score)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
        if batch_idx >= max_batch: break
    return torch.cat(scores)


if __name__ == "__main__":
    print(time.ctime(), 'building train graph ......')
    g = build_graph()
    print(time.ctime(), 'building train graph done')
    train(g, batch_size=4096, device='cuda:6')
    pass

