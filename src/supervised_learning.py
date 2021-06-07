#!/usr/bin/env python3

import dgl
import numpy as np
import time
import torch
from utils import build_graph, build_predict_graph, Identical
from dgl.nn.pytorch.conv import GraphConv


def concat_src_dst(edges):
    return {"edge_emb": torch.cat((edges.src['x'], edges.dst['x']), dim=1)}


class GCNNet(torch.nn.Module):

    def __init__(self, num_rels, hidden_dims, out_dim, rel_emb_dim=32, device='cpu'):
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
        self.fc = torch.nn.Linear(in_features=hidden_dims[-1]*2, out_features=out_dim).to(device)
        self.ident = Identical().to(device)

    def forward(self, data):
        input_nodes, edge_subg, blocks = data
        out_edge_emb = torch.sparse.mm(
            blocks[0].inc('out').to(self.device),
            self.out_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        ).div(blocks[0].inc('out').sum(dim=1, keepdim=True))
        in_edge_emb = torch.sparse.mm(
            blocks[0].inc('in').to(self.device),
            self.in_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        ).div(blocks[0].inc('in').sum(dim=1, keepdim=True))
        out_edge_emb = torch.nn.functional.pad(
            out_edge_emb,
            (0, 0, 0, input_nodes.size(0)-out_edge_emb.size(0))
        )
        in_edge_emb = torch.nn.functional.pad(
            in_edge_emb,
            (0, 0, 0, input_nodes.size(0)-in_edge_emb.size(0))
        )
        input_feat = torch.cat(
            (out_edge_emb, in_edge_emb),
            dim=1
        )
        x = self.ident(blocks[0], input_feat)
        for layer, block in zip(self.gcn_layers, blocks[1:]):
            x = torch.relu(layer(block, x))
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(concat_src_dst)
            return self.fc(edge_subg.edata['edge_emb'])


def train(g, train_eid=None, fanouts=(8, 8, 8), hidden_dims=(32, 32), rel_emb_dim=32, batch_size=256, device='cpu'):
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
        preds = gcn((input_nodes, edge_subg, blocks))
        loss = torch.nn.functional.cross_entropy(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if batch_idx % 100 == 0:
            acc = (preds.argmax(dim=1) == labels).sum() / preds.size(0)
            print(time.ctime(), "%d-th mini batch, loss: %s, accuracy: %s" % (batch_idx, loss.item(), acc.item()))
            torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised/gcn_weight.pth.%d"%batch_idx)
    torch.save(gcn.state_dict(), "/home/liusx/torch_models/supervised/gcn_weight.pth.done")


@torch.no_grad()
def predict(model_path,
            fanouts=(8, 8, 8),
            hidden_dims=(32, 32),
            rel_emb_dim=32,
            batch_size=1001,
            device='cpu'):
    g = build_predict_graph(max_batch=10000)
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
            gcn((input_nodes, edge_subg, blocks)),
            dim=1
        )[(torch.arange(edge_subg.num_edges()), edge_subg.edata['type'])]
        probs.append(prob)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
    return torch.cat(probs, dim=0)


if __name__ == "__main__":
    train(build_graph(), batch_size=4096, device='cuda:7')