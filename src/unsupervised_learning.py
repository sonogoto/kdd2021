#!/usr/bin/env python3
import torch
from dgl.nn.pytorch.conv import GraphConv
import dgl
import time
import numpy as np
from utils import build_graph, build_predict_graph, Identical


def _apply_fn(edges):
    return {"src": edges.src['x'], "dst": edges.dst['x']}


class GCNNet(torch.nn.Module):

    def __init__(self, num_rels, feat_dim, hidden_dims, rel_emb_dim=32, device='cpu'):
        super(GCNNet, self).__init__()
        self.device = device
        self.rel_emb = torch.nn.Embedding(num_rels, rel_emb_dim).to(device)
        self.out_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.in_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.gcn_layers = torch.nn.ModuleList(
            [GraphConv(feat_dim+rel_emb_dim*2, hidden_dims[0]).to(device)]
        )
        self.ident = Identical()
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self.gcn_layers.append(
                GraphConv(size, hidden_dims[layer_idx+1]).to(device)
            )

    def forward(self, data):
        input_feat, pos_edge_subg, neg_edge_subg, blocks = data
        out_edge_emb = torch.sparse.mm(
            blocks[0].inc('out').to(self.device),
            self.out_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        in_edge_emb = torch.sparse.mm(
            blocks[0].inc('in').to(self.device),
            self.in_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        out_edge_emb = torch.nn.functional.pad(
            out_edge_emb,
            (0, 0, 0, input_feat.size(0)-out_edge_emb.size(0))
        )
        in_edge_emb = torch.nn.functional.pad(
            in_edge_emb,
            (0, 0, 0, input_feat.size(0)-in_edge_emb.size(0))
        )
        input_feat = torch.cat(
            (input_feat, out_edge_emb, in_edge_emb),
            dim=1
        )
        x = self.ident(blocks[0], input_feat)
        for layer, block in zip(self.gcn_layers, blocks[1:]):
            x = torch.relu(layer(block, x))
        with pos_edge_subg.local_scope():
            pos_edge_subg.ndata['x'] = x
            pos_edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = pos_edge_subg.edata['src'], pos_edge_subg.edata['dst']
            r_emb = self.rel_emb(pos_edge_subg.edata['type'])
            h_r_emb = h_emb + r_emb
            pos_score = torch.sqrt(torch.pow(h_r_emb-t_emb,2).sum(dim=1))
        with neg_edge_subg.local_scope():
            neg_edge_subg.ndata['x'] = x
            neg_edge_subg.apply_edges(_apply_fn)
            t_emb = neg_edge_subg.edata['dst']
            neg_score = torch.sqrt(torch.pow(h_r_emb-t_emb,2).sum(dim=1))
        return pos_score, neg_score

    @torch.no_grad()
    def predict(self, data):
        input_feat, edge_subg, blocks = data
        out_edge_emb = torch.sparse.mm(
            blocks[0].inc('out').to(self.device),
            self.out_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        in_edge_emb = torch.sparse.mm(
            blocks[0].inc('in').to(self.device),
            self.in_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        out_edge_emb = torch.nn.functional.pad(
            out_edge_emb,
            (0, 0, 0, input_feat.size(0)-out_edge_emb.size(0))
        )
        in_edge_emb = torch.nn.functional.pad(
            in_edge_emb,
            (0, 0, 0, input_feat.size(0)-in_edge_emb.size(0))
        )
        input_feat = torch.cat(
            (input_feat, out_edge_emb, in_edge_emb),
            dim=1
        )
        x = self.ident(blocks[0], input_feat)
        for layer, block in zip(self.gcn_layers, blocks[1:]):
            x = torch.relu(layer(block, x))
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = edge_subg.edata['src'], edge_subg.edata['dst']
            r_emb = self.rel_emb(edge_subg.edata['type'])
            score = torch.sqrt(torch.pow(h_emb+r_emb-t_emb,2).sum(dim=1))
        return score

    @torch.no_grad()
    def embed(self, data):
        input_feat, edge_subg, blocks = data
        out_edge_emb = torch.sparse.mm(
            blocks[0].inc('out').to(self.device),
            self.out_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        in_edge_emb = torch.sparse.mm(
            blocks[0].inc('in').to(self.device),
            self.in_transform(self.rel_emb(blocks[0].edata['type'].to(self.device)))
        )
        out_edge_emb = torch.nn.functional.pad(
            out_edge_emb,
            (0, 0, 0, input_feat.size(0)-out_edge_emb.size(0))
        )
        in_edge_emb = torch.nn.functional.pad(
            in_edge_emb,
            (0, 0, 0, input_feat.size(0)-in_edge_emb.size(0))
        )
        input_feat = torch.cat(
            (input_feat, out_edge_emb, in_edge_emb),
            dim=1
        )
        x = self.ident(blocks[0], input_feat)
        for layer, block in zip(self.gcn_layers, blocks[1:]):
            x = torch.relu(layer(block, x))
        with edge_subg.local_scope():
            edge_subg.ndata['x'] = x
            edge_subg.apply_edges(_apply_fn)
            h_emb, t_emb = edge_subg.edata['src'], edge_subg.edata['dst']
        return h_emb, t_emb


def train(g, train_eid=None, fanouts=(8,8,8), hidden_dims=(64, 32), rel_emb_dim=32, batch_size=256, device='cpu', model_path=None):
    if train_eid is None:
        train_eid = g.edges('eid')
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid, sampler, device=device,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    gcn = GCNNet(
        num_rels=1315,
        feat_dim=768,
        hidden_dims=hidden_dims,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    if model_path is not None:
        gcn.load_state_dict(torch.load(model_path, map_location=device))
    opt = torch.optim.Adam(gcn.parameters())
    print(time.ctime(), "preparing node features ......")
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)

    print(time.ctime(), "preparing node features done")
    batch_idx = 0
    for input_nodes, pos_edge_subg, neg_edge_subg, blocks in dataloader:
        batch_idx += 1
        if batch_idx == 1:
            print("start training ......")
        input_feat = node_feat[input_nodes.to('cpu')].type(torch.float32).to(gcn.device)
        pos_score, neg_score = gcn((input_feat, pos_edge_subg, neg_edge_subg, blocks))
        loss = (pos_score - neg_score + 5).clamp(min=0).mean()
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch, loss: %s" % (batch_idx, loss.item()))
            torch.save(gcn.state_dict(), "/home/liusx/torch_models/remove_nodes/gcn_weight.pth.%d"%batch_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(gcn.state_dict(), "/home/liusx/torch_models/remove_nodes/gcn_weight.pth.done")


def predict(model_path,
            fanouts=(8,8,8),
            hidden_dims=(64, 32),
            rel_emb_dim=32,
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
        feat_dim=768,
        hidden_dims=hidden_dims,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    gcn.load_state_dict(torch.load(model_path, map_location=device))
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
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


def node_embeddings(model_path,
                    fanouts=(8,8,8),
                    hidden_dims=(64, 32),
                    rel_emb_dim=32,
                    batch_size=4096,
                    device='cpu'):
    g_sampling = build_graph()
    nodes = g_sampling.nodes()
    if nodes.size(0) % 2 == 1:
        nodes = torch.cat((nodes, torch.tensor([0])))
    u, v = nodes[:nodes.size(0)/2], nodes[nodes.size(0)/2:]
    g = dgl.graph((u, v))
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, g.edges('eid'), sampler, device=device, g_sampling=g_sampling,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32)
    gcn = GCNNet(
        num_rels=1315,
        feat_dim=768,
        hidden_dims=hidden_dims,
        rel_emb_dim=rel_emb_dim,
        device=device
    )
    gcn.load_state_dict(torch.load(model_path, map_location=device))
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    batch_idx = 0
    h_embs, t_embs = [], []
    for input_nodes, edge_subg, blocks in dataloader:
        batch_idx += 1
        input_feat = node_feat[input_nodes.to('cpu')].type(torch.float32).to(gcn.device)
        h_emb, t_emb = gcn.embed((input_feat, edge_subg, blocks))
        h_embs.append(h_emb.to('cpu'))
        t_embs.append(t_emb.to('cpu'))
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
    return torch.cat(h_embs + t_embs, dim=0).to('cpu')


if __name__ == "__main__":
    print(time.ctime(), 'building train graph ......')
    g = build_graph()
    print(time.ctime(), 'building train graph done')
    train(g, batch_size=4096, device='cuda:6')
    pass

