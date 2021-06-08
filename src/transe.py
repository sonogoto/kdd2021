#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import time
from utils import load_norm_inc
from math import sqrt


class TransE(torch.nn.Module):

    def __init__(self, num_rels=1315, num_eneities=87143637, emb_dim=128):
        super(TransE, self).__init__()
        self.rel_emb = torch.nn.Embedding(num_rels, emb_dim, max_norm=1)
        torch.nn.init.uniform_(self.rel_emb.weight, -6./sqrt(emb_dim), 6./sqrt(emb_dim))
        self.entity_emb = torch.nn.Embedding(num_eneities, emb_dim)
        torch.nn.init.uniform_(self.rel_emb.weight, -6./sqrt(emb_dim), 6./sqrt(emb_dim))

    def forward(self, data, device='cpu'):
        h_id, e_id, t_id, neg_id = data
        h_emb = self.entity_emb(h_id).to(device)
        r_emb = self.rel_emb(e_id).to(device)
        t_emb = self.entity_emb(t_id).to(device)
        neg_emb = self.entity_emb(neg_id).to(device)
        pos_score = torch.norm(h_emb + r_emb - t_emb, dim=1)
        # replace tail
        if np.random.rand() > .5:
            neg_score = torch.norm(h_emb + r_emb - neg_emb, dim=1)
        # replace header
        else:
            neg_score = torch.norm(neg_emb + r_emb - t_emb, dim=1)
        return pos_score, neg_score

    @torch.no_grad()
    def predict(self, data):
        h_id, e_id, t_id = data
        h_emb = self.entity_emb(h_id)
        r_emb = self.rel_emb(e_id)
        t_emb = self.entity_emb(t_id)
        return torch.norm(h_emb + r_emb - t_emb, dim=1)


def get_batch_inc(batch, inc, device):
    x = inc[batch].tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor([x.row, x.col], dtype=torch.int64),
        torch.from_numpy(x.data),
        size=(batch.size(0), 1315),
        device=device
    )


def train(batch_size=4096, device='cpu'):
    print(time.ctime(), "preparing data ......")
    data = torch.from_numpy(np.load("/data/wikikg90m_kddcup2021/processed/train_hrt.npy"))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=32)
    print(time.ctime(), "preparing data done")
    print(time.ctime(), "building TransE model ......")
    transe = TransE()
    print(time.ctime(), "building TransE model done")
    opt = torch.optim.Adam(transe.parameters())
    batch_idx = 0
    for triplets in dataloader:
        batch_idx += 1
        neg_id = torch.randint(87143637, size=(triplets.size(0),), dtype=torch.int64, device='cpu')
        h_id, e_id, t_id = triplets[:,0], triplets[:,1], triplets[:,2]
        p_score, n_score = transe((h_id, e_id, t_id, neg_id), device=device)
        loss = (p_score - n_score + 5).clamp(min=0).mean()
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch, loss: %s" % (batch_idx, loss.item()))
            # torch.save(transe.state_dict(), "/home/liusx/torch_models/transe/transe.pth.%d"%batch_idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(transe.state_dict(), "/home/liusx/torch_models/transe/transe.pth.done")


def predict(model_path,
            batch_size=1001,
            max_batch=10000,
            device='cpu'):
    h, r = np.loadtxt("/data/wikikg90m_kddcup2021/processed/val_hr.txt",
                      delimiter=" ",
                      dtype=np.int64,
                      max_rows=max_batch
                      ).T
    repeat = np.ones(h.shape[0], dtype=np.int32) + 1000
    h = np.repeat(h, repeat)
    r = np.repeat(r, repeat)
    t_candidate = np.load("/data/wikikg90m_kddcup2021/processed/val_t_candidate.npy")[:max_batch].reshape(-1)
    data = torch.tensor([h, r, t_candidate]).T
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=32)
    in_inc, out_inc = load_norm_inc()
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    transe = TransE(device=device)
    transe.load_state_dict(torch.load(model_path, map_location=device))
    scores = []
    batch_idx = 0
    for triplets in dataloader:
        batch_idx += 1
        h_id, e_id, t_id = triplets[:,0], triplets[:,1].to(device), triplets[:,2]
        h_in_inc = get_batch_inc(h_id, in_inc, device)
        h_out_inc = get_batch_inc(h_id, out_inc, device)
        h_node_feat = node_feat[h_id].to(device)
        t_in_inc = get_batch_inc(t_id, in_inc, device)
        t_out_inc = get_batch_inc(t_id, out_inc, device)
        t_node_feat = node_feat[t_id].to(device)
        score = transe.predict((h_in_inc, h_out_inc, h_node_feat,
                                e_id, t_in_inc, t_out_inc, t_node_feat))
        scores.append(score)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % batch_idx)
        if batch_idx >= max_batch: break
    return torch.cat(scores)


if __name__ == "__main__":
    train(device='cuda:7')