#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import time
from utils import load_norm_inc


class TransE(torch.nn.Module):

    def __init__(self, num_rels=1315, feat_dim=768, rel_emb_dim=32, out_dim=32, device='cpu'):
        super(TransE, self).__init__()
        self.device = device
        self.rel_emb_dim = rel_emb_dim
        self.rel_emb = torch.nn.Parameter(torch.Tensor(num_rels, rel_emb_dim)).to(device)
        self.out_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.in_transform = torch.nn.Linear(rel_emb_dim, rel_emb_dim).to(device)
        self.entity_transform = torch.nn.Linear(feat_dim+2*rel_emb_dim, out_dim).to(device)
        self.relation_transform = torch.nn.Linear(rel_emb_dim, out_dim).to(device)

    def forward(self, data):
        h_in_inc, h_out_inc, h_node_feat, \
         eid, pos_t_in_inc, pos_t_out_inc, pos_t_node_feat, \
         neg_t_in_inc, neg_t_out_inc, neg_t_node_feat = data
        r_emb = F.normalize(self.relation_transform(self.rel_emb[eid]), dim=1, p=2)
        h_in_emb = self.in_transform(torch.sparse.mm(
            h_in_inc,
            self.rel_emb
        ))
        h_out_emb = self.out_transform(torch.sparse.mm(
            h_out_inc,
            self.rel_emb
        ))
        h_feat = torch.cat(
            (h_node_feat, h_in_emb, h_out_emb),
            dim=1
        )
        h_emb = F.normalize(self.entity_transform(h_feat), dim=1, p=2)
        pos_t_in_emb = self.in_transform(torch.sparse.mm(
            pos_t_in_inc,
            self.rel_emb
        ))
        pos_t_out_emb = self.out_transform(torch.sparse.mm(
            pos_t_out_inc,
            self.rel_emb
        ))
        pos_t_feat = torch.cat(
            (pos_t_node_feat, pos_t_in_emb, pos_t_out_emb),
            dim=1
        )
        pos_t_emb = F.normalize(self.entity_transform(pos_t_feat), dim=1, p=2)
        neg_t_in_emb = self.in_transform(torch.sparse.mm(
            neg_t_in_inc,
            self.rel_emb
        ))
        neg_t_out_emb = self.out_transform(torch.sparse.mm(
            neg_t_out_inc,
            self.rel_emb
        ))
        neg_t_feat = torch.cat(
            (neg_t_node_feat, neg_t_in_emb, neg_t_out_emb),
            dim=1
        )
        neg_t_emb = F.normalize(self.entity_transform(neg_t_feat), dim=1, p=2)
        pos_score = torch.norm(h_emb + r_emb - pos_t_emb, dim=1)
        neg_score = torch.norm(h_emb + r_emb - neg_t_emb, dim=1)
        return pos_score, neg_score

    @torch.no_grad()
    def predict(self, data):
        h_in_inc, h_out_inc, h_node_feat, \
        eid, t_in_inc, t_out_inc, t_node_feat = data
        r_emb = F.normalize(self.relation_transform(self.rel_emb[eid]), dim=1, p=2)
        h_in_emb = self.in_transform(torch.sparse.mm(
            h_in_inc,
            self.rel_emb
        ))
        h_out_emb = self.out_transform(torch.sparse.mm(
            h_out_inc,
            self.rel_emb
        ))
        h_feat = torch.cat(
            (h_node_feat, h_in_emb, h_out_emb),
            dim=1
        )
        h_emb = F.normalize(self.entity_transform(h_feat), dim=1, p=2)
        t_in_emb = self.in_transform(torch.sparse.mm(
            t_in_inc,
            self.rel_emb
        ))
        t_out_emb = self.out_transform(torch.sparse.mm(
            t_out_inc,
            self.rel_emb
        ))
        t_feat = torch.cat(
            (t_node_feat, t_in_emb, t_out_emb),
            dim=1
        )
        t_emb = F.normalize(self.entity_transform(t_feat), dim=1, p=2)
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
    print(time.ctime(), "loading incidence matrix ......")
    in_inc, out_inc = load_norm_inc()
    print(time.ctime(), "loading incidence matrix done")
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    transe = TransE(device='cuda:7')
    opt = torch.optim.Adam(transe.parameters())
    batch_idx = 0
    for triplets in dataloader:
        batch_idx += 1
        neg_t_id = torch.randint(87143637, size=(triplets.size(0),), dtype=torch.int64, device='cpu')
        h_id, e_id, pos_t_id = triplets[:,0], triplets[:,1].to(device), triplets[:,2]
        h_in_inc = get_batch_inc(h_id, in_inc, device)
        h_out_inc = get_batch_inc(h_id, out_inc, device)
        h_node_feat = node_feat[h_id].to(device)
        pos_t_in_inc = get_batch_inc(pos_t_id, in_inc, device)
        pos_t_out_inc = get_batch_inc(pos_t_id, out_inc, device)
        pos_t_node_feat = node_feat[pos_t_id].to(device)
        neg_t_in_inc = get_batch_inc(neg_t_id, in_inc, device)
        neg_t_out_inc = get_batch_inc(neg_t_id, out_inc, device)
        neg_t_node_feat = node_feat[neg_t_id].to(device)
        p_score, n_score = transe((h_in_inc, h_out_inc, h_node_feat,
                                   e_id, pos_t_in_inc, pos_t_out_inc, pos_t_node_feat,
                                   neg_t_in_inc, neg_t_out_inc, neg_t_node_feat))
        loss = (p_score - n_score + 10).clamp(min=0).mean()
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch, loss: %s" % (batch_idx, loss.item()))
            torch.save(transe.state_dict(), "/home/liusx/torch_models/transe/transe.pth.%d"%batch_idx)
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