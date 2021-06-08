#!/usr/bin/env python3

import torch
import numpy as np
import time
from ogb.lsc import WikiKG90MEvaluator


class MLP(torch.nn.Module):

    def __init__(self, feat_dim, hidden_dims, out_dim, device):
        super(MLP, self).__init__()
        dims = [feat_dim, ] + hidden_dims + [out_dim, ]
        self.linear_layers = torch.nn.ModuleList()
        for idx in range(len(dims)-1):
            self.linear_layers.append(
                torch.nn.Linear(dims[idx], dims[idx+1]).to(device)
            )

    def forward(self, data):
        u, v = data
        x = u - v
        for layer in self.linear_layers:
            x = torch.relu(layer(x))
        return x

    def predict(self, data):
        with torch.no_grad():
            return self(data)


def train(hidden_dims=[256, 128], out_dim=1315, batch_size=4096, device='cpu'):
    print(time.ctime(), "preparing train data ......")
    train_data = torch.from_numpy(
        np.load("/data/wikikg90m_kddcup2021/processed/train_hrt.npy")
    )
    dataloader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=32,
        drop_last=False
    )
    print(time.ctime(), "preparing train data done")
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    mlp = MLP(768, hidden_dims, out_dim, device)
    opt = torch.optim.Adam(mlp.parameters())
    batch_idx = 0
    for triplets in dataloader:
        batch_idx += 1
        u = node_feat[triplets[:, 0]].type(torch.float32).to(device)
        v = node_feat[triplets[:, 2]].type(torch.float32).to(device)
        labels = triplets[:, 1].to(device)
        preds = mlp((u, v))
        loss = torch.nn.functional.cross_entropy(preds, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if batch_idx % 100 == 0:
            acc = (preds.argmax(dim=1) == labels).sum() / preds.size(0)
            print(time.ctime(), "%d-th mini batch, loss: %s, accuracy: %s" % (batch_idx, loss.item(), acc.item()))
            torch.save(mlp.state_dict(), "/home/liusx/torch_models/mlp/mlp.pth.%d"%batch_idx)
    torch.save(mlp.state_dict(), "/home/liusx/torch_models/mlp/mlp.pth.done")


def evaluate(model_path, hidden_dims=[256, 128], out_dim=1315, device='cpu', skip_rows=0, max_batch=10000):
    print(time.ctime(), "preparing predict data ......")
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
    pred_data = torch.tensor([h, r, t_candidate]).T
    dataloader = torch.utils.data.DataLoader(
        pred_data,
        shuffle=False,
        batch_size=1001,
        num_workers=32,
        drop_last=False
    )
    print(time.ctime(), "preparing predict data done")
    node_feat = torch.from_numpy(
        np.memmap("/run/dataset/entity_feat_fill_nan.npy", mode="r", dtype=np.float16, shape=(87143637, 768))[:]
    ).type(torch.float16)
    mlp = MLP(768, hidden_dims, out_dim, device)
    mlp.load_state_dict(torch.load(model_path, map_location=device))
    batch_idx = 0
    preds = []
    for triplets in dataloader:
        batch_idx += 1
        u = node_feat[triplets[:, 0]].type(torch.float32).to(device)
        v = node_feat[triplets[:, 2]].type(torch.float32).to(device)
        # pred = torch.softmax(mlp.predict((u, v)), dim=1)[(torch.arange(triplets.size(0)), triplets[:, 1])]
        pred = torch.softmax(mlp.predict((u, v)), dim=1).max(dim=1).values
        preds.append(pred)
        if batch_idx % 100 == 0:
            print(time.ctime(), "%d-th mini batch" % (batch_idx,))
    preds = torch.cat(preds, dim=0).reshape(-1, 1001).cpu().numpy()
    t_pred_top10 = np.argsort(-preds, axis=1)[:,:10]
    t_correct_index = np.load(
        "/data/wikikg90m_kddcup2021/processed/val_t_correct_index.npy")[skip_rows:skip_rows+max_batch]
    evaluator = WikiKG90MEvaluator()
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}
    return evaluator.eval(input_dict)


if __name__ == "__main__":
    train(device='cuda:6')

