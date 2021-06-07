#!/usr/bin/env python3

import torch
import numpy as np
import time


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
        x = torch.cat((u, v), dim=1)
        for layer in self.linear_layers:
            x = torch.sigmoid(layer(x))
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
    mlp = MLP(768+768, hidden_dims, out_dim, device)
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


if __name__ == "__main__":
    train(device='cuda:6')

