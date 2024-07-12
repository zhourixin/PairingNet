import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential as Seq
from encoder import FlattenNet, TextureEncoder, MyDeepGCN, PatchEncoder, MyDeepGCN2
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import GATConv

def generate_batch_tensor(num_nodes,num_batch, device):
    x = torch.zeros(num_nodes*num_batch, dtype=torch.int64).to(device)
    for i in range(num_batch, num_nodes*num_batch, num_nodes):
        x[i:i+num_nodes] = i // num_nodes
    return x

class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p, args):
        super(GraphUnet, self).__init__()
        self.bs = args.batch_size
        self.ks = ks
        self.bottom_gcn = MyDeepGCN(args)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.SAGpools = nn.ModuleList()
        self.l_n = len(ks)

        for i in range(self.l_n):
            self.down_gcns.append(Seq(*[MyDeepGCN(args)]))
            self.up_gcns.append(Seq(*[MyDeepGCN(args)]))
            self.SAGpools.append(Seq(*[SAGPooling(dim, ks[i], GATConv)]))

            # self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))
        
        # self.SAGpool = SAGPooling(dim, 0.5, GATConv)

    def forward(self, h, g, bs):

        adj_ms = []
        down_outs = []
        indices_list = []
        to_length = []
        hs = []
        
        N = h.shape[0]//bs
        org_h = h
        BS_list = generate_batch_tensor(N, bs, h.device)
        for i in range(self.l_n):
            h = self.down_gcns[i][0](h, g)
            
            
            adj_ms.append(g)
            down_outs.append(h)
            to_length.append(h.shape[0])
            h, g, _, BS_list, indexes, _ = self.SAGpools[i][0](h, g, batch=BS_list)
            indices_list.append(indexes)
        
        h = self.bottom_gcn(h, g)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, indexes = adj_ms[up_idx], indices_list[up_idx]
            h = self.unpools[i](to_length[up_idx], h, indexes)
            h = h.add(down_outs[up_idx])
            h = self.up_gcns[i][0](h, g)
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs



class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, num_points, h, idx):
        new_h = h.new_zeros([num_points, h.shape[1]])
        new_h[idx] = h
        return new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)