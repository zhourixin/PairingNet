import __init__
import torch
import torch.nn as nn
from encoder import FlattenNet, MyDeepGCN, PatchEncoder_average, FlattenNet_average


class SelfGateV1(nn.Module):
    """GRU update-gate-like fusion module"""
    def __init__(self):
        super(SelfGateV1, self).__init__()
        # self.fc = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc_2 = nn.Linear(128, 64)
        self.fc1_2 = nn.Linear(64, 1)
        self.activate = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()


        self.proj1 = nn.Linear(128, 128)
        self.proj2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, 64)


        self.fc3 = nn.Linear(256, 128)

    def forward(self, c, t):
        """
        :param q: [batch_size, n, dim]
        :param c: [batch_size, n, dim]
        :param t: [batch_size, n, dim]
        :return: mixed_feature [batch_size, n, dim]
        """
        bs, n, dim = c.size()


        w = self.fc(torch.cat((c, t), dim=-1))
        w = self.activate(w)
        w = self.sigmoid(self.fc1(w))

        c = c * w.view(bs, n, -1)
        t = t * (1 - w).view(bs, n, -1)
        mixed_feature = torch.add(c, t)

        return mixed_feature, w

class SelfGateV2(nn.Module):
    """GRU update-gate-like fusion module"""
    def __init__(self):
        super(SelfGateV2, self).__init__()
        self.fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 64)
        self.activate = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, c, t):
        """
        :param q: [batch_size, n, dim]
        :param c: [batch_size, n, dim]
        :param t: [batch_size, n, dim]
        :return: mixed_feature [batch_size, n, dim]
        """
        bs, n, dim = c.size()

        w = self.fc(torch.cat((c, t), dim=-1))
        w = self.activate(w)
        w = self.sigmoid(self.fc1(w))

        c = c * w.view(bs, n, -1)
        t = t * (1 - w).view(bs, n, -1)
        # mixed_feature = torch.add(c, t)
        mixed_feature = torch.cat((c, t), dim=-1)

        return mixed_feature, w

class Vanilla(nn.Module):
    """
        This is the pipeline for considering both contour and texture information.
        In this pipeline, we concatenate the contour feature and texture feature
        directly.
    """

    def __init__(self, args):
        super(Vanilla, self).__init__()
        self.flatten_net = FlattenNet(args.flattenNet_config)
        self.encoder_c = MyDeepGCN(args)

        self.flatten_net = FlattenNet_average(args.flattenNet_config)
        self.encoder_t = PatchEncoder_average()

        self.gcn_t = MyDeepGCN(args)
        self.model = args.model_type
        self.fc = nn.Linear(args.in_channels+2, args.in_channels)
        self.fc2 = nn.Linear(2 * args.in_channels, args.in_channels)
        self.c_feature = args.flattenNet_config["output_dim"]

        self.selfgate = SelfGateV1()
        

    def forward(self, inputs):
        contour = inputs['pcd']
        img = inputs['img']  # bs, n, 3, 7, 7
        c_input = inputs['c_input']
        t_input = inputs['t_input']
        bs, c, _, _ = img.size()
        adj = inputs['adj']
        # att_mask = inputs['att_mask']
        _, n, _, _ = c_input.shape
        c_feature = self.c_feature
        
        contour += torch.tensor([1, 1]).cuda()
        
        flatted_c = self.flatten_net(c_input)

        # _, c = flatted_c.shape
        flatted_c = flatted_c.view(bs, n, -1)
        contour_in_c = contour - torch.mean(contour, dim=1, keepdim=True)
        contour_in_c -= torch.tensor([1, 1]).cuda()
        flatted_c = torch.cat((flatted_c, contour_in_c), dim=-1)
        flatted_c = self.fc(flatted_c)
        flatted_c = flatted_c.view(-1, c_feature)
        l_c = self.encoder_c(flatted_c, adj)
        l_c = l_c.view(bs, n, -1)

        t_input = t_input.view(bs*n, 3, 7, 7)
        l_t = self.encoder_t(t_input)  # bs*n, 64, 1, 1
        l_t = l_t.view(bs*n, -1)
        l_t = self.gcn_t(l_t, adj)
        l_t = l_t.view(bs, n, -1)
        q = torch.cat((l_c, l_t), dim=-1)
        # f_fused = self.fc2(q)


        f_fused2, w_ = self.selfgate(l_c, l_t)

        return f_fused2, q, w_


class TransformerEncoderModel(nn.Module):
    def __init__(self, args, d_model=64, nhead=2, num_layers=2):
        super(TransformerEncoderModel, self).__init__()

        from linear_attention_transformer import LinearAttentionTransformer
        self.tranct_length = args.tranct_length
        self.transformer_encoder_c = LinearAttentionTransformer(
            dim = 64,
            heads = 8,
            depth = 5,
            max_seq_len = self.tranct_length,
            n_local_attn_heads = 4
        )
        self.transformer_encoder_t = LinearAttentionTransformer(
            dim = 64,
            heads = 8,
            depth = 5,
            max_seq_len = self.tranct_length,
            n_local_attn_heads = 4
        )


        self.act = nn.Tanh()
        self.out_l_1 = nn.Linear(args.in_channels_stage2, args.global_out_channels)
        self.fc_s = nn.Linear(args.max_length, args.max_length//2)
        self.fc_t = nn.Linear(args.max_length, args.max_length//2)
        self.selfgate = SelfGateV2()


    def readout(self, hs):
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_mean) 
        return h
    def readout_2(self, hs):
        h_max = [torch.max(h, 1)[0] for h in hs]
        h_sum = [torch.sum(h, 1) for h in hs]
        h_mean = [torch.mean(h, 1) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean) # 12*48=578
        return h
    def FC_layer(self, h):
        h = self.out_l_1(h)
        return h

    def forward(self, src, contour):

        src_c = src[:,:self.tranct_length,0:64]
        src_t = src[:,:self.tranct_length,64:]

        src_c = self.transformer_encoder_c(src_c)
        src_t = self.transformer_encoder_t(src_t)

        F_fused, w_ = self.selfgate(src_c, src_t)

        F_global = []
        for i in range(F_fused.shape[0]):
            f_global = self.readout([F_fused[i]])
            F_global.append(f_global.unsqueeze(0))
        F_global = torch.cat(F_global, dim=0) 
        F_global = self.FC_layer(F_global)

        return F_global, w_



if __name__ == "__main__":
    print(1)