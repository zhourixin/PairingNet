import __init__
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import Sequential as Seq
from torch_geometric.nn import GATConv, DeepGCNLayer
from interpolation import get_gcn_feature


class Project(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True):
        super(Project, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        out = self.fc(x)
        N, C, D = out.size()
        out = self.bn(out.view(-1, D)).view(N, C, D)
        out = self.act(out)
        return out


class FlattenNet(nn.Module):
    """
    This is the net to encode the patches to point feature.[bs, n, 64]
    """
    def __init__(self, config):
        super(FlattenNet, self).__init__()
        self.fc1 = nn.Linear(config['input_dim'], config['output_dim'], bias=False)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        bs, n, _, _ = x.size()
        x = x.view(bs, n, -1)
        x = self.activate(self.fc1(x))
        return x


class FlattenNet_average(nn.Module):
    """
    This is the net to encode the patches to point feature.[bs, n, 64]
    """
    def __init__(self, config):
        super(FlattenNet_average, self).__init__()
        self.fc1 = nn.Linear(config['input_dim'], config['output_dim'], bias=False)
        self.activate = nn.ReLU(inplace=True)

        self.conv1 = Conv(1, 64, kernel_size=3, stride=1, padding=0)
        # self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        bs, n, patch_size, patch_size = x.size()
        x = x.view(bs*n, 1, patch_size, patch_size)
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = x.view(bs, n, -1)
        return x



def pre_encoder1(img, full_pcd, k):
    """
    To convert the contour point coordinate to patches.
    only contour line is considered.
    :param img: [bs, height, width]
    :param full_pcd: [bs, n, 2]
    :param k: patch_size
    :return: pre-encoded contour feature [bs, n, size, size]
    """
    device = img.device
    full_pcd = full_pcd.long()
    bs, fn, _ = full_pcd.size()
    img = img * 0
    bs_idx = torch.arange(0, bs).repeat_interleave(fn).to(device)
    img[bs_idx, full_pcd[:, :, 0].view(-1), full_pcd[:, :, 1].view(-1)] = 1
    template_map = (torch.zeros((k, k)) == 0).nonzero().to(device)
    x_idx, y_idx = template_map[:, 0], template_map[:, 1]
    x_idx, y_idx = x_idx.repeat(bs, fn, 1), y_idx.repeat(bs, fn, 1)  # [bs, n, 225]
    
    # 这里之后可以降采样，跳着采样（k也需要增大）
    x_idx += (full_pcd[:, :, 0].unsqueeze(-1) - k//2)
    y_idx += (full_pcd[:, :, 1].unsqueeze(-1) - k//2)
    bs_idx = torch.arange(0, bs).repeat_interleave(fn * k**2).to(device)
    c = img[bs_idx, x_idx.view(-1), y_idx.view(-1)].view(bs, fn, k, k)

    return c


def pre_encoder2(img, full_pcd, k):
    """
    To convert the contour poin-t coordinate to patches.
    interior + exterior
    :param img: [bs, height, width]
    :param full_pcd: [bs, n, 2]
    :param k: patch_size
    :return: pre-encoded contour feature [bs, n, size, size]
    """
    device = img.device
    full_pcd = full_pcd.long()
    bs, fn, _ = full_pcd.size()
    bs_idx = torch.arange(0, bs).repeat_interleave(fn).to(device)
    img[bs_idx, full_pcd[:, :, 0].view(-1), full_pcd[:, :, 1].view(-1)] = 1
    template_map = (torch.zeros((k, k)) == 0).nonzero().to(device)
    x_idx, y_idx = template_map[:, 0], template_map[:, 1]
    x_idx, y_idx = x_idx.repeat(bs, fn, 1), y_idx.repeat(bs, fn, 1)  # [bs, n, k*k]
    x_idx += (full_pcd[:, :, 0].unsqueeze(-1) - k//2)
    y_idx += (full_pcd[:, :, 1].unsqueeze(-1) - k//2)
    bs_idx = torch.arange(0, bs).repeat_interleave(fn * k**2).to(device)
    c = img[bs_idx, x_idx.view(-1), y_idx.view(-1)].view(bs, fn, k, k)

    return c


def pre_encoder3(img, full_pcd, k):
    """
    To convert the contour point coordinate to patches.
    interior + contour line + exterior
    :param img: tensor [bs, height, width]
    :param full_pcd: tensor [bs, n, 2]
    :param k: int patch_size
    :return: tensor pre-encoded contour feature [bs, n, size, size]
    """
    device = img.device
    full_pcd = full_pcd.long()
    img = img * 2
    bs, fn, _ = full_pcd.size()
    bs_idx = torch.arange(0, bs).repeat_interleave(fn).to(device)
    img[bs_idx, full_pcd[:, :, 0].view(-1), full_pcd[:, :, 1].view(-1)] = 1
    template_map = (torch.zeros((k, k)) == 0).nonzero().to(device)
    x_idx, y_idx = template_map[:, 0], template_map[:, 1]
    x_idx, y_idx = x_idx.repeat(bs, fn, 1), y_idx.repeat(bs, fn, 1)  # [bs, n, k*k]
    x_idx += (full_pcd[:, :, 0].unsqueeze(-1) - k//2)
    y_idx += (full_pcd[:, :, 1].unsqueeze(-1) - k//2)
    bs_idx = torch.arange(0, bs).repeat_interleave(fn * k**2).to(device)
    c = img[bs_idx, x_idx.view(-1), y_idx.view(-1)].view(bs, fn, k, k)

    return c


def img_patch_encoder(img, full_pcd, k): #  img:1,3,331,318  k: patch size
    device = img.device
    full_pcd = full_pcd.long()
    bs, fn, _ = full_pcd.size() # 1，2778，2
    template_map = (torch.zeros((3, k, k)) == 0).nonzero().to(device) #得到所有True位置的索引 [147,3]
    channel, x_idx, y_idx = template_map[:, 0], template_map[:, 1], template_map[:, 2]
    channel, x_idx, y_idx = channel.repeat(bs, fn, 1), x_idx.repeat(bs, fn, 1), y_idx.repeat(bs, fn, 1)  # [bs, n, k*k]
    x_idx += (full_pcd[:, :, 0].unsqueeze(-1) - k//2)
    y_idx += (full_pcd[:, :, 1].unsqueeze(-1) - k//2)
    bs_idx = torch.arange(0, bs).repeat_interleave(fn*3*k**2).to(device) #408366
    c = img[bs_idx, channel.view(-1), x_idx.view(-1), y_idx.view(-1)].view(bs, fn, 3, k, k) # 1,2778,3,7,7

    return c


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Filter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Filter, self).__init__()
        stride = 1
        self.conv = nn.Sequential(*[
            Conv(in_channels, out_channels, kernel_size=3, stride=stride, activation=True),
            nn.Upsample((112, 112), mode='bilinear', align_corners=False)
        ])

    def forward(self, x):
        y = self.conv(x)
        return y


class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.features_out_hook = []
        self.resnet = resnet50(weights="IMAGENET1K_V2")
        # self.resnet = resnet50(weights=None)
        self.filter1 = Filter(64, 64)
        self.filter2 = Filter(256, 64)
        self.filter3 = Filter(512, 64)
        self.filter4 = Filter(2048, 64)
        # self.fc = nn.Linear(4 * 64, 64)
        self.activate = nn.ReLU(inplace=True)
        self.get_hook_feature()
        self.conv_1 = Conv(4 * 64, 64, kernel_size=3, stride=1, activation=True)
        # self.fc1 = nn.Linear(2048+64, 512)
        # self.fc2 = nn.Linear(512, 64)

    def hook(self, module, fea_in, fea_out):
        self.features_out_hook.append(fea_out.data)  # 勾的是指定层的输出
        return None

    def get_hook_feature(self):
        layer_name = ['maxpool', 'layer1', 'layer2', 'layer4']
        # layer_name = ['maxpool', 'layer1', 'layer2']
        for (name, module) in self.resnet.named_modules():
            if name in layer_name:
                module.register_forward_hook(hook=self.hook)

    def forward(self, x, contour):
        self.features_out_hook = []
        # with torch.no_grad():
        self.resnet(x)
        conv1, conv2, conv3, conv5 = self.features_out_hook
        # conv1, conv2, conv3 = self.features_out_hook

        # bs, n, d = contour.shape
        # global_feature = nn.functional.adaptive_max_pool2d(conv5, (1, 1))
        # global_feature = global_feature.view(bs, 1, -1)
        # global_feature = torch.repeat_interleave(global_feature, n, 1)
        feature_map1 = self.filter1(conv1)
        feature_map2 = self.filter2(conv2)
        feature_map3 = self.filter3(conv3)
        feature_map5 = self.filter4(conv5)

        full_feature = torch.cat((feature_map1, feature_map2, feature_map3, feature_map5), dim=1)
        # full_feature = torch.cat((feature_map1, feature_map2, feature_map3), dim=1)
        full_feature = self.conv_1(full_feature)
        # bs * 256 * 112 * 112
        l_t = get_gcn_feature(full_feature, contour)

        # l_t = torch.cat((l_t, global_feature), dim=-1)
        # l_t = self.activate(self.fc1(l_t))
        # l_t = self.activate(self.fc2(l_t))
        # l_t = self.fc(l_t)

        return l_t


class PatchEncoder(nn.Module):
    def __init__(self):
        super(PatchEncoder, self).__init__()
        self.conv1 = Conv(3, 128, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

class PatchEncoder_average(nn.Module):
    def __init__(self):
        super(PatchEncoder_average, self).__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(32, 64, kernel_size=3, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return x

class AngleEncoder(nn.Module):
    def __init__(self):
        super(AngleEncoder, self).__init__()
        self.fc = nn.Linear(1, 128)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, pcd):
        c = self.pre_encoder(pcd)
        return c

    @staticmethod
    def pre_encoder(pcd):
        bs, n, _ = pcd.size()
        l_s = torch.roll(pcd, -1, dims=1)
        r_s = torch.roll(pcd, 1, dims=1)
        vec_s1 = l_s - pcd
        vec_s2 = r_s - pcd
        dis_s1 = torch.linalg.norm(vec_s1, dim=-1)
        dis_s2 = torch.linalg.norm(vec_s2, dim=-1)
        angel = torch.arccos(
            torch.clip((torch.sum(torch.multiply(vec_s1, vec_s2), -1) + 1e-6) / (dis_s1 * dis_s2 + 1e-6), -1, 1)
        )
        cross = torch.cross(
            torch.cat((vec_s1, torch.zeros(bs, n, 1).to('cuda')), dim=-1),
            torch.cat((vec_s2, torch.zeros(bs, n, 1).to('cuda')), dim=-1),
            dim=-1)
        cross = cross[:, :, 2]
        cross[cross >= 0] = 1
        cross[cross < 0] = -1
        angel = torch.multiply(angel, cross)
        angel[cross < 0] += 2 * 3.1415926
        angel /= (1 * 3.1415926)
        point_feature = angel.unsqueeze(-1)

        return point_feature


class GCNLayer(nn.Module):
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.conv = GATConv(args.in_channels, args.n_filters, heads=args.gat_head, concat=False)
        # self.conv = GCNConv(args.in_channels, args.n_filters)
        if args.norm == 'batch':
            self.norm = nn.BatchNorm1d(args.in_channels)
        elif args.norm == 'layer':
            self.norm = nn.LayerNorm(args.in_channels)
        self.act = nn.ReLU(inplace=True) if args.act == 'relu' else nn.Identity()
        self.deep_gcn = DeepGCNLayer(self.conv, self.norm, self.act, block=args.block)

    def forward(self, x, a):
        return self.deep_gcn(x, a)


class MyDeepGCN(nn.Module):
    def __init__(self, args):
        super(MyDeepGCN, self).__init__()
        self.deep_gcn = Seq(*[GCNLayer(args) for _ in range(args.n_blocks)])

    def forward(self, x, a):
        for block in self.deep_gcn:
            x = block(x, a)
        return x


class GCNLayer_stage2(nn.Module):
    def __init__(self, args):
        super(GCNLayer_stage2, self).__init__()
        self.conv = GATConv(args.in_channels_stage2, args.n_filters_stage2, heads=args.gat_head_stage2, concat=False)
        # self.conv = GCNConv(args.in_channels, args.n_filters)
        if args.norm == 'batch':
            self.norm = nn.BatchNorm1d(args.in_channels_stage2)
        elif args.norm == 'layer':
            self.norm = nn.LayerNorm(args.in_channels_stage2)
        self.act = nn.ReLU(inplace=True) if args.act == 'relu' else nn.Identity()
        self.deep_gcn = DeepGCNLayer(self.conv, self.norm, self.act, block=args.block_stage2)

    def forward(self, x, a):
        return self.deep_gcn(x, a)

class MyDeepGCN_stage2(nn.Module):
    def __init__(self, args):
        super(MyDeepGCN_stage2, self).__init__()
        self.deep_gcn = Seq(*[GCNLayer_stage2(args) for _ in range(args.n_blocks_stage2)])

    def forward(self, x, a):
        for block in self.deep_gcn:
            x = block(x, a)
        return x

class MyDeepGCN2(nn.Module):
    def __init__(self, args):
        super(MyDeepGCN2, self).__init__()
        self.deep_gcn = Seq(*[GCNLayer(args) for _ in range(args.n_blocks+2)])

    def forward(self, x, a):
        for block in self.deep_gcn:
            x = block(x, a)
        return x


if __name__ == "__main__":
    inp = torch.randn((2, 10, 64))
    adj = torch.zeros((10, 10), dtype=torch.int)
    adj[2, 2] = 1
    adj = adj.to_sparse().indices()

    net = MyDeepGCN()
    out = net(inp, adj)
