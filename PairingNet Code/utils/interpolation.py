import torch


def bilinear_interpolation(x, y, feature_map, eps=1e-6):
    torch.clip_(x, min=eps, max=feature_map.shape[-1] - 1)
    torch.clip_(y, min=eps, max=feature_map.shape[-1] - 1)
    # feature_map = F.pad(feature_map, (0, 1, 0, 1, 0, 0))
    bs, c, _, _ = feature_map.shape
    nums = len(x[0])
    x = x.to('cuda')
    y = y.to('cuda')
    x_1 = torch.floor(x)
    y_1 = torch.floor(y)
    x_2 = torch.ceil(x)
    y_2 = torch.ceil(y)
    x_1, x_2, y_1, y_2 = x_1.long(), x_2.long(), y_1.long(), y_2.long()
    expend_idx = torch.arange(0, bs).repeat_interleave(nums, dim=0)
    max_idx = feature_map.shape[-1] - 1
    p00 = feature_map[expend_idx, :, x_1.clip(eps, max_idx).view(1, -1), y_1.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    p01 = feature_map[expend_idx, :, x_1.clip(eps, max_idx).view(1, -1), y_2.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    p10 = feature_map[expend_idx, :, x_2.clip(eps, max_idx).view(1, -1), y_1.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    p11 = feature_map[expend_idx, :, x_2.clip(eps, max_idx).view(1, -1), y_2.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    x_1, x_2, y_1, y_2 = x_1.type_as(p00), x_2.type_as(p00), y_1.type_as(p00), y_2.type_as(p00)
    m = (x_2-x)
    n = (x-x_1)
    j = (y_2-y)
    k = (y-y_1)
    feature = \
        torch.multiply(m, j).view(bs, -1, 1) * p00 + \
        torch.multiply(m, k).view(bs, -1, 1) * p10 + \
        torch.multiply(n, j).view(bs, -1, 1) * p01 + \
        torch.multiply(n, k).view(bs, -1, 1) * p11

    return feature


def ibw_interpolation(feature_map, contour, eps=torch.tensor(1e-6).to('cuda')):
    x = contour[:, :, 0]
    y = contour[:, :, 1]
    torch.clip_(x, min=eps, max=feature_map.shape[-1] - 1)
    torch.clip_(y, min=eps, max=feature_map.shape[-1] - 1)
    nums = len(x[0])
    bs, c, _, _ = feature_map.shape
    one = torch.tensor(1.).to('cuda')
    x1, x2, x3, y1, y2, y3 = \
        torch.floor(x)-one, torch.floor(x), torch.floor(x) + one, \
        torch.floor(y)-one, torch.floor(y), torch.floor(y) + one
    x1, x2, x3, y1, y2, y3 = x1.long(), x2.long(), x3.long(), y1.long(), y2.long(), y3.long()
    expend_idx = torch.arange(0, bs).repeat_interleave(nums, dim=0)
    max_idx = feature_map.shape[-1] - 1
    q_lup = feature_map[expend_idx, :, x1.clip(eps, max_idx).view(1, -1), y1.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_up = feature_map[expend_idx, :, x1.clip(eps, max_idx).view(1, -1), y2.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_rup = feature_map[expend_idx, :, x1.clip(eps, max_idx).view(1, -1), y3.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_l = feature_map[expend_idx, :, x2.clip(eps, max_idx).view(1, -1), y1.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_c = feature_map[expend_idx, :, x2.clip(eps, max_idx).view(1, -1), y2.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_r = feature_map[expend_idx, :, x2.clip(eps, max_idx).view(1, -1), y3.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_lb = feature_map[expend_idx, :, x3.clip(eps, max_idx).view(1, -1), y1.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_b = feature_map[expend_idx, :, x3.clip(eps, max_idx).view(1, -1), y2.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    q_rb = feature_map[expend_idx, :, x3.clip(eps, max_idx).view(1, -1), y3.clip(eps, max_idx).view(1, -1)].view(bs, -1, c)
    x1, x2, x3, y1, y2, y3 = x1 + one/2, x2 + one/2, x3 + one/2, y1 + one/2, y2 + one/2, y3 + one/2
    dist_lup = 1 / (torch.sqrt((x1 - x)**2 + (y1 - y) ** 2) + eps)
    dist_up = 1 / (torch.sqrt((x1 - x)**2 + (y2 - y) ** 2) + eps)
    dist_rup = 1 / (torch.sqrt((x1 - x)**2 + (y3 - y) ** 2) + eps)
    dist_l = 1 / (torch.sqrt((x2 - x)**2 + (y1 - y) ** 2) + eps)
    dist_c = 1 / (torch.sqrt((x2 - x)**2 + (y2 - y) ** 2) + eps)
    dist_r = 1 / (torch.sqrt((x2 - x)**2 + (y3 - y) ** 2) + eps)
    dist_lb = 1 / (torch.sqrt((x3 - x)**2 + (y1 - y) ** 2) + eps)
    dist_b = 1 / (torch.sqrt((x3 - x)**2 + (y2 - y) ** 2) + eps)
    dist_rb = 1 / (torch.sqrt((x3 - x)**2 + (y3 - y) ** 2) + eps)
    dist_sum = dist_lup + dist_up + dist_rb + dist_b + dist_lb + dist_r + dist_l + dist_c + dist_rup
    feature = \
        (dist_lup.view(bs, -1, 1) * q_lup + dist_up.view(bs, -1, 1) * q_up + dist_rup.view(bs, -1, 1) * q_rup +
         dist_l.view(bs, -1, 1) * q_l + dist_c.view(bs, -1, 1) * q_c + dist_r.view(bs, -1, 1) * q_r +
         dist_lb.view(bs, -1, 1) * q_lb + dist_b.view(bs, -1, 1) * q_b + dist_rb.view(bs, -1, 1) * q_rb) / \
        dist_sum.view(bs, -1, 1)
    return feature


def get_gcn_feature(cnn_feature, img_poly):
    """
    :param cnn_feature: bs, c, w, h
    :param img_poly: bs, n, 2
    :return:
    """
    img_poly = img_poly.clone()
    img_poly = img_poly.unsqueeze(1)
    feature = torch.nn.functional.grid_sample(cnn_feature, img_poly).squeeze(2).permute(0, 2, 1)

    return feature
