import torch
import torch.nn.functional as F
from torch import nn
# __all__ = ['InfoNCE', 'info_nce']

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.07, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, gt_pairs=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode, GT_pairs=gt_pairs)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', GT_pairs=None):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)



        logits = logits/temperature
        mask = get_mask(logits, GT_pairs)
        # print(mask)
        return compute_loss(logits,mask).mean()
        

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



def get_mask(logits,GT_pairs):
    s_index_list = GT_pairs[0]
    t_index_list = GT_pairs[1]


    s_t_dict = {}
    mask = torch.eye(s_index_list.shape[0], s_index_list.shape[0]).to(logits.device)
    for index_source in s_index_list:
        s_t_dict[index_source.item()]= {}
        location_list = torch.where(s_index_list==index_source)
        s_t_dict[index_source.item()][-1] = location_list
    i = 0
    for index_source in s_index_list:
        index_target = t_index_list[i]
        # for index_target in t_index_list:
        s_t_dict[index_source.item()][index_target.item()] = torch.where(t_index_list==index_target)
        i = i + 1
    for key1 in s_t_dict.keys():
        key_dict = s_t_dict[key1]
        row_list = key_dict[-1][0]
        del key_dict[-1]
        for key2 in key_dict.keys():
            coloum_list = key_dict[key2][0]
            for ii in row_list:
                for jj in coloum_list:
                    mask[ii,jj] = 1


    return mask 

def compute_loss(logits,mask):
    # import ipdb
    # ipdb.set_trace()
    return - torch.log((F.softmax(logits,dim=1) * mask).sum(1))

def index_tensor(tnsr):
    result = []
    for i in range(tnsr.shape[0]):
        if tnsr[i] in tnsr[:i]:
            # a = (tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0]
            result.append((tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0].item())
        else:
            result.append(i)
    return torch.tensor(result)



class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, query, positive_key, GT_pairs):
        neg_index = self.get_negative_index(query, GT_pairs)

        anchor = query
        positive = positive_key
        negative = positive_key[neg_index]

        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    def get_negative_index(self, query,GT_pairs):
        s_index = GT_pairs[2]
        t_index = GT_pairs[3]
        s_list = index_tensor(s_index)
        t_list = index_tensor(t_index)

        mask = torch.eye(query.shape[0], query.shape[0]).to(query.device)
        mask[s_list, t_list] = 1

        result = torch.zeros(mask.size(0), dtype=torch.long)
        selected_indices = set()
        for i, row in enumerate(mask):
            zero_indices = (row == 0).nonzero(as_tuple=True)[0]
            zero_indices_select = [index for index in zero_indices if index.item() not in selected_indices]
            if len(zero_indices_select) > 0:
                selected_index = zero_indices[torch.randint(len(zero_indices), (1,))]
                result[i] = selected_index
                selected_indices.add(selected_index.item())
            elif len(zero_indices_select) == 0:
                zero_indices = (row == 0).nonzero(as_tuple=True)[0]
                selected_index = zero_indices[torch.randint(len(zero_indices), (1,))]
                result[i] = selected_index
                selected_indices.add(selected_index.item())
            else: 
                return False



        return result 