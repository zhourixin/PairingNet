import argparse
import os
flattenNet_config = {
    'input_dim': 9,  # 3**2 / 7*7 / 11*11
    'hidden': 32,
    'output_dim': 64,
    'dropout': 0.0,
    'k': 3
}


args = argparse.ArgumentParser()
'''model_config'''
args.add_argument('--patch_size', type=int, default=7)  # 3x3, 7x7, 11x11
args.add_argument('--c_model', type=str, default='l', help='contour encode method')  # l, io, ilo
args.add_argument('--loss',  type=str, default='focal', help='loss function type')
args.add_argument('--feature_dim', type=int, default=64, help='output feature dim')

'''GCN setting'''
args.add_argument('--k', default=16, type=int, help='neighbor num (default:16)')
args.add_argument('--block', default='res+', type=str, help='graph backbone block type {plain, res, dense}')
args.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
args.add_argument('--norm', default='batch', type=str, help='{batch, instance, None} normalization')
args.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
args.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
args.add_argument('--n_blocks', default=14, type=int, help='number of basic blocks')
args.add_argument('--in_channels', default=64, type=int, help='the channel size of input point cloud ')
args.add_argument('--dropout', default=0.2, type=float, help='ratio of dropout')
args.add_argument('--gat_head', default=1, type=int, help='GATConv head numbers')

'''G-Unet setting'''
args.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])


'dilated knn'
args.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
args.add_argument('--stochastic', default=True, type=bool, help='stochastic for gcn, True or False')

'''training parameter'''
args.add_argument('--flattenNet_config', default=flattenNet_config, help='network make patch 2 feature')
args.add_argument('--channel', type=int, default=3, help='img channel')
args.add_argument('--epoch', type=int, default=128, help='train epoch')
args.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args.add_argument('--weight_decay', type=float, default=5e-4, help='l2 normalization')
args.add_argument('--matching_batch_size', type=int, default=25) #
args.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoints or not')
args.add_argument('--max_length', type=int, default=2900)

'''stage2 deep GCN'''
args.add_argument('--n_blocks_stage2', default=12, type=int, help='number of stage2 basic blocks')
args.add_argument('--block_stage2', default='res+', type=str, help='graph backbone block type {plain, res, dense}')
args.add_argument('--n_filters_stage2', default=128, type=int, help='number of channels of deep features')
args.add_argument('--gat_head_stage2', default=1, type=int, help='GATConv head numbers')
args.add_argument('--matching_channels_stage2', default=128, type=int, help='the channel size of input point cloud ')
args.add_argument('--stage2_matching_weight', default=0.05, type=float, help='contrast loss weight')
args.add_argument('--stage2_lr', type=float, default=1e-3, help='learning rate')
args.add_argument('--stage2_epoch', type=int, default=128, help='train epoch')
args.add_argument('--warmup_steps', type=int, default=10, help='train epoch')
args.add_argument('--stage2_weight_decay', type=float, default=1e-3, help='l2 normalization')

args.add_argument('--global_out_channels', default=128, type=int, help='the channel size of global output')
args.add_argument('--contrast_weight', default=1, type=float, help='contrast loss weight')
args.add_argument('--contrast_temperature', default=0.12, type=float, help='contrast loss weight')
args.add_argument('--local_rank', default=0, type=int)

args.add_argument('--embed_dim', default=768, type=int, help='the embed_dim of vit-b 16')
args.add_argument('--vit_length', default=196, type=int, help='the vit_length of vit-b 16')
args.add_argument('--tranct_length', default=1408, type=int, help='the length of truncation')



'''about path downsample'''
dataset_select =390

args.add_argument('--dataset_select', type=int, default=dataset_select, help='select dataset')
args.add_argument('--train_set', type=str, default='./{}/train_set_with_downsample.pkl'.format(dataset_select), help='train data path')
args.add_argument('--valid_set', type=str, default='./{}/valid_set_with_downsample.pkl'.format(dataset_select), help='valid data path')
args.add_argument('--test_set', type=str, default='./{}/test_set_with_downsample.pkl'.format(dataset_select), help='test data path')
args.add_argument('--search_set', type=str, default='./{}/test_set_with_downsample.pkl'.format(dataset_select), help='searching data path')

args.add_argument('--in_channels_stage2', default=128, type=int, help='the channel size of input point cloud ')
# stage2_data_model = "unmerged"
stage2_data_model = "merged"
args.add_argument('--stage2_data_model', type=str, default=stage2_data_model, help='select dataset')
args.add_argument('--stage2_feature_path', type=str, default='./stage1_feature/{}'.format(stage2_data_model), help='searching data path')
exp_path = os.getcwd()
args.add_argument('--exp_path', type=str, default="./", help='exp path')
args.add_argument('--model_type', type=str, default='searching', help='matching_train or matching_test or save_matching_feature or searching_train or searching_test')

args = args.parse_args()
args.in_channels = args.feature_dim


