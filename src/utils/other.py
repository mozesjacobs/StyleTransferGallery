import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import sys
sys.path.append('../')

from src.dataset.Datasets import FeaturesUCFDataset

def load_features_ucf(args):
    train_set = FeaturesUCFDataset(fpath='../clip_data/training_features.npz',
                                   labels=args.labels)
    test_set = FeaturesUCFDataset(fpath='../clip_data/testing_features.npz',
                                  labels=args.labels,
                                  mean=train_set.norm_mean,
                                  std=train_set.norm_std)
    return train_set, test_set

def make_model(args, device):
    if args.model == "PredCodeNetFull":
        from src.models.PredCodeNetFull import Net
        return Net(args, device)

def load_decoder(args):
    if args.decoder == "LinearDecoder":
        from src.models.decoder import LinearDecoder
        return LinearDecoder(args.r_dim, args.frame_size, args.hidden_dim, args.channels)

def convert_labels_to_str(labels):
    if labels is None:
        return 'all'
    else:
        res = ''
        for l in labels:
            res += str(l)
        return res

def get_path_extra(args):
    labels = convert_labels_to_str(args.labels)
    return labels

def get_checkpoint_path(args):
    cp_folder = args.model_folder + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    cp_folder += args.decoder + "/" + args.session_name + "/"
    return cp_folder + 'checkpoint.pt', cp_folder

def get_args_path(args):
    args_folder = args.model_folder + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    args_folder += args.decoder + "/" + args.session_name + "/"
    return args_folder + "args.txt", args_folder

def get_tb_path(args):
    train_name = args.tensorboard_folder + args.data_set + '/' + get_path_extra(args) + "/" + args.model + "/"
    train_name += args.decoder + "/" + args.session_name + '/train'
    test_name = args.tensorboard_folder + args.data_set + '/' + get_path_extra(args) + "/" + args.model + "/"
    test_name += args.decoder + "/" + args.session_name + '/val'
    return train_name, test_name

def get_experiments_path(args):
    exp_path = args.experiments + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    return exp_path + args.decoder + "/" + args.session_name + "/"