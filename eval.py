import warnings
import argparse
import os

import numpy as np

from model import  ConvGRUVSR
import torch

from torch.utils.data import DataLoader

from dataset import VolumesDataset

parser = argparse.ArgumentParser(description="PyTorch SRNet")
parser.add_argument("--data_dir", type=str, help="the directory of data ")
parser.add_argument("--dataset_name", type=str, help="the name of dataset")
parser.add_argument("--batch_size", type=int, default=1,
                    help="training batch size")
parser.add_argument("--model_checkpoint_path", type=str,
                    default="", help="the pth path of saved model")
parser.add_argument("--dim", type=str, help="the dimension of the volume data")
parser.add_argument("--sample_num", type=int, default=4,
                    help="the number of basic block layers in the model")
parser.add_argument("--downsample_factor", type=tuple,
                    default=(0.25, 0.25, 0.25), help="the factor of downsampling")

parser.add_argument("--attn", action="store_true", help="use attention mechanism or not")
parser.add_argument("--lambda_content", type=float, default=1, help="pixel-wise loss weight Default=1")
parser.add_argument("--lambda_ssim", type=float, default=1, help="ssim loss weight Default=1")
parser.add_argument("--block_num", default=3, type=int)
warnings.filterwarnings("ignore")


def PSNR(y_pred, y):
    # normalize
    diff = y_pred - y
    mse = torch.mean(torch.square(diff))
    return (10 * torch.log10(1 / mse)).item()


def SSIM(y_pred, y):
    from ssim import ssim
    ssim_val = ssim(y_pred, y, data_range=1.0, size_average=True).item()
    return ssim_val


if __name__ == '__main__':

    net_name = 'ConvGRUVSR'
    opt = parser.parse_args()

    print("===>Setting GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parser.parse_args()
    print("===>Setting dataset configuration")
    dataset_name = opt.dataset_name
    dim = eval(opt.dim)
    hidden_channels = [2, 4, 8]
    num_layers = len(hidden_channels)
    block_num = opt.block_num
    net = ConvGRUVSR(hidden_channels=hidden_channels, num_layers=num_layers, block_num = block_num, factor=4, use_attn=opt.attn)
    net = net.to(device)
    
    dsam_scale_factor = opt.downsample_factor

    print("===>Building model")
    net = net.to(device)
    net.eval()
    print("===>Loading model parameters")
    state_dict = torch.load(opt.model_checkpoint_path, map_location=device)
    net.load_state_dict(state_dict['model'])
    print("===>Loading data")
    test_set = VolumesDataset(opt.data_dir, dim, dsam_scale_factor, seq_num=0)

    test_loader = DataLoader(test_set, batch_size=1)

    with torch.no_grad():
        dir_name = 'predicted_'+dataset_name+'_'+net_name        
        addition = '_' + str(block_num)+'_block'
        if not opt.attn:
            addition = addition + '_no_attn'
        if opt.lambda_content == 0:
            addition = addition + '_no_content'
        if opt.lambda_ssim == 0:
            addition = addition + '_no_ssim'
        dir_name += addition+'/'
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        if dataset_name == 'SquareCylinder':
            for data in test_loader:
                x, _ = data
                x = x.to(device) 
                y_pred = net(x)
                # squeeze the batch size demension 
                y_pred = y_pred.squeeze(0)
                seq_len = len(y_pred)
                for i in range(seq_len):
                    file_path = dir_name + f"normInten_{i * 40 + 2848}.raw"
                    data = y_pred[i].cpu().numpy()
                    data = np.float32(data)
                    print('file saved to', file_path)
                    data.tofile(file_path)
                print("Done!")
                
        elif dataset_name == 'viscousFingers_smoothlen0.30_run01':
            for data in test_loader:
                x, _ = data
                x = x.to(device)
                y_pred = net(x)
                # squeeze the batch size demension 
                y_pred = y_pred.squeeze(0)
                seq_len = len(y_pred)
                for i in range(seq_len):
                    file_path = dir_name + "normInten_{:0>4d}.raw".format(i + 85)
                    print('file saved to', file_path)
                    y_pred[i].cpu().numpy().tofile(file_path)
                print("Done!")

        elif dataset_name == 'ionization_ab_H':
            for data in test_loader:
                x, _ = data
                x = x.to(device)
                y_pred = net(x)

                # squeeze the batch size demension 
                y_pred = y_pred.squeeze(0)    
                seq_len = len(y_pred)
                for i in range(seq_len):
                    file_path = dir_name + "normInten_{:0>4d}.raw".format(i + 70)
                    print('file saved to', file_path)
                    data = y_pred[i].cpu().numpy()
                    data = np.float32(data)
                    data.tofile(file_path)
                print("Done!")

        elif dataset_name == 'hurricane_wind':
            for data in test_loader:
                x, _ = data
                x = x.to(device)
                y_pred = net(x)
                # squeeze the batch size demension 
                y_pred = y_pred.squeeze(0)
                seq_len = len(y_pred)
                for i in range(seq_len):
                    
                    file_path = dir_name + f"normInten_{35 + i}.raw"
                    data = y_pred[i].cpu().numpy()
                    data = np.float32(data)
                    data.tofile(file_path)
                    print('file saved to', file_path)

                print("Done!")

