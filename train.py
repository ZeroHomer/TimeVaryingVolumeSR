from torch import nn

import os

from torch.utils.data import DataLoader
from torchvision import transforms

from model import ConvLSTMVSR, ConvGRUVSR

from dataset import VolumesDataset, RandomFlip3D
from ssim import ssim, ms_ssim
import argparse
import torch
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR

# Training settings


parser = argparse.ArgumentParser(description="PyTorch SSRAN")
parser.add_argument("--train_data_dir", type=str, help="the directory of training data ")
parser.add_argument("--val_data_dir", default="", type=str, help="the directory of validation data ")

parser.add_argument("--dataset_name", type=str, help="the name of dataset")
parser.add_argument("--dim", type=str, help="the dimension of the volume data")
parser.add_argument("--scale_factor", type=int, default=4, help="the factor of downsampling and upsampling")

parser.add_argument("--train_batch_size", type=int, default=3, help="training batch size")
parser.add_argument("--val_batch_size", type=int, default=30, help="validating batch size")

parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs to train for")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=1e-4")

parser.add_argument("--lambda_content", type=float, default=1, help="pixel-wise loss weight Default=1")
parser.add_argument("--lambda_ssim", type=float, default=1, help="ssim loss weight Default=1")
parser.add_argument("--attn", action="store_true", help="use attention mechanism or not")

parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--psnr", action="store_true", help="Use psnr?")
parser.add_argument("--ssim", action="store_true", help="Use ssim?")

parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")

parser.add_argument("--block_num", default=3, type=int)

warnings.filterwarnings("ignore")


def PSNR(y_pred, y):
    # normalize
    diff = y_pred - y
    mse = torch.mean(torch.square(diff))
    return (10 * torch.log10(1 / mse)).item()


def save_checkpoint(epoch, net, optimizer, loss, psnr, ssim, net_name, dataset_name, addition=''):
    checkpoint_dir = f"checkpoint_{net_name}_{dataset_name + addition}/"
    model_out_path = checkpoint_dir + "{}_epoch_{}.pth".format(net_name, epoch)

    if isinstance(net, torch.nn.DataParallel):
        state = {"epoch": epoch, "model": net.module.state_dict(), 'optimizer': optimizer.state_dict(),
                 'loss': loss, 'psnr': psnr, "ssim": ssim}
    else:
        state = {"epoch": epoch, "model": net.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss,
                 'psnr': psnr, "ssim": ssim}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def criterion_ssim(y_pred, y):
    # ssim_loss = 1 - ssim(y_pred, y, data_range=1, size_average=True)
    ssim_loss = 1 - ms_ssim(y_pred, y, data_range=1, win_size=3)
    return ssim_loss


def SSIM(y_pred, y):
    ssim_val = ssim(y_pred, y, data_range=1.0, size_average=True).item()
    return ssim_val


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    net_name = 'ConvGRUVSR'
    device = None
    print("===>Setting GPU")
    if opt.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(device)

    print("===>Set dataset configuration")
    dataset_name = opt.dataset_name

    dim = eval(opt.dim)
    dsam_scale_factor = (1 / opt.scale_factor, 1 / opt.scale_factor, 1 / opt.scale_factor)

    print("===>Loading data")
    transform = transforms.Compose({
        RandomFlip3D(0.5)
    })
    seq_num = 6
    train_set = VolumesDataset(opt.train_data_dir, dim, dsam_scale_factor, transform, seq_num=seq_num)
    val_set = VolumesDataset(opt.val_data_dir, dim, dsam_scale_factor, seq_num=0)
    train_loader = DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    print("===>Building model")
    hidden_channels = [2, 4, 8]
    num_layers = len(hidden_channels)
    block_num = opt.block_num
    norm_dim = (dim[0] // opt.scale_factor, dim[1] // opt.scale_factor, dim[2] // opt.scale_factor)
    net = ConvGRUVSR(hidden_channels=hidden_channels, num_layers=num_layers, block_num = block_num, factor=opt.scale_factor, use_attn=opt.attn)
    net = net.to(device)

    print("===>Setting optimizers and loss functions")
    optimizer = torch.optim.RAdam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-4)
    criterion_content = nn.L1Loss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, verbose=True)
    min_epoch_loss, max_psnr, max_ssim = 1e9, -1e9, 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=device)
            opt.start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            min_epoch_loss = checkpoint["loss"]
            max_psnr = checkpoint['psnr']
            max_ssim = checkpoint['ssim']

    print("===>Training")
    n_epochs = opt.n_epochs
    start_epoch = opt.start_epoch
    print_step = 2
    patience = 0

    for epoch in range(1, n_epochs + 1 - start_epoch):
        print("============================================================================================")
        print(f"Epoch: {epoch + start_epoch}")
        print("============================================================================================")
        epoch_loss, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
        print("Training")
        net.train()
        net = net.to(device)
        for step, data in enumerate(train_loader, 1):
            if step % print_step == 0:
                print("********************************************************************************************")
                print("step:", step)

            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_pred = net(x)
            content_loss, ssim_loss, total_loss = 0, 0, 0
            optimizer.zero_grad()
            for t in range(seq_num):
                content_loss += criterion_content(y_pred[:, t, :, :, :, :], y[:, t, :, :, :, :])
                ssim_loss += criterion_ssim(y_pred[:, t, :, :, :, :], y[:, t, :, :, :, :])
            total_loss = opt.lambda_content * content_loss + opt.lambda_ssim * ssim_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            if step % print_step == 0:
                print("Total loss: {} Content loss: {} SSIM loss:{}".format(total_loss.item() / seq_num,
                                                                            content_loss.item() / seq_num,
                                                                            ssim_loss.item() / seq_num))
        print("********************************************************************************************")
        print("Testing")
        net.eval()
        net = net.to(device)
        with torch.no_grad():
            seq_len = 0
            for step, data in enumerate(val_loader, 1):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                gen_y = net(x)
                # switch batch dimension and seq_len dimension
                gen_y = gen_y.transpose(0, 1)
                y = y.transpose(0, 1)
                seq_len = y.size(0)
                for i in range(seq_len):
                    psnr_val = PSNR(gen_y[i], y[i])
                    ssim_val = SSIM(gen_y[i], y[i])
                    print(f"[{i}] PSNR: {psnr_val} SSIM: {ssim_val}")
                    epoch_psnr += psnr_val
                    epoch_ssim += ssim_val

            epoch_psnr = epoch_psnr / step / seq_len
            epoch_ssim = epoch_ssim / step / seq_len
            print('PSNR: {}  SSIM:{}'.format(epoch_psnr, epoch_ssim))

            if epoch_psnr <= max_psnr and epoch_ssim <= max_ssim:
                patience += 1
            else:
                patience = 0
            print('Patience:', patience)

        # save the best
        addition = '_' + str(block_num)+'_block'
        if not opt.attn:
            addition = addition + '_no_attn'
        if opt.lambda_content == 0:
            addition = addition + '_no_content'
        if opt.lambda_ssim == 0:
            addition = addition + '_no_ssim'

        if epoch_psnr > max_psnr:
            if opt.psnr:
                save_checkpoint(epoch + start_epoch, net, optimizer, epoch_loss, epoch_psnr, epoch_ssim, net_name,
                                dataset_name, addition)
                pass
            max_psnr = epoch_psnr

        if epoch_ssim > max_ssim:
            if opt.ssim:
                save_checkpoint(epoch + start_epoch, net, optimizer, epoch_loss, epoch_psnr, epoch_ssim, net_name,
                                dataset_name, addition)
                pass
            max_ssim = epoch_ssim

        print("Max PSNR: {}, Max SSIM: {} ".format(max_psnr, max_ssim))

        if epoch_loss < min_epoch_loss:
            print("Loss decreased", min_epoch_loss - epoch_loss)
            min_epoch_loss = epoch_loss

        if (epoch + start_epoch) % 100 == 0:
            save_checkpoint(epoch + start_epoch, net, optimizer, epoch_loss, epoch_psnr, epoch_ssim, net_name,
                            dataset_name, addition)

print("===>Training finished")
