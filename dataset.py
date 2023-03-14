import random

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import glob
import numpy as np
import torch

import torch
import random
import numpy as np
from scipy.spatial.transform import Rotation as R


class RandomFlip3D:
    def __init__(self, prob=0.5):
        """
        构造函数，初始化旋转角度和旋转轴的范围
        :param prob :发生旋转的概率
        :param axis: 绕某一个轴翻转

        """
        self.prob = prob

    def __call__(self, tensor):
        """
        随机选择一个轴翻转3D张量
        :param tensor: 3D张量，形状为(depth, height, width)
        :return: 旋转后的3D张量
        """

        if np.random.rand() < self.prob:
            data = tensor.numpy()
            # 随机选择一个轴进行旋转
            axis = random.choice((0, 1, 2))
            data = np.flip(data, axis=axis).copy()
            return torch.from_numpy(data)
        else:
            return tensor


class RandomCrop3D(object):
    def __init__(self, volume_sz, cropVolume_sz):
        c, d, h, w = volume_sz
        assert (d, h, w) >= cropVolume_sz
        self.volume_sz = tuple((d, h, w))
        self.cropVolume_sz = tuple(cropVolume_sz)

    def __call__(self, volume_list):
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.volume_sz, self.cropVolume_sz)]
        return self._crop(volume_list, *slice_dhw)

    @staticmethod
    def _get_slice(volume_sz, cropVolume_sz):
        try:
            lower_bound = torch.randint(volume_sz - cropVolume_sz, (1,)).item()
            return lower_bound, lower_bound + cropVolume_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(volume_list, slice_d, slice_h, slice_w):
        res = []
        for volume in volume_list:
            res.append(volume[:, slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]])
        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)


class VolumesDataset(Dataset):
    def __init__(self, data_source_path, dim, dsam_scale_factor=(0.25, 0.25, 0.25),
                 transform=None, seq_num=5):
        """:argument
            data_source_path: the directory of the volume data
            dim: the dimension of the volume data
            crop_scale_factor: the scaling factor of cropping
            dsam_scale_factor: the scaling factor of down-sampling
            transform: the transform of volume data

        """
        super(VolumesDataset, self).__init__()
        self.data_source_path = data_source_path
        self.dim = dim

        self.dsam_scale_factor = dsam_scale_factor
        self.transform = transform

        self.data_paths = sorted(glob.glob(data_source_path + '/*.raw'))

        self.length = len(self.data_paths)
        self.seq_num = seq_num
        if seq_num == 0:
            self.seq_num = self.length

        # self.idx_list = [i for i in range(1, len(self.data_paths) - 1)]  # for gan

    def __len__(self):
        # return len(self.idx_list)
        return self.length - self.seq_num + 1

    def __getitem__(self, t):
        return self.read_volumes(t)

    def read_volumes(self, t):
        high_res_list = []
        for i in range(t, t + self.seq_num):
            data_path = self.data_paths[i]
            high_res = torch.from_numpy(np.fromfile(data_path, dtype=np.float32))
            high_res = high_res.view([self.dim[0], self.dim[1], self.dim[2]])
            if self.transform:
                high_res = self.transform(high_res)
            high_res = high_res.unsqueeze(0)
            high_res_list.append(high_res)

        high_res_seq = high_res_list[0].unsqueeze(0)
        low_res_seq = nn.functional.interpolate(high_res_seq, scale_factor=self.dsam_scale_factor,
                                                mode='trilinear')

        for i in range(1, len(high_res_list)):
            high_res = high_res_list[i].unsqueeze(0)
            low_res = nn.functional.interpolate(high_res, scale_factor=self.dsam_scale_factor,
                                                mode='trilinear')
            high_res_seq = torch.cat([high_res_seq, high_res], dim=0)
            low_res_seq = torch.cat([low_res_seq, low_res], dim=0)

        return low_res_seq, high_res_seq

    def read1volume(self, t):
        # (1) get data path
        data_path = self.data_paths[t]
        # (2) read data from path
        high_res = torch.from_numpy(np.fromfile(data_path, dtype=np.float32))
        # (3) reshape
        high_res = high_res.view([1, self.dim[0], self.dim[1], self.dim[2]])
        # (4) crop original data to get crop data.
        if self.transform:
            high_res = self.transform([high_res])

        high_res = high_res.unsqueeze(0)
        # low_res = nn.functional.interpolate(high_res.unsqueeze(0), scale_factor=self.dsam_scale_factor,
        #                                     mode='trilinear').squeeze(0)
        low_res = nn.functional.interpolate(high_res, scale_factor=self.dsam_scale_factor,
                                            mode='trilinear')

        return low_res, high_res

    def read3volumes(self, t):
        # (1) get data path
        prev_data_path = self.data_paths[self.idx_list[t] - 1]
        t_data_path = self.data_paths[self.idx_list[t]]
        next_data_path = self.data_paths[self.idx_list[t] + 1]

        # (2) read data from path
        prev_high_res = torch.from_numpy(np.fromfile(prev_data_path, dtype=np.float32))
        t_high_res = torch.from_numpy(np.fromfile(t_data_path, dtype=np.float32))
        next_high_res = torch.from_numpy(np.fromfile(next_data_path, dtype=np.float32))

        # (3) reshape
        prev_high_res = prev_high_res.view([1, self.dim[0], self.dim[1], self.dim[2]])
        t_high_res = t_high_res.view([1, self.dim[0], self.dim[1], self.dim[2]])
        next_high_res = next_high_res.view([1, self.dim[0], self.dim[1], self.dim[2]])

        # (4) crop original data to get crop data.
        if self.transform:
            prev_high_res, t_high_res, next_high_res = self.transform(
                [prev_high_res, t_high_res, next_high_res])

        # (5) concatenate this three consecutive volumes
        high_res_volumes = torch.cat([prev_high_res, t_high_res, next_high_res], dim=0)

        # (6) downsample cropped data and concatenate them
        prev_low_res = nn.functional.interpolate(prev_high_res.unsqueeze(0), scale_factor=self.dsam_scale_factor,
                                                 mode='trilinear').squeeze_(0)
        t_low_res = nn.functional.interpolate(t_high_res.unsqueeze(0), scale_factor=self.dsam_scale_factor,
                                              mode='trilinear').squeeze_(0)
        next_low_res = nn.functional.interpolate(prev_high_res.unsqueeze(0), scale_factor=self.dsam_scale_factor,
                                                 mode='trilinear').squeeze_(0)
        low_res_volumes = torch.cat([prev_low_res, t_low_res, next_low_res], dim=0)

        return low_res_volumes, high_res_volumes


if __name__ == '__main__':
    dataset = VolumesDataset()
