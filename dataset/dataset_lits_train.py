from os.path import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from PIL import Image


# 假设 transforms 模块存在
# from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class Train_Dataset(dataset):
    def __init__(self, args):
        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        # 设置初始CT值范围
        self.ct_min = float('inf')
        self.ct_max = float('-inf')
        # 设置默认的归一化范围
        self.normalize_min = -300
        self.normalize_max = 300

    def preprocess_ct(self, ct_array):
        """预处理CT图像"""
        # 1. 更新CT值范围
        self.ct_min = min(self.ct_min, ct_array.min())
        self.ct_max = max(self.ct_max, ct_array.max())

        # 2. 截断到归一化范围
        ct_array = np.clip(ct_array, self.normalize_min, self.normalize_max)

        # 3. 归一化到[0,1]范围
        ct_array = (ct_array - self.normalize_min) / (self.normalize_max - self.normalize_min)

        # 4. 检查归一化结果
        if ct_array.min() < 0 or ct_array.max() > 1:
            print(f"Warning: Normalized data out of range [{ct_array.min():.3f}, {ct_array.max():.3f}]")
            print(f"Current CT value range: [{self.ct_min}, {self.ct_max}]")
            print(f"Normalization range: [{self.normalize_min}, {self.normalize_max}]")
            ct_array = np.clip(ct_array, 0, 1)

        # 5. 打印数据统计信息
        print(f"CT array stats - min: {ct_array.min():.3f}, max: {ct_array.max():.3f}, mean: {ct_array.mean():.3f}")

        return ct_array.astype(np.float32)

    def validate_dimensions(self, ct_array, seg_array, index):
        """验证数据维度"""
        # 1. 检查数据有效性
        if ct_array.size == 0 or seg_array.size == 0:
            raise ValueError(f"Empty array at index {index}")

        # 2. 检查数据维度
        if len(ct_array.shape) != 3 or len(seg_array.shape) != 3:
            raise ValueError(
                f"Invalid data dimensions at index {index}. "
                f"Expected 3D arrays, got CT: {len(ct_array.shape)}D ({ct_array.shape}), "
                f"seg: {len(seg_array.shape)}D ({seg_array.shape})"
            )

        # 3. 检查数据维度是否匹配
        if ct_array.shape != seg_array.shape:
            raise ValueError(
                f"Shape mismatch at index {index}. "
                f"CT: {ct_array.shape}, seg: {seg_array.shape}"
            )

        # 4. 检查数据范围
        if ct_array.min() < self.normalize_min or ct_array.max() > self.normalize_max:
            print(f"Warning: CT values out of range [{ct_array.min()}, {ct_array.max()}]")
            print(f"Current CT value range: [{self.ct_min}, {self.ct_max}]")
            print(f"Normalization range: [{self.normalize_min}, {self.normalize_max}]")

        if seg_array.min() < 0 or seg_array.max() > 1:
            print(f"Warning: Segmentation values out of range [{seg_array.min()}, {seg_array.max()}]")

    def __getitem__(self, index):
        ct_path, seg_path = self.filename_list[index]

        try:
            # 直接读取图像
            ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)

            # 打印原始数据统计信息
            print(
                f"Original CT array stats - min: {ct_array.min()}, max: {ct_array.max()}, mean: {ct_array.mean():.3f}")
            print(
                f"Original seg array stats - min: {seg_array.min()}, max: {seg_array.max()}, mean: {seg_array.mean():.3f}")

            # 使用 preprocess_ct 方法进行预处理
            ct_array = self.preprocess_ct(ct_array)

            # 验证数据维度
            self.validate_dimensions(ct_array, seg_array, index)

            # 统一数据大小到 32x24x128x128
            target_shape = (32, 24, 128, 128)

            # 处理深度维度 (D)
            if ct_array.shape[0] > target_shape[0]:
                # 如果深度大于目标大小，从中心裁剪
                start = (ct_array.shape[0] - target_shape[0]) // 2
                ct_array = ct_array[start:start + target_shape[0]]
                seg_array = seg_array[start:start + target_shape[0]]
            elif ct_array.shape[0] < target_shape[0]:
                # 如果深度小于目标大小，进行填充
                pad_size = target_shape[0] - ct_array.shape[0]
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                ct_array = np.pad(ct_array, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                seg_array = np.pad(seg_array, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')

            # 处理高度和宽度维度 (H, W)
            if ct_array.shape[1] != target_shape[2] or ct_array.shape[2] != target_shape[3]:
                ct_array = np.array(
                    [np.array(Image.fromarray(slice).resize((target_shape[3], target_shape[2]))) for slice in ct_array])
                seg_array = np.array(
                    [np.array(Image.fromarray(slice).resize((target_shape[3], target_shape[2]), Image.NEAREST)) for
                     slice in seg_array])

            # 打印处理后的数据统计信息
            print(
                f"Processed CT array stats - min: {ct_array.min():.3f}, max: {ct_array.max():.3f}, mean: {ct_array.mean():.3f}")
            print(
                f"Processed seg array stats - min: {seg_array.min()}, max: {seg_array.max()}, mean: {seg_array.mean():.3f}")

            # 转为Tensor并确保维度
            ct_tensor = torch.FloatTensor(ct_array)  # [D, H, W]
            seg_tensor = torch.FloatTensor(seg_array)  # [D, H, W]
            seg_tensor = seg_tensor.long()  # 确保标签是长整型

            # 添加channel维度
            ct_tensor = ct_tensor.unsqueeze(0)  # [1, D, H, W]

            return ct_tensor, seg_tensor

        except Exception as e:
            if self.args.strict_mode:
                raise RuntimeError(f"Error loading {ct_path}: {str(e)}")
            else:
                return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """生成一个有效的虚拟样本，用于数据加载失败时返回"""
        # 创建一个与目标大小相同的虚拟样本
        dummy_ct = torch.zeros((1, 32, 24, 128, 128))  # [1, D, H, W]
        dummy_seg = torch.zeros((32, 24, 128, 128), dtype=torch.long)  # [D, H, W]
        return dummy_ct, dummy_seg

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list


if __name__ == "__main__":
    sys.path.append('/ssd/lzq/3DUNet')
    from config import args

    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i, ct.size(), seg.size())