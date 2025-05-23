from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
from utils_module import common, logger, loss, metrics, weights_init
import os
from collections import OrderedDict
from model.Med_Seg_ViG import Med_Seg_ViG
import numpy as np


def val(model, val_loader, loss_func, n_labels, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # print(f"Validation data range: [{data.min():.3f}, {data.max():.3f}]")
        # print(f"Validation target range: [{target.min()}, {target.max()}]")

        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = loss_func(output, target)

        val_loss.update(loss.item(), data.size(0))
        val_dice.update(output, target)

    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels == 3:
        val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels, alpha, device, epoch):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        # 更新损失和 Dice 系数
        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output, target)

    # 记录日志
    val_log = OrderedDict({
        'Train_Loss': train_loss.avg,
        'Train_dice_liver': train_dice.avg[1]
    })
    if n_labels == 3:
        val_log.update({'Train_dice_tumor': train_dice.avg[2]})

    return val_log


if __name__ == '__main__':

    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')

    # dataloader
    train_loader = DataLoader(dataset=Train_Dataset(args), batch_size=1, num_workers=args.n_threads,
                              shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args), batch_size=1, num_workers=args.n_threads, shuffle=False)

    model = Med_Seg_ViG(in_channels=1, out_channels=args.n_labels, training=True).to(device)

    model.apply(weights_init.init_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = loss.TverskyLoss()

    # common.print_network(model)
    model = model.to(device=args.gpu_id[0])
    # model = torch.nn.DataParallel(model, device_ids=args.gpu_id)

    log = logger.Train_Logger(save_path, "train_log")

    best = [0, 0]
    trigger = 0
    alpha = 0.4

    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)

        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha, device, epoch)
        val_log = val(model, val_loader, loss, args.n_labels, device)
        log.update(epoch, train_log, val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break


def preprocess_ct(self, ct_array):
    self.ct_min = min(self.ct_min, ct_array.min())
    self.ct_max = max(self.ct_max, ct_array.max())

    ct_array = np.clip(ct_array, self.normalize_min, self.normalize_max)
    ct_array = (ct_array - self.normalize_min) / (self.normalize_max - self.normalize_min)

    if ct_array.min() < 0 or ct_array.max() > 1:
        print(f"Warning: Normalized data out of range [{ct_array.min():.3f}, {ct_array.max():.3f}]")
        print(f"Current CT value range: [{self.ct_min}, {self.ct_max}]")
        print(f"Normalization range: [{self.normalize_min}, {self.normalize_max}]")

    return ct_array.astype(np.float32)


def __init__(self, args):
    self.args = args
    self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
    self.ct_min = float('inf')
    self.ct_max = float('-inf')
    self.normalize_min = -300
    self.normalize_max = 300