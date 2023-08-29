import argparse

import torch.multiprocessing
from visualization import CloudVisualizer

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter
import transforms3d
from tqdm import tqdm
import sys, os, random

from data_utils import ModelNet40_Reg, ModelNet40_Cla, Registration3dmatchData,ThreeDMatch,RegistrationkittiData, Kitti
from FGR_model import FGR
from scipy.spatial.transform import Rotation
from evaluate_funcs import calculate_R_msemae, calculate_t_msemae, compute_batch_error

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_one_epoch(net, test_loader):
    total_loss = 0
    num_examples = 0
    rotations = []
    translations = []
    rotations_pred = []
    translations_pred = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            src, target, rotation, translation, euler, gt_mask = data
            batchsize = src.shape[0]

            # translation = translation - torch.mean(translation, dim=1, keepdim=True).to(translation)
            # src = src - torch.mean(src, dim=2, keepdim=True).to(src)
            # target = target - torch.mean(target, dim=2, keepdim=True).to(target)

            rotation_pred, translation_pred = net(src, target)
            rotation_pred = rotation_pred.cpu()
            translation_pred = translation_pred.cpu()

            # 变换点云
            transformed_src = torch.bmm(rotation_pred, src) + translation_pred.reshape(batchsize, 3, 1)
            # print('src.shape=',src.shape,'\ntarget.shape=',target.shape)
            # vis = CloudVisualizer(0.01, os.path.join('../result','fgr', "{}_fgr_kitti".format(str(i))))
            # vis.reset(src.permute(0,2,1)[1, :, :3].cpu().numpy(),target.permute(0,2,1)[1, :, :3].cpu().numpy(),
            #           transformed_src.permute(0,2,1)[1, :, :3].detach().cpu().numpy())

            num_examples += batchsize

            # 保存rotation and translation, detach()后不再计算梯度
            rotations.append(rotation.detach().cpu().numpy())
            translations.append(translation.detach().cpu().numpy())
            rotations_pred.append(rotation_pred.detach().cpu().numpy())
            translations_pred.append(translation_pred.detach().cpu().numpy())

            # loss = chamfer_distance(transformed_src.permute(0, 2, 1), target.permute(0, 2, 1))[0]
            # total_loss += loss.item() * batchsize

        rotations = np.concatenate(rotations, axis=0)
        translations = np.concatenate(translations, axis=0)
        rotations_pred = np.concatenate(rotations_pred, axis=0)
        translations_pred = np.concatenate(translations_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations, \
           translations, rotations_pred, translations_pred


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(1234)
    batchsize = 32
    numworkers = 4
    # num_subsampled_points = 768
    max_angle = 45
    max_t = 1.0
    partial_source = False
    unseen = False
    noise = False
    single = -1
    R_rmse, R_mae, t_rmse, t_mae = [], [], [], []
    test_loader = DataLoader(
        dataset=ModelNet40_Reg(2048, partition='test', max_angle=max_angle, max_t=1.0, noise=noise,unseen = unseen,
                              single = single, partial_source=partial_source, percent = 1.0),
        batch_size=batchsize,
        shuffle=False,
        num_workers=4
    )

    # testset = Registration3dmatchData(ThreeDMatch(split='val'), sample_point_num=2048, max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=numworkers)
    #
    # testset= RegistrationkittiData(Kitti(split='test',type='tracking'), sample_point_num=2048 ,max_angle=max_angle,
    #                                   max_t=max_t, unseen=unseen, noise=noise,single=single,is_testing=True)
    # test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=4)

    net = FGR()

    test_loss, test_R, test_t, test_R_pred, test_t_pred= test_one_epoch(net, test_loader)

    # 计算各种评估误差
    # [t_error, angle_error, angle_mat_error]
    test_errors = compute_batch_error(test_R, test_R_pred, test_t, test_t_pred)
    test_R_mse, test_R_mae = calculate_R_msemae(test_R, test_R_pred)
    test_R_rmse = np.sqrt(test_R_mse)
    test_t_mse, test_t_mae = calculate_t_msemae(test_t, test_t_pred)
    test_t_rmse = np.sqrt(test_t_mse)

    R_rmse.append(test_R_rmse)
    R_mae.append(test_R_mae)
    t_rmse.append(test_t_rmse)
    t_mae.append(test_t_mae)
    print('Test: Loss: %f, MSE(R): %f, RMSE(R): %f, MAE(R): %f, '
                  'MSE(t): %f, RMSE(t): %f, MAE(t): %f, error(t): %f, '
                'error(R): %f, error(matR): %f'
                  % (test_loss, test_R_mse, test_R_rmse, test_R_mae,test_t_mse,
                    test_t_rmse, test_t_mae, test_errors[0], test_errors[1], test_errors[2]))
