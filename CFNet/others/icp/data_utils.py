import os, sys, glob, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/ljm/data'

# (9840, 2048, 3), (9840, 1)
# download in：https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
def load_data(partition, file_type='modelnet40'):
    # 读取训练集or测试集
    DATA_DIR = '/home/ljm/data'
    if file_type == 'modelnet40':
        file_name = 'modelnet40_ply_hdf5_2048'
    else:
        print('Error file name!')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label  # (9840, 2048, 3), (9840, 1)


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


def add_outliers(pointcloud, gt_mask):
    # pointcloud: 			Point Cloud (ndarray) [NxC]
    # output: 				Corrupted Point Cloud (ndarray) [(N+300)xC]
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.from_numpy(pointcloud)

    num_outliers = 20
    N, C = pointcloud.shape
    outliers = 2*torch.rand(num_outliers, C)-1 					# Sample points in a cube [-0.5, 0.5]
    pointcloud = torch.cat([pointcloud, outliers], dim=0)
    gt_mask = torch.cat([gt_mask, torch.zeros(num_outliers)])

    idx = torch.randperm(pointcloud.shape[0])
    pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]
    return pointcloud.numpy(), gt_mask


# 加入高斯噪声
def jitter_pointcloud(pointcloud, sigma=0.05, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    # pointcloud += sigma * np.random.randn(N, C)
    return pointcloud

def farthest_subsample_points(pointcloud1, num_subsampled_points):
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)

    return pointcloud1[idx1, :], gt_mask


class ModelNet40_Cla(Dataset):
    def __init__(self, num_points, partition='train',file_type='modelnet40'):
        self.data, self.label = load_data(partition, file_type=file_type)
        self.num_points = num_points
        self.partition = partition
        self.label = self.label.squeeze()  # 去掉维度为1的条目

    def __getitem__(self, item):
        # (batch, 3, num_points)
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ThreeDMatch(Dataset):
    def __init__(self,split):
        self.root = os.path.join(DATA_DIR, 'threedmatch')
        self.split = split
        DATA_FILES = {
            'train': os.path.join(BASE_DIR,'split/train_3dmatch.txt'),
            'val':os.path.join(BASE_DIR,'split/val_3dmatch.txt'),
        }
        subset_names = open(DATA_FILES[split]).read().split()
        print(subset_names)
        self.files = []
        self.length = 0
        for name in subset_names:
            fname = name + "*.npz"
            fnames_txt = glob.glob(os.path.join(self.root,fname))
            for fname_txt in fnames_txt:
                self.files.append(fname_txt)
                self.length += 1
        # print(self.files)

    def __getitem__(self, index):
        template_file= self.files[index]
        template_data = np.load(os.path.join(self.root,template_file))
        template_keypts = template_data['pcd']
        template_rgb = template_data['color']
        template =np.concatenate((template_keypts,template_rgb), axis=1)

        return template

    def __len__(self):
        return self.length

class ModelNet40_Reg(Dataset):
    def __init__(self, num_points=1024, num_subsampled_rate=1, partition='train', max_angle=45, max_t=1.0,
                 noise=False, partial_source=False, unseen=False, single=-1,file_type='modelnet40'):
        self.partial_source = partial_source  # 是否部分重叠(第二个点云部分缺失)
        self.data, self.label = load_data(partition, file_type=file_type)
        self.num_points = num_points
        self.file_type = file_type
        self.partition = partition
        self.label = self.label.squeeze()  # 去掉维度为1的条目
        self.max_angle = np.pi / 180 * max_angle
        self.max_t = max_t
        self.noise = noise
        self.unseen =unseen
        self.single = single
        self.num_subsampled_rate = num_subsampled_rate

        if file_type == 'modelnet40' and self.unseen:
            # simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

        if self.single >= 0:
            ######## simulate training and testing on a single categories
            self.data = self.data[self.label == self.single]
            self.label = self.label[self.label == self.single]

    def __getitem__(self, item):
        if self.file_type == 'modelnet40':
            pointcloud = self.data[item][:self.num_points]
        else:
            pointcloud = self.data[item]
        if self.noise:
            pointcloud = jitter_pointcloud(pointcloud)

        anglex = np.random.uniform(-self.max_angle, self.max_angle)
        angley = np.random.uniform(-self.max_angle, self.max_angle)
        anglez = np.random.uniform(-self.max_angle, self.max_angle)
        # anglex = 10
        # angley = -20
        # anglez = 10

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        # 平移矩阵T
        translation_ab = np.array(
            [np.random.uniform(-self.max_t, self.max_t), np.random.uniform(-self.max_t, self.max_t),
             np.random.uniform(-self.max_t, self.max_t)])
        # translation_ab = np.array([0.2,-0.1,0.1])

        translation_ba = -R_ba.dot(translation_ab)
        # 第item个物体 点云1 [3xN]
        pointcloud1 = pointcloud.T

        euler_ba = -euler_ab[::-1]
        # 打乱点的顺序(3, num_points)
        pointcloud1 = np.random.permutation(pointcloud1.T).T

        # 将点云1按角度旋转
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        gt_mask = torch.tensor([0, 0, 0])

        # return (batch, 3, num_points)
        # 两个点云(源点云+目标点云)，旋转矩阵R_ab，T_ab, 欧拉角，点云1旋转平移得到点云2
        return pointcloud2.astype('float32'), pointcloud1.astype('float32'), R_ba.astype('float32'), \
               translation_ba.astype('float32'), euler_ab.astype('float32'), gt_mask

    def __len__(self):
        return self.data.shape[0]

class Registration3dmatchData(Dataset):
    def __init__(self, data_class,sample_point_num=2048,max_angle=45, max_t=1.0,
                 noise=False, partial_source=False, unseen=False, single=-1,is_testing=False):
        super(Registration3dmatchData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.max_angle = np.pi / 180 * max_angle
        self.max_t = max_t
        self.noise = noise
        self.unseen = unseen
        self.single = single
        self.sample_point_num = sample_point_num

    def __getitem__(self, index):
        pointcloud= self.data_class[index] # [N,6]
        pointcloud = pointcloud[:, :3] # [N,3]

        N_tmp =pointcloud.shape[0]
        sel_ind = np.random.choice(N_tmp, self.sample_point_num)
        pointcloud =pointcloud[sel_ind,:]

        anglex = np.random.uniform(-self.max_angle, self.max_angle)
        angley = np.random.uniform(-self.max_angle, self.max_angle)
        anglez = np.random.uniform(-self.max_angle, self.max_angle)
        # anglex = 10
        # angley = -20
        # anglez = 10

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        # 平移矩阵T
        translation_ab = np.array(
            [np.random.uniform(-self.max_t, self.max_t), np.random.uniform(-self.max_t, self.max_t),
             np.random.uniform(-self.max_t, self.max_t)])
        # translation_ab = np.array([0.2,-0.1,0.1])

        translation_ba = -R_ba.dot(translation_ab)
        # 第item个物体 点云1 [3xN]
        pointcloud1 = pointcloud.T

        euler_ba = -euler_ab[::-1]
        # 打乱点的顺序(3, num_points)
        pointcloud1 = np.random.permutation(pointcloud1.T).T

        # 将点云1按角度旋转
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        gt_mask = torch.tensor([0, 0, 0])

        # return (batch, 3, num_points)
        # 两个点云(源点云+目标点云)，旋转矩阵R_ab，T_ab, 欧拉角，点云1旋转平移得到点云2
        return pointcloud2.astype('float32'), pointcloud1.astype('float32'), R_ba.astype('float32'), \
               translation_ba.astype('float32'), euler_ab.astype('float32'), gt_mask

    def __len__(self):
        return len(self.data_class)

class Kitti(Dataset):
    def __init__(self, split, type='object'):
        self.type = type
        self.root = os.path.join(DATA_DIR, f'kitti_{self.type}')
        # print(self.root) /home/ljm/data/kitti_object
        self.split = split
        DATA_FILES = {
            'train': os.path.join(self.root, 'training/velodyne'),
            'test': os.path.join(self.root, 'testing/velodyne'),
        }
        self.files = []
        self.length = 0
        if type == 'object':
            fnames_txt = glob.glob(os.path.join(DATA_FILES[split],"*.bin"))
        else:
            fnames_txt = glob.glob(os.path.join(DATA_FILES[split], "*","*.bin"))
        for i, fname_txt in enumerate(fnames_txt):
            self.files.append(fname_txt)
            self.length += 1

    def __getitem__(self, index):
        template_file = self.files[index]
        template_data = np.fromfile(os.path.join(self.root, template_file), dtype=np.float32).reshape(-1, 4)
        template_pcd = template_data[:, :3]
        return torch.from_numpy(template_pcd).type(torch.float)

    def __len__(self):
        return self.length


class RegistrationkittiData(Dataset):
    def __init__(self, data_class, sample_point_num=2048, max_angle=45, max_t=1.0,
                 noise=False, partial_source=False, unseen=False, single=-1, is_testing=False):
        super(RegistrationkittiData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.max_angle = np.pi / 180 * max_angle
        self.max_t = max_t
        self.noise = noise
        self.unseen = unseen
        self.single = single
        self.sample_point_num = sample_point_num

    def __getitem__(self, index):
        pointcloud = self.data_class[index]

        pointcloud = pointcloud - torch.mean(pointcloud, dim=1, keepdim=True).to(pointcloud)

        N_tmp =pointcloud.shape[0]
        sel_ind = np.random.choice(N_tmp, self.sample_point_num)
        pointcloud =pointcloud[sel_ind,:]

        anglex = np.random.uniform(-self.max_angle, self.max_angle)
        angley = np.random.uniform(-self.max_angle, self.max_angle)
        anglez = np.random.uniform(-self.max_angle, self.max_angle)
        # anglex = 10
        # angley = -20
        # anglez = 10

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        # 平移矩阵T
        translation_ab = np.array(
            [np.random.uniform(-self.max_t, self.max_t), np.random.uniform(-self.max_t, self.max_t),
             np.random.uniform(-self.max_t, self.max_t)])
        # translation_ab = np.array([0.2,-0.1,0.1])

        translation_ba = -R_ba.dot(translation_ab)
        # 第item个物体 点云1 [3xN]
        pointcloud1 = pointcloud.T

        euler_ba = -euler_ab[::-1]
        # 打乱点的顺序(3, num_points)
        pointcloud1 = np.random.permutation(pointcloud1.T).T

        # 将点云1按角度旋转
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        gt_mask = torch.tensor([0, 0, 0])

        # return (batch, 3, num_points)
        # 两个点云(源点云+目标点云)，旋转矩阵R_ab，T_ab, 欧拉角，点云1旋转平移得到点云2
        return pointcloud2.astype('float32'), pointcloud1.astype('float32'), R_ba.astype('float32'), \
               translation_ba.astype('float32'), euler_ab.astype('float32'), gt_mask

    def __len__(self):
        return len(self.data_class)

