import numpy as np
from glob import glob
import os
import json
from torchvision import transforms
import pdb
import hydra
import torch
import omegaconf
from tqdm import tqdm

import open3d as o3d
import itertools    # join lists of list in one_list
import matplotlib.pyplot as plt

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel", }

NAMES2ID = {v: k for k, v in ID2NAMES.items()}


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)
    return x + noise


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def transform(pc, extrinsic_mat):
    zup = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype='f')  # Z_UP
    return np.dot(extrinsic_mat @ zup, pc.T).T


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class generic_data_loader(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.catg = cfg.class_name
        self.cfg = cfg
        self.cat = []
        self.cat.append(NAMES2ID[cfg.class_name])

        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] in self.cat]

        selected_cat = []
        for i in range(len(annots)):
            if annots[i]['class_id'] not in selected_cat:
                selected_cat.append(annots[i]['class_id'])
        print('loaded {} samples of categories: '.format(len(annots)), selected_cat)

        pcd_paths_np = []
        for i in range(len(selected_cat)):
            pcd_paths_np += glob(os.path.join(BASEDIR, cfg.data.pcd_root, selected_cat[i], '*.pcd'))

        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        split_models = open(os.path.join(BASEDIR, cfg.data.splits_root, "{}.txt".format(split))).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]

        mesh_names = []
        camera_param_np = []
        camera_param_np_2 = []
        pointCloud_lst = []
        pointCloud_lst_2 = []
        print("Loading {} data, please wait\n".format(split))
        for fn in tqdm(pcd_paths_np):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue

            cat_name = fn.split('/')[-2]
            mesh_names.append(model_id)

            pc_list = []
            cam_lst = []
            camera_mat = np.load(os.path.join(BASEDIR, cfg.data.poses_root, cat_name, '{}.npz'.format(model_id)))
            for i in range(24):
                cam_lst.append(camera_mat['world_mat_{}'.format(i)][:,:3])
                pc_list.append(transform(naive_read_pcd(fn)[0], camera_mat['world_mat_{}'.format(i)][:,:3]))
            camera_param_np.append(cam_lst)
            camera_param_np_2.append(cam_lst[::-1])
            pointCloud_lst.append(pc_list)
            pointCloud_lst_2.append(pc_list[::-1])

        print("\n\nPlease wait, arranging the data\n\n")
        self.camera_param_np = list(itertools.chain.from_iterable(camera_param_np))     # combine array elements in
        self.camera_param_np_2 = list(itertools.chain.from_iterable(camera_param_np_2))  # combine array elements in
        self.transformed_pcds = list(itertools.chain.from_iterable(pointCloud_lst))  # combine array elements in
        self.transformed_pcds_2 = list(itertools.chain.from_iterable(pointCloud_lst_2))  # combine array elements in
        self.mesh_names = list(np.repeat(mesh_names, 24))     # repeat list

        print("\n\nloaded data contains: ")
        print("  * camera_param 1: {}".format(len(self.camera_param_np)))
        print("  * camera_param 2: {}".format(len(self.camera_param_np_2)))
        print("  * transformed_pcds 1: {}".format(len(self.transformed_pcds)))
        print("  * transformed_pcds 2: {}".format(len(self.transformed_pcds_2)))
        print("  * mesh_names: {}\n\n".format(len(self.mesh_names)))

    def __getitem__(self, idx):
        pcd1 = self.transformed_pcds[idx]
        pcd2 = self.transformed_pcds_2[idx]
        camera_matrix = self.camera_param_np[idx]
        camera_matrix2 = self.camera_param_np_2[idx]
        mesh_name = self.mesh_names[idx]

        if self.cfg.augmentation.normalize_pc:
            pcd1 = normalize_pc(pcd1)
            pcd2 = normalize_pc(pcd2)

        if self.cfg.augmentation.down_sample:
            pcd1 = farthest_point_sample(pcd1, self.cfg.sample_points)

        if self.cfg.augmentation.gaussian_noise:
            pcd1 = add_noise(pcd1, sigma=self.cfg.lamda)
            pcd2 = add_noise(pcd2, sigma=self.cfg.lamda2)


        return pcd1.astype(np.float32), camera_matrix, pcd2.astype(np.float32), camera_matrix2, mesh_name,

    def __len__(self):
        return len(self.mesh_names)


class canonical_data_loader(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.catg = cfg.class_name
        self.cfg = cfg
        self.cat = []
        self.cat.append(NAMES2ID[cfg.class_name])

        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] in self.cat]

        selected_cat = []
        for i in range(len(annots)):
            if annots[i]['class_id'] not in selected_cat:
                selected_cat.append(annots[i]['class_id'])
        print('loaded {} samples of categories: '.format(len(annots)), selected_cat)

        pcd_paths_np = []
        for i in range(len(selected_cat)):
            pcd_paths_np += glob(os.path.join(BASEDIR, cfg.data.pcd_root, selected_cat[i], '*.pcd'))

        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        split_models = open(os.path.join(BASEDIR, cfg.data.splits_root, "{}.txt".format(split))).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]

        mesh_names = []
        pointCloud_lst = []
        print("Loading {} data, please wait\n".format(split))
        for fn in tqdm(pcd_paths_np):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue

            mesh_names.append(model_id)
            pointCloud_lst.append(naive_read_pcd(fn)[0])

        self.transformed_pcds = pointCloud_lst
        self.mesh_names = mesh_names
        print("\nmesh_names: {}".format(len(self.mesh_names)))
        print("\point clouds: {}".format(len(self.transformed_pcds)))


    def __getitem__(self, idx):
        pcd1 = self.transformed_pcds[idx]
        mesh_name = self.mesh_names[idx]

        if self.cfg.augmentation.normalize_pc:
            pcd1 = normalize_pc(pcd1)

        if self.cfg.augmentation.uniform_sampling:
            pcd1_updated = farthest_point_sample(pcd1, self.cfg.sample_points)

        else:
            pcd1_updated = pcd1

        if self.cfg.augmentation.gaussian_noise:
            pcd1_updated = add_noise(pcd1_updated, sigma=self.cfg.lamda)

        return pcd1_updated.astype(np.float32),  mesh_name


    def __len__(self):
        return len(self.transformed_pcds)





def debug(data):
    '''

    Parameters
    ----------
    data :: loaded batch of [pc1, pose1, pc2, pose2, name]

    Returns :: visualize if the relative pose is correct or not
               1. Inverse transform of pc1 and pc2 should be in a same initial pose
               2. Transform(pose2,   Transform(Inv(pose1),pc1)) => pc1 should transform to the pose 2
    -------

    '''
    aa = data[0][0]
    bb = data[2][0]
    # taa = data[1][1][0][:, :3]
    # tbb = data[3][1][0][:, :3]
    taa = data[1][0]
    tbb = data[3][0]

    '''Transform both the PCs to original pose'''
    aa2 = torch.matmul(taa.double().T, aa.double().T).T
    bb2 = torch.matmul(tbb.double().T,bb.double().T).T

    pdb.set_trace()
    show_points(aa, bb, True)
    show_points(aa2,bb2, True)

    '''same as transformation 1 : separate transformations'''
    aa2bb = torch.matmul(tbb.double(), aa2.double().T).T
    show_points(aa2bb,bb, True)
    bb2aa = torch.matmul(taa.double(), bb2.double().T).T
    show_points(bb2aa,aa, True)

    '''same as transformation 2 : in one line'''
    aa3bb = torch.matmul(tbb.double() @ taa.double().T , aa.double().T).T
    show_points(aa3bb, bb, True)
    bb3aa = torch.matmul(taa.double() @ tbb.double().T, bb.double().T).T
    show_points(bb3aa, aa, True)

    ''' Batch wise transformation '''
    AA2BB = torch.transpose(torch.bmm( torch.bmm(data[3].double(), torch.transpose(data[1].double(),1,2)) , torch.transpose(data[0].double(),1,2)), 1,2)
    show_points(AA2BB[5], data[2][5], True)
    BB2AA = torch.transpose(torch.bmm(torch.bmm(data[1].double(), torch.transpose(data[3].double(), 1, 2)),torch.transpose(data[2].double(), 1, 2)), 1, 2)
    show_points(BB2AA[5], data[0][5], True)

def show_points(points1, points2=0, both=False):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)

    if both == False:
        o3d.visualization.draw([pcd1])
    else:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        o3d.visualization.draw([pcd1, pcd2])


# main to test dataloader pipeline
def test_imgs_loader(cfg):

    train_dataset = generic_data_loader(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers, drop_last=False)

    train_iter = tqdm(train_dataloader)
    for i, data in enumerate(train_iter):
        print(len(data))
        debug(data)
        pdb.set_trace()
        show_points(data[0][0])
        show_points(data[0][0], data[0][2], True)
        debug(data)
        plt.show()
        # functions_bank.show_keypoints(data[0][0], data[0][0])



@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    test_imgs_loader(cfg)

if __name__ == '__main__':
    main()
