import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np
import einops
from torch.autograd import Variable



def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out


class residual_block(nn.Module):
    """
    # Residual block:
    # Input:    in_ (input channels)        # 1024
                out_ (output channels)      # 512
    # Output:
                x = [B x 512 x 2048]
    """
    def __init__(self, in_, out_):
        super(residual_block, self).__init__()
        self.conv21 = torch.nn.Conv1d(in_, in_, 1)
        self.bn21 = nn.BatchNorm1d(in_)
        self.conv22 = torch.nn.Conv1d(in_, out_, 1)
        self.bn22 = nn.BatchNorm1d(out_)

    def forward(self, input):                               # [B x 1024 x 2048]
        x = F.relu(self.bn21(self.conv21(input)))           # [B x 1024 x 2048]
        x = self.bn22(self.conv22(x))                       # [B x 512 x 2048]
        input = self.bn22(self.conv22(input))               # [B x 512 x 2048]
        x += input                                          # [B x 512 x 2048]
        x = F.relu(x)                                       # [B x 512 x 2048]
        return x


'''
 STN3d, STNkd are same as the pointnet classes
 The last layer of and Pointnetfeat has modified to get [N, 512] fetures
'''
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    '''
    input:  [N, 3, 2048]
    output: [N, 512]
    global_feat = True, feature_transform = True
    '''
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=64)     # feature_transform = True
        #
        # self.conv21 = torch.nn.Conv1d(1024, 512, 1)
        # self.conv22 = torch.nn.Conv1d(512, 256, 1)
        # self.conv23 = torch.nn.Conv1d(256, 10, 1)
        # self.bn21 = nn.BatchNorm1d(512)
        # self.bn22 = nn.BatchNorm1d(256)
        #
        # self.softmax = nn.Softmax(dim=2)
        # # self.fc21 = nn.Linear(1024, 512)
        # # self.bn21 = nn.BatchNorm1d(512)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # feature_transform = True
        trans_feat = self.fstn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)

        # global_feat [B, 1024]
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))          # [B x 1024 x 2048]

        # # Down-sampling from 1024 to M key-points
        # x = F.relu(self.bn21(self.conv21(x)))        # [B x 512 x 2048]
        # x = F.relu(self.bn22(self.conv22(x)))        # [B x 256 x 2048]
        # x = self.conv23(x)                           # [B x 10 x 2048]
        #
        # x = self.softmax(x)                          # [B x 10 x 2048] => [B x 10 x 2048{0 to 1}]
        # x = torch.bmm(x, pc.permute(0,2,1))          # [Bx10x2048] <-> [Bx2048x3]  =>  [Bx10x3]

        return x


'''========================================================'''

class Unsupervised_kpnet_without_Residual_block(nn.Module):
    """
        Unsupervised Key-point net: 3D key-points estimation from point clouds using an unsupervised approach
        Inputs:
            point-cloud: [Bx2048x3]
        Computes:
            - Point cloud features nx1024 [1024 features for every point]
            - MLP computes nxM features representing class probability for every point [M = total number of key-points]
            - Use softmax between all the points for every M => select one point that based to the class with highest probability
              So, the total M key-points will be separated => that are the estimated key-points
        Output:
            - the key-points 3D positions [BxMx3] => based on the separated key-points indexes

        Sub-modules;
        1. pointnet features extracter for point clouds
        2. MLP ans Softmax
    """
    def __init__(self, cfg):
        super(Unsupervised_kpnet_without_Residual_block, self).__init__()
        self.pointnet_encoder = PointNetfeat()

        self.conv21 = torch.nn.Conv1d(1024, 512, 1)
        self.conv22 = torch.nn.Conv1d(512, 256, 1)
        self.conv23 = torch.nn.Conv1d(256, cfg.key_points, 1)
        self.bn21 = nn.BatchNorm1d(512)
        self.bn22 = nn.BatchNorm1d(256)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, pc):
        x = self.pointnet_encoder(pc.permute(0, 2, 1))   # [B x 1024 x 2048]

        # Down-sampling from 1024 to M key-points
        x = F.relu(self.bn21(self.conv21(x)))        # [B x 512 x 2048]
        x = F.relu(self.bn22(self.conv22(x)))        # [B x 256 x 2048]
        x = self.conv23(x)                           # [B x 10 x 2048]

        x = self.softmax(x)                          # [B x 10 x 2048] => [B x 10 x 2048{0 to 1}]
        # x = torch.bmm(x, pc)          # [Bx10x2048] <-> [Bx2048x3]  =>  [Bx10x3]

        return torch.bmm(x, pc)


class Unsupervised_kpnet(nn.Module):
    """
        Unsupervised Key-point net: 3D keypoints estimation from point clouds using an unsupervised approach
        Inputs:
            point-cloud: [Bx2048x3]
        Computes:
            - Point cloud features MxN [N (1024) features for every point (total N=2048)]
            - Residual Blocks down sample the MxN features
            - Conv1D computes N features for K key-points [KxN]
            - Soft-max normalize the features such that the sum of all the features (N) for single key-point (K1) become 1
              [KxN{0to1}]
            - Matrix Multiplication estimates the K key-points by averaging the points of the input PC based on the computed features
              So, the total [Kx3] key-points will be separated => that are the estimated key-points
        Output:
            - the key-points 3D positions [BxKx3]

        Sub-modules;
        1. Pointnet features extracter for point clouds
        2. Residual block, Conv1D ans Softmax
    """
    def __init__(self, cfg):
        super(Unsupervised_kpnet, self).__init__()
        self.pointnet_encoder = PointNetfeat()
        self.block1 = residual_block(1024, 512)
        self.block2 = residual_block(512, 256)
        self.conv23 = torch.nn.Conv1d(256, cfg.key_points, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, pc):
        x = self.pointnet_encoder(pc.permute(0, 2, 1))   # [B x 1024 x 2048]

        # Down-sampling from 1024 to M key-points
        x = self.block1(x)          # [B x 512 x 2048]
        x = self.block2(x)          # [B x 256 x 2048]
        x = self.conv23(x)          # [B x cfg.key_points x 2048]
        x = self.softmax(x)          # [B x cfg.key_points x 2048] => [B x cfg.key_points x 2048{0 to 1}]
        x = torch.bmm(x, pc)        # [Bx cfg.key_points x2048] <-> [Bx2048x3]  =>  [Bx cfg.key_points x3]

        return x


class sc3k(nn.Module):
    '''
    Input:  pcd1: data[0], pcd2: data[2]
    Output: kp1:keypoints1, kp2: keypoints2
    '''
    def __init__(self, cfg):
        super(sc3k, self).__init__()
        self.estimate_kp = Unsupervised_kpnet(cfg)
        # self.estimate_kp = Unsupervised_kpnet_without_Residual_block(cfg)
        self.task = cfg.task
        self.split = cfg.split

    def forward(self, data):
        device = next(self.parameters()).device 
        if self.split == 'train' or self.task == 'generic':
            kp1 = self.estimate_kp(data[0].float().to(device)) # cuda())
            kp2 = self.estimate_kp(data[2].float().to(device)) # cuda())
            return kp1, kp2

        elif self.task == 'canonical':
            kp1 = self.estimate_kp(data[0].float().to(device)) # cuda())
            return kp1


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    cfg.split = 'train'
    pc = torch.randn(5, 2048, 3)
    data = [pc, pc, pc, pc]
    model = sc3k(cfg).cuda()
    # pdb.set_trace()
    kp1, kp2 = model(data)
    print(kp1.shape, kp1.shape)


if __name__ == '__main__':
    main()


