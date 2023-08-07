"""
This file contains additional functions for the unsupervised keypoints estimation method

engr.mz@hotmail.com
May 31, 2022
"""


import os
import torch
import torch.nn.functional as F
import open3d as o3d
import seaborn as sns


def compute_loss(kp1, kp2, data, writer, step, cfg, split='split??'):
    device = kp1.device
    l_sep1 = cfg.parameters.separation * separation_loss(kp1)
    l_sep2 = cfg.parameters.separation * separation_loss(kp2)
    l_overlap1 = cfg.parameters.overlap * overlap_loss(kp1, cfg.overlap_threshold)
    l_overlap2 = cfg.parameters.overlap * overlap_loss(kp2,  cfg.overlap_threshold)
    l_shape1 = cfg.parameters.shape * shape_loss(kp1, data[0].float().to(device))
    l_shape2 = cfg.parameters.shape * shape_loss(kp2, data[2].float().to(device))
    l_consist = cfg.parameters.consist * consistancy_loss(kp1, kp2, data[1].float().to(device), data[3].float().to(device))
    l_volume1 = cfg.parameters.volume * volume_loss(kp1, data[0].float().to(device))
    l_volume2 = cfg.parameters.volume * volume_loss(kp2, data[2].float().to(device))
    l_pose = cfg.parameters.pose *pose_loss(kp1, kp2, data[1].float().to(device), data[3].float().to(device))

    writer.add_scalar('{}_loss/consist'.format(split), l_consist, step)
    writer.add_scalar('{}_loss/relative_pose'.format(split), l_pose, step)
    writer.add_scalar('{}_loss/sep1'.format(split), l_sep1, step)
    writer.add_scalar('{}_loss/sep2'.format(split), l_sep2, step)
    writer.add_scalar('{}_loss/overlap1'.format(split), l_overlap1, step)
    writer.add_scalar('{}_loss/overlap2'.format(split), l_overlap2, step)
    writer.add_scalar('{}_loss/shape1'.format(split), l_shape1, step)
    writer.add_scalar('{}_loss/shape2'.format(split), l_shape2, step)
    writer.add_scalar('{}_loss/volume1'.format(split), l_volume1, step)
    writer.add_scalar('{}_loss/volume2'.format(split), l_volume2, step)

    return l_sep1 + l_sep2 + l_overlap1 + l_overlap2 + l_shape1 + l_shape2 + l_consist + l_volume1 + l_volume2 + l_pose #+ l_reconstruction


def consistancy_loss(kp1, kp2, rot1, rot2):
    '''

    Parameters
    ----------
    kp1     Estimated key-points 1
    kpT     Transformed version of the estimated key-points 2

    Returns     Loss => the corresponding key-points should be estimated in the same 3D positions
    -------

    '''
    kp2_to_kp1 = torch.transpose(torch.bmm(torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2)), torch.transpose(kp2.double(), 1, 2)), 1, 2)
    return F.mse_loss(kp1, kp2_to_kp1.float())


def chamfer_distance(pc, recons_pc):
    '''
    Parameters
    ----------
    pc              Input point cloud
    recons_pc       Reconstructed point cloud

    Returns Shape loss -> how far the reconstructed points (PC) are estimated from the input point cloud
    -------

    '''
    # pdb.set_trace()

    pred_to_gt = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - recons_pc[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(recons_pc))], dim=0)
    gt_to_pred = torch.cat([torch.squeeze(
        torch.norm(recons_pc[i].unsqueeze(1) - pc[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(pc))], dim=0)

    return torch.mean(pred_to_gt) + torch.mean(gt_to_pred)


def shape_loss(pc, kp):
    '''
    Parameters
    ----------
    pc      Input point cloud
    kp      Estimated key-points

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''
    loss = torch.cat([torch.squeeze(
        torch.norm(pc[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)
    return torch.mean(loss)


def overlap_loss_torch_error(kp, threshold=0.05):
    '''
    Parameters
    ----------
    kp:         Key-points
    threshold   allowable overlap between the key-points
    Method:     Find distance of every point from all the points
                select the minimum distances that are greater than 0 (distance from itself)
                return count of the separated distances => final loss

    Returns     separation loss -> avoid estimation of multiple key-points on the same 3D location
    -------
    '''

    distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None)) for i in range(len(kp))], dim=0)

    return torch.count_nonzero(distances[(distances < threshold)] >0) / len(kp)*len(kp)


def overlap_loss(kp, threshold=0.05):
    '''
    Parameters
    ----------
    kp:         Key-points
    threshold   allowable overlap between the key-points
    Method:     Find distance of every point from all the points
                select the minimum distances that are greater than 0 (distance from itself)
                return count of the separated distances => final loss

    Returns     separation loss -> avoid estimation of multiple key-points on the same 3D location
    -------
    '''

    distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None)) for i in range(len(kp))], dim=0)

    return torch.sum(distances[(distances < threshold)] >0).float() / len(distances)*len(distances)


def separation_loss(kp):
    '''
    Parameters
    ----------
    kp:         Key-points
    Method:     compute distances of each point from all the points in "kp"
                consider minimum two distances (distance of a point form itself (distance==0) and the next closest (distance>0))
                take mean of the distances from the closest point (distance>0)

    Returns     separation loss ->  average distance of every point from closest points
    -------
    '''
    min_distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None).topk(2, largest=False, dim=0)[
            0]) for i in range(len(kp))], dim=0)

    return 1/torch.mean(min_distances[min_distances>0])


def volume_loss(kp, pc):
    '''

    Parameters: 3D IoU loss
                => same as coverage loss of clara's Paper
                => https://github.com/cfernandezlab/Category-Specific-Keypoints/blob/master/models/losses.py
    Smooth L1 loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    '''
    val_max_pc, _ = torch.max(pc, 1)    # Bx3
    val_min_pc, _ = torch.min(pc, 1)    # Bx3
    dim_pc = val_max_pc - val_min_pc    # Bx3
    val_max_kp, _ = torch.max(kp, 1)    # Bx3
    val_min_kp, _ = torch.min(kp, 1)    # Bx3
    dim_kp = val_max_kp - val_min_kp    # Bx3

    return F.smooth_l1_loss(dim_kp, dim_pc)

    # https: // pytorch3d.readthedocs.io / en / latest / modules / ops.html
    # pytorch3d.ops.box3d_overlap(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 0.0001) → Tuple[
    #     torch.Tensor, torch.Tensor]

    """
    https://github.com/cfernandezlab/Category-Specific-Keypoints/blob/master/models/losses.py
    class CoverageLoss(nn.Module):
    def __init__(self, opt):
        super(CoverageLoss, self).__init__()
        self.opt = opt
        self.cov_criteria = nn.SmoothL1Loss() # reduction='none'

    def forward(self, kp, pc):
        # singular values - not efficient
        '''U, Spc, V = torch.svd(pc) 
        U, Skp, V = torch.svd(kp) 
        Spc = torch.div(Spc,torch.sum(Spc[:,:3], dim= 1).unsqueeze(1))
        Skp = torch.div(Skp,torch.sum(Skp[:,:3], dim= 1).unsqueeze(1))
        cov_loss = self.cov_criteria(Skp, Spc)''' 

        # volume
        val_max_pc, _ = torch.max(pc,2)
        val_min_pc, _ = torch.min(pc,2)
        dim_pc = val_max_pc - val_min_pc
        val_max_kp, _ = torch.max(kp,2)
        val_min_kp, _ = torch.min(kp,2)
        dim_kp = val_max_kp - val_min_kp
        cov_loss = self.cov_criteria(dim_kp, dim_pc)

        return cov_loss
        
        
        """


def pose_loss(kp1, kp2, rot1, rot2):
    '''

    Parameters
    ----------
    kp1     Estimated key-points 1
    kp2     Transformed version of the estimated key-points 2
    rot1    pose of KP1
    rot2    pose of KP2

    rot     GT relative pose b/w kp1 and kp2

    Returns     Loss => Error in relative pose b/w kp1 and kp2 [Forbunius Norm]
    -------

    '''
    device = kp1.device
    gt_rot = torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2))
    mat = batch_compute_similarity_transform_torch(kp1, kp2)
    # mat = batch_compute_similarity_transform_torch(kp1.permute(0, 2, 1), kp2.permute(0, 2, 1))
    frob = torch.sqrt(torch.sum(torch.square(gt_rot - mat)))    # Forbunius Norm

    angle_ = torch.mean(torch.arcsin(
        torch.clamp(torch.min(torch.tensor(1.).to(device), frob / (2. * torch.sqrt(torch.tensor(2.).to(device)))), -0.99999,
                    0.99999)))
    # angle_ = 2.0 * torch.mean(torch.arcsin(torch.clamp(torch.min(torch.tensor(1.).cuda(), frob / (2. * torch.sqrt(torch.tensor(2.).cuda()))), -0.99999, 0.99999 )))
    # angle_ = torch.rad2deg(2.0 * torch.mean(torch.arcsin(
    #     torch.clamp(torch.min(torch.tensor(1.).cuda(), frob / (2. * torch.sqrt(torch.tensor(2.).cuda()))), -0.99999,
    #                 0.99999))))

    return angle_



# o3d.visualization.draw_geometries([pcd])


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    help: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427

    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))      # position
    R = torch.linalg.inv(R)                 # rotation

    #
    # # 5. Recover scale.
    # scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
    #
    # # 6. Recover translation.
    # t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
    #
    # # 7. Error:
    # S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    #
    # if transposed:
    #     S1_hat = S1_hat.permute(0,2,1)
    #
    # return S1_hat

    return R


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
