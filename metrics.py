import os
import open3d as o3d
import seaborn as sns
import torch
import numpy as np
import pdb
import copy
def pc_to_pcd(pc, color=7):
    palette_PC = sns.color_palette()
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)  ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    return pcd


def kp_to_pcd(kp):
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
    pcd.translate(kp[0])
    pcd.paint_uniform_color(palette[0])

    for i in range(1, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)  # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i == 7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point
    return pcd



def coverage(kp, pc):
    '''

    Parameters: 3D Coverage loss
                => same as coverage loss of clara's Paper
                => https://github.com/cfernandezlab/Category-Specific-Keypoints/blob/master/models/losses.py
    Smooth L1 loss: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    ----------
    kp: Estimated key-points [BxNx3]
    pc: Point cloud [Bx2048x3]

    Returns: Int value -> IoU b/w kp and pc
    -------

    '''
    device = kp.device
    val_max_pc, _ = torch.max(pc, 1)  # Bx3
    val_min_pc, _ = torch.min(pc, 1)  # Bx3
    dim_pc = val_max_pc - val_min_pc  # Bx3
    val_max_kp, _ = torch.max(kp, 1)  # Bx3
    val_min_kp, _ = torch.min(kp, 1)  # Bx3
    dim_kp = val_max_kp - val_min_kp  # Bx3

    ''' % coverage of kp over pc'''
    temp = torch.tensor([[0, 0, 0]], dtype=torch.float32).to(device)
    dis_kp = torch.cdist(temp, dim_kp).squeeze()  # distance of kp (BB) from origin
    dis_pc = torch.cdist(temp, dim_pc).squeeze()  # distance of PC (BB) from origin

    overlapping = 1 - torch.abs(dis_pc - dis_kp) / dis_pc
    overlapping[overlapping < 0] = 0
    return torch.mean(overlapping) * 100


def inclusivity(kp, pc, threshold=0.03):
    '''
    Parameters
    ----------
    pc      	Input point cloud
    kp      	Estimated key-points
    threshold  	threshold value

    Returns Shape loss -> how far the key-points are estimated from the input point cloud
    -------

    '''
    loss = torch.cat([torch.squeeze(
        torch.norm(pc.unsqueeze(1) - kp.unsqueeze(0), dim=2, p=None).topk(1, largest=False, dim=0)[0])], dim=0)

    return torch.tensor((len(loss[loss < threshold]) / len(loss)) * 100)
    # return loss  # percentage of points closest to the surface


def DAS_unsupervised(kpts):
    '''
        Compute the DAS values:
        input: kpts (list of keypints for all the samples of the text set) i.e. [samples, keypoints, channel] [4920, 10, 3]
        Ref: first keypoint from the sets of the keypoints  [1,2,3,4,5]
        Pred: select keypoints for each sample one by one (other than the reference)

        Procedure:
        For every point in Ref, compute its 1NN from the Pred.
        positive += 1 if the indexes of both the points (Ref[i], 1NN(Pred, Ref[i])) are same
        Compute the ratio: total_positive / total KP * 100

    Returns
    -------

        Ratio: the percentage of corresponding keypoints

    '''

    ref = np.expand_dims(kpts[0], 1)  # k1 x 1 x 3
    ref_index = [i for i in range(len(ref))]
    predictions = kpts[1:]

    positive = 0
    for pred in predictions:
        pred_ = np.expand_dims(pred, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(ref - pred_), -1))  # k1 x k2
        index_min_dist = np.argmin(dist, -1)  # Index of KNN point of pred [K1, 1]
        positive += np.count_nonzero(index_min_dist == ref_index)

    positive = (positive / (len(predictions) * len(ref))) * 100  # [positive / total keypoints (samples*No_of_kp)] * 100

    return positive


def pose_loss(kp1, kp2, rot1, rot2):
    '''

    Parameters
    ----------
    kp1     Estimated key-points 1  [1xkpx3]
    kp2     Transformed version of the estimated key-points 2 [1xkpx3]
    rot1    pose of KP1 [1x3x3]
    rot2    pose of KP2 [1x3x3]

    rot     GT relative pose b/w kp1 and kp2

    Returns     Loss => Error in relative pose b/w kp1 and kp2 [Forbunius Norm]
    -------

    '''
    device = kp1.device
    gt_rot = torch.bmm(rot1.double(), torch.transpose(rot2.double(), 1, 2))



    mat = batch_compute_similarity_transform_torch(kp1, kp2)

    # ''' Just for visualizations:: Transformations are working fine'''
    # pdb.set_trace()
    # pcd1 = kp_to_pcd(kp1[0].cpu().numpy())
    # pcd2 = kp_to_pcd(kp2[0].cpu().numpy())
    # o3d.visualization.draw_geometries([pcd1, pcd2])
    # transf_kp2_to_kp1 = np.hstack([np.vstack([mat[0].cpu(), [0, 0, 0]]),[[0], [0], [0], [1]]])
    # mesh_t = copy.deepcopy(pcd2).transform(transf_kp2_to_kp1)
    # o3d.visualization.draw_geometries([pcd1, mesh_t])
    #
    # gt_transf = np.hstack([np.vstack([gt_rot[0].cpu(), [0, 0, 0]]), [[0], [0], [0], [1]]])
    # mesh_t_gt = copy.deepcopy(pcd2).transform(gt_transf)


    # mat = batch_compute_similarity_transform_torch(kp1.permute(0, 2, 1), kp2.permute(0, 2, 1))
    frob = torch.sqrt(torch.sum(torch.square(gt_rot - mat)))  # Forbunius Norm
    angle_ = torch.rad2deg(2.0 * torch.mean(torch.arcsin(torch.clamp(torch.min(torch.tensor(1.).to(device), frob / (2. * torch.sqrt(torch.tensor(2.).to(device)))), -0.99999, 0.99999))))
    return angle_


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
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))  # position
    R = torch.linalg.inv(R)  # rotation

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


def save_kp_and_pc_in_pcd(pc, kp, output_dir, save=True, name=""):
    '''

    Parameters
    ----------
    points      point cloud  [2048, 3]
    kp          estimated key-points  [10, 3]
    both        if plot both or just the point clouds

    Returns     show the key-points/point cloud
    -------

    '''

    palette_PC = sns.color_palette()
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")

    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008) ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    ''' Add Keypoitnts '''
    for i in range(0, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.035) # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i==7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point



    if save:
        if not os.path.exists(output_dir+'/ply'):
            os.makedirs(output_dir+'/ply')
        o3d.io.write_triangle_mesh("{}/{}.ply".format(output_dir+'/ply', name), pcd)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if not os.path.exists(output_dir+'/png'):
            os.makedirs(output_dir+'/png')
        vis.capture_screen_image("{}/{}.png".format(output_dir+'/png', name))
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd])

    # pdb.set_trace()


