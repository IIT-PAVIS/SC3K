import os
import hydra
import torch
import omegaconf
from tqdm import tqdm
# from utils import AverageMeter
import data_loader as dataset
import metrics as function_bank
import visualizations as visualizer
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
import network
import logging
logger = logging.getLogger(__name__)
BASEDIR = os.path.dirname(os.path.abspath(__file__))


def test_canonical_pose(cfg):
    KeypointDataset = getattr(dataset, '{}_data_loader'.format(cfg.task))

    test_dataset = KeypointDataset(cfg, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, drop_last=True)

    model = network.sc3k(cfg).cuda()
    best_model_path = os.path.join(BASEDIR, cfg.data.best_model_path)
    model.load_state_dict(torch.load(best_model_path))

    coverage_ = []
    inclusivity_ = []
    keypoints = []
    not_repeat = []
    with torch.no_grad():
        model.eval()
        for batch_id, batch_pcd in enumerate(tqdm(test_dataloader)):
            batch_pred = model(batch_pcd)

            keypoints.append(batch_pred[0].cpu().numpy())
            coverage_.append(function_bank.coverage(batch_pred, batch_pcd[0].cuda()).cpu().numpy()) # [10x3], [2048x3]
            inclusivity_.append(torch.mean(function_bank.inclusivity(batch_pred[0], batch_pcd[0][0].cuda(), threshold=0.05)).cpu().numpy())  # [1x10x3], [1x2048x3]

            ''' Save the qualitative results'''
            if cfg.save_results and len(not_repeat) < 15:
                if batch_pcd[1][0] not in not_repeat:
                    not_repeat.append(batch_pcd[1][0])
                    visualizer.save_kp_and_pc_in_pcd(batch_pcd[0][0], batch_pred[0].cpu().numpy(), '{}_visualizations'.format(cfg.task), save=True,name="{}_".format(batch_id) + batch_pcd[1][0])

    DAS_unsup = function_bank.DAS_unsupervised(keypoints)

    logger.info('Avg. coverage_useek: {:.2f}'.format(np.mean(coverage_)))
    logger.info('Avg. inclusivity_useek: {:.2f}'.format(np.mean(inclusivity_)))
    logger.info('Avg. DAS_unsup: {:.2f}'.format(DAS_unsup))


def test_generic_pose(cfg):

    KeypointDataset = getattr(dataset, '{}_data_loader'.format(cfg.task))

    test_dataset = KeypointDataset(cfg, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=cfg.num_workers, drop_last=True)

    model = network.sc3k(cfg).cuda()
    best_model_path = os.path.join(BASEDIR, cfg.data.best_model_path)
    model.load_state_dict(torch.load(best_model_path))

    coverage_ = []
    inclusivity_ = []
    keypoints = []
    pose_error = []
    kp1_generic = []
    not_repeat = []
    with torch.no_grad():
        model.eval()
        for batch_id, batch_pcd in enumerate(tqdm(test_dataloader)):
            batch_pred, batch_pred2 = model(batch_pcd)

            pose_error.append(function_bank.pose_loss(batch_pred, batch_pred2, batch_pcd[1].float().cuda(), batch_pcd[3].float().cuda()).cpu().numpy())  # pose/2 => because its pose*2
            kp1_generic.append(torch.transpose(torch.bmm(torch.transpose(batch_pcd[1].cuda().double(), 1, 2), torch.transpose(batch_pred.double(), 1, 2)), 1, 2).cpu().numpy()[0])
            keypoints.append(batch_pred[0].cpu().numpy())
            coverage_.append(function_bank.coverage(batch_pred, batch_pcd[0].cuda()).cpu().numpy()) # [10x3], [2048x3]
            inclusivity_.append(torch.mean(function_bank.inclusivity(batch_pred[0], batch_pcd[0][0].cuda(), threshold=0.05)).cpu().numpy())  # [1x10x3], [1x2048x3]

            ''' Save the qualitative results'''
            if cfg.save_results and len(not_repeat) < 20:
                if batch_pcd[4][0] not in not_repeat:
                    not_repeat.append(batch_pcd[4][0])
                    # pdb.set_trace()
                    visualizer.save_kp_and_pc_in_pcd(batch_pcd[0][0], batch_pred[0].cpu().numpy(), '{}_visualizations'.format(cfg.task), save=True,name="{}_".format(batch_id) + batch_pcd[4][0])


    DAS_unsup = function_bank.DAS_unsupervised(kp1_generic)

    logger.info('Avg. coverage_useek: {:.2f}'.format(np.mean(coverage_)))
    logger.info('Avg. inclusivity_useek: {:.2f}'.format(np.mean(inclusivity_)))
    logger.info('Avg. DAS_unsup: {:.2f}'.format(DAS_unsup))
    logger.info('Avg. pose_error: mean: {} median: {}'.format(np.mean(pose_error),np.median(pose_error)))



@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    logger.info(cfg.pretty())

    if cfg.split != "test":
        print("Please set cfg.split as \'test\' in the configuration file")
        return 0

    if cfg.task == "generic":
        test_generic_pose(cfg)
    elif cfg.task == "canonical":
        test_canonical_pose(cfg)
    else:
        print("Invalid task. Please check \'cfg/task\'")


if __name__ == '__main__':
    main()

