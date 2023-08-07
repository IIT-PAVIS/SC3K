import hydra
import torch
import omegaconf
from tqdm import tqdm
import data_loader as dataset
from utils import AverageMeter
import utils as function_bank
import network
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)

def train(cfg):
    writer = SummaryWriter("train_summary")

    KeypointDataset = getattr(dataset, 'generic_data_loader')

    train_dataset = KeypointDataset(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, drop_last=False)

    val_dataset = KeypointDataset(cfg, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                 num_workers=cfg.num_workers, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.sc3k(cfg).to(device) # cuda()   # unsupervised network
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    meter = AverageMeter()
    best_loss = 1e10
    train_step = 0
    val_step = 0
    for epoch in range(cfg.max_epoch):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for i, data in enumerate(train_iter):

            kp1, kp2 = model(data)
            loss = function_bank.compute_loss(kp1, kp2, data, writer, train_step, cfg, split='train')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
            writer.add_scalar('train_loss/overall', loss, train_step)  # write training loss
            train_step += 1  # increment in train_step

        train_loss = meter.avg
        logger.info(f'Epoch: {epoch}, Average Train loss: {meter.avg}')

        # validation loss
        model.eval()
        meter.reset()
        val_iter = tqdm(val_dataloader)
        for i, data in enumerate(val_iter):
            with torch.no_grad():
                kp1, kp2 = model(data)
                loss = function_bank.compute_loss(kp1, kp2, data, writer, val_step, cfg, split='val')

                writer.add_scalar('val_loss/overall', loss, val_step)  # write validation loss
                val_step += 1  # increment in val_step

            val_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
        val_loss = meter.avg
        if val_loss < best_loss:
           logger.info("best epoch: {}".format(epoch))
           best_loss = meter.avg
           torch.save(model.state_dict(),'Best_{}_{}kp.pth'.format(cfg.class_name, cfg.key_points))

        logger.info(f'Epoch: {epoch}, Average Val loss: {meter.avg}')
        writer.add_scalars('loss_per_epoch', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)  # write validation loss

    writer.close()  # close the summary writer
    logger.info(" Reached to {} epoch \n".format(cfg.max_epoch))
    torch.save(model.state_dict(),  '{}_{}kp_{}.pth'.format(cfg.class_name, cfg.key_points, cfg.max_epoch))



@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_logs'.format(cfg.task)
    logger.info(cfg.pretty())
    cfg.task = 'generic'
    train(cfg)


if __name__ == '__main__':
    main()

