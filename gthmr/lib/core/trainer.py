import os
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from VIBE.lib.core.config import VIBE_DATA_DIR
from VIBE.lib.utils.utils import move_dict_to_device, AverageMeter

from VIBE.lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

from gthmr.lib.utils.mdm_utils import viz_motions
from nemo.utils.misc_utils import to_np, to_tensor
import ipdb

logger = logging.getLogger(__name__)


class Trainer():

    def __init__(
        self,
        data_loaders,
        generator,
        gen_optimizer,
        end_epoch,
        criterion,
        start_epoch=0,
        lr_scheduler=None,
        device=None,
        writer=None,
        debug=False,
        debug_freq=1000,
        logdir='output',
        resume=None,
        performance_type='min',
        num_iters_per_epoch=1000,
    ):

        # Prepare dataloaders
        self.train_2d_loader, self.train_3d_loader, self.valid_loader, self.train_eval_loader = data_loaders

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float(
            'inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(
            ['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resume from a pretrained model
        if resume is not None:
            self.resume_pretrained(resume)

    def train(self):
        # Single epoch training routine

        losses = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}',
                  fill='#',
                  max=self.num_iters_per_epoch)

        for i in range(self.num_iters_per_epoch):
            # Dirty solution to reset an iterator
            target_2d = target_3d = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)

            # <======= Feedforward generator
            if target_2d and target_3d:
                inp = torch.cat((target_2d['features'], target_3d['features']),
                                dim=0).to(self.device)
            elif target_3d:
                inp = target_3d['features'].to(self.device)
            else:
                inp = target_2d['features'].to(self.device)

            timer['data'] = time.time() - start
            start = time.time()

            preds = self.generator(inp)

            timer['forward'] = time.time() - start
            start = time.time()

            gen_loss, loss_dict = self.criterion(
                generator_outputs=preds,
                data_2d=target_2d,
                data_3d=target_3d,
            )
            if 'extra_loss' in preds[-1]:
                extra_loss = preds[-1]['extra_loss']
                gen_loss = gen_loss + extra_loss
                loss_dict['extra_loss'] = extra_loss
            # =======>

            timer['loss'] = time.time() - start
            start = time.time()

            # <======= Backprop generator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

            # =======>

            # <======= Log training info
            total_loss = gen_loss

            losses.update(total_loss.item(), inp.size(0))

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer[
                'loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.2f}'
                self.writer.add_scalar('train_loss/' + k,
                                       v,
                                       global_step=self.train_global_step)

            for k, v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss',
                                   total_loss.item(),
                                   global_step=self.train_global_step)

            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video,
                                                       preds[-1],
                                                       target_3d.copy(),
                                                       vis_hmr=False,
                                                       dataset=dataset)
                self.writer.add_video('train-video',
                                      vid_tensor,
                                      global_step=self.train_global_step,
                                      fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

    def validate(self, which_set='valid'):

        self.generator.eval()

        start = time.time()

        summary_string = ''

        prefix = 'Validation' if which_set == 'valid' else 'Valid on Train'
        bar = Bar(prefix, fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k, v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(
            np.load(osp.join(VIBE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        loader = self.valid_loader if which_set == 'valid' else self.train_eval_loader

        for i, target in enumerate(loader):
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['features']

                preds = self.generator(inp, J_regressor=J_regressor)

                # Viz MDM style
                # if i >= 0:
                if i == 0:
                    # Process GT data
                    smpl = self.generator.my_regressor.smpl
                    theta = to_tensor(target['theta'][:, :,
                                                      3:75]).reshape(-1, 72)
                    zero_betas = torch.zeros((theta.shape[0], 10)).cuda()
                    gt_output = smpl(betas=zero_betas,
                                     body_pose=theta[:, 3:],
                                     global_orient=theta[:, :3],
                                     pose2rot=True)
                    gt_joints = gt_output.smpl_joints
                    if 'smpl_output' in preds[-1]:
                        pred_joints = preds[-1]['smpl_output'].smpl_joints
                    else:
                        pred_joints = preds[-1]['smpl_joint']

                    def proc_smpl_joints(smpl_joints):
                        n_joints = smpl_joints.shape[1]
                        smpl_joints = smpl_joints.reshape(
                            preds[-1]['batch_size'], preds[-1]['seqlen'],
                            n_joints,
                            3)[:, :, :22]  # (N, T, 22, 3), ignore hands
                        smpl_joints = smpl_joints.permute(0, 2, 3, 1)
                        smpl_joints = to_np(smpl_joints)
                        return smpl_joints

                    pred_joints = proc_smpl_joints(pred_joints)
                    gt_joints = proc_smpl_joints(gt_joints)

                    # viz_batch_size = preds[-1]['batch_size']
                    batch_size = preds[-1]['batch_size']
                    viz_batch_size = 10
                    idxs = np.arange(0, batch_size, batch_size //
                                     viz_batch_size)[:viz_batch_size]

                    log_dir = self.writer.get_logdir()
                    log_dir = osp.join(log_dir, f'{which_set}_{self.epoch:06d}_{i:06d}')
                    os.makedirs(log_dir, exist_ok=True)
                    N = gt_joints.shape[0]
                    

                    if 'smpl_joints_w_trans' in preds[-1]:
                        smpl_joints_w_trans = proc_smpl_joints(preds[-1]['smpl_joints_w_trans'])
                        all_joints = np.vstack(
                            [gt_joints[idxs], pred_joints[idxs],smpl_joints_w_trans[idxs]])
                        all_text = ['gt'] * N + ['pred'] * N + ['+ trans'] * N
                        viz_motions(viz_batch_size,
                                    3,
                                    log_dir,
                                    all_joints,
                                    all_text=all_text)
                    else:
                        all_joints = np.vstack(
                            [gt_joints[idxs], pred_joints[idxs]])
                        all_text = ['gt'] * N + ['pred'] * N
                        viz_motions(viz_batch_size,
                                    2,
                                    log_dir,
                                    all_joints,
                                    all_text=all_text)

                # convert to 14 keypoint format for evaluation
                n_kp = preds[-1]['kp_3d'].shape[-2]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()

            
                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)

                if 'theta' in preds[-1]:
                    pred_verts = preds[-1]['verts'].view(-1, 6890,
                                                         3).cpu().numpy()
                    target_theta = target['theta'].view(-1, 85).cpu().numpy()

                    self.evaluation_accumulators['pred_verts'].append(
                        pred_verts)
                    self.evaluation_accumulators['target_theta'].append(
                        target_theta)
            # =============>

            # # <============= DEBUG
            # if self.debug and self.valid_global_step % self.debug_freq == 0:
            #     from lib.utils.vis import batch_visualize_vid_preds
            #     video = target['video']
            #     dataset = 'common'
            #     vid_tensor = batch_visualize_vid_preds(video,
            #                                            preds[-1],
            #                                            target,
            #                                            vis_hmr=False,
            #                                            dataset=dataset)
            #     self.writer.add_video('valid-video',
            #                           vid_tensor,
            #                           global_step=self.valid_global_step,
            #                           fps=10)
            # # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            if which_set == 'valid':
                self.valid_global_step += 1
            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)


    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.validate('valid')
            performance = self.evaluate()
            self.validate('train')
            self.evaluate()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr',
                                       param_group['lr'],
                                       global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

            self.train()


        self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performance': performance,
            'gen_optimizer': self.gen_optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename,
                            osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.best_performance = checkpoint['performance']

            logger.info(
                f"=> loaded checkpoint '{model_path}' "
                f"(epoch {self.start_epoch}, performance {self.best_performance})"
            )
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            if len(self.evaluation_accumulators[k]) > 0:
                self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
        target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        # Absolute error (MPJPE)
        errors = torch.sqrt(
            ((pred_j3ds -
              target_j3ds)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(
            pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(
            ((S1_hat -
              target_j3ds)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        m2mm = 1000

        if len(self.evaluation_accumulators['pred_verts']) > 0:
            pred_verts = self.evaluation_accumulators['pred_verts']
            target_theta = self.evaluation_accumulators['target_theta']
            pve = np.mean(
                compute_error_verts(target_theta=target_theta,
                                    pred_verts=pred_verts)) * m2mm
        else:
            pve = -1

        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(
            compute_error_accel(joints_pred=pred_j3ds,
                                joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v:.4f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        for k, v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe
