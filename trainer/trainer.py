import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None, writer=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.writer = writer
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            train_steps = epoch * len(self.data_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), train_steps)
            acc_metrics = np.zeros(len(self.metrics))
            for i, metric in enumerate(self.metrics):
                acc_metrics[i] += metric(output, target)
                self.writer.add_scalar(f'train/{metric.__name__}', acc_metrics[i], train_steps)
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.writer.add_image('train/input', make_grid(data[:32].cpu(), nrow=4), train_steps)
                self.writer.add_image('train/target', make_grid(target[:32].cpu(), nrow=4), train_steps)
                self.writer.add_image('train/output', make_grid(output[:32].cpu(), nrow=4), train_steps)
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    # len(self.data_loader) * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
            total_metrics += acc_metrics
            total_loss += loss.item()

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                valid_steps = epoch * len(self.valid_data_loader) + batch_idx
                self.writer.add_scalar('valid/loss', loss.item(), valid_steps)
                acc_metrics = np.zeros(len(self.metrics))
                for i, metric in enumerate(self.metrics):
                    acc_metrics[i] += metric(output, target)
                    self.writer.add_scalar(f'valid/{metric.__name__}', acc_metrics[i], valid_steps)
                self.writer.add_image('valid/input', make_grid(data[:32].cpu(), nrow=4), valid_steps)
                self.writer.add_image('valid/target', make_grid(target[:32].cpu(), nrow=4), valid_steps)
                self.writer.add_image('valid/output', make_grid(output[:32].cpu(), nrow=4), valid_steps)
                total_val_loss += loss.item()
                total_val_metrics += acc_metrics

                self.scheduler.step(loss.item())
                
        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
