"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch
import numpy as np
# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm, get_grad_norm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            batch.y = batch.y[0].view(1, -1)
            batch.w = batch.w[0].view(1, -1)
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            self.logger.debug(f'output size: {batch_output.shape}')
            batch_pred = torch.sigmoid(batch_output)

            logging.debug(f'match type and y type {type(batch_pred)} {type(batch.y)}')
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()

            #batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.y.float().float(), weight=batch.w.float())
            logging.debug(f'batch w size : {batch.w.shape}')
            batch_loss = self.loss_func(batch_output, batch.y.float(), weight=batch.w)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            predict_noise = batch_pred > 0.5
            predict_hits = batch_pred < 0.5
            true_noise = batch.y == 1
            true_hits = batch.y == 0
            self.logger.debug(f'\n--train batch predict noise: {predict_noise.sum().item()} true hits: {predict_hits.sum().item()} \n--train batch ground  noise: {true_noise.sum().item()} true hits: {true_hits.sum().item()}')
            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        summary['train_acc'] = sum_correct / sum_total
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f acc: %.3f', summary['train_loss'], summary['train_acc'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch.y = batch.y[0].view(1, -1)
            batch.w = batch.w[0].view(1, -1)
            batch = batch.to(self.device)
            # Make predictions on this batch
            batch_output = self.model(batch)
            #batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.y.float()).item()
            batch_loss = self.loss_func(batch_output, batch.y.float()).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            predict_noise = batch_pred > 0.5
            predict_hits = batch_pred < 0.5
            true_noise = batch.y == 1
            true_hits = batch.y == 0
            self.logger.debug(f'\n--val batch predict noise: {predict_noise.sum().item()} true hits: {predict_hits.sum().item()} \n--val batch ground  noise: {true_noise.sum().item()} true hits: {true_hits.sum().item()}')
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
