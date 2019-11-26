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

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        threshold_list = np.linspace(0, 10, num=11)
        summary = dict()
        tp_list = np.zeros(len(threshold_list))
        fp_list = np.zeros(len(threshold_list))
        fn_list = np.zeros(len(threshold_list))
        tn_list = np.zeros(len(threshold_list))
        prediction = []
        gt = []
        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            # Make predictions on this batch
            batch_output = self.model(batch)

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            prediction = np.append(prediction, batch_pred)
            logging.info(f'prediction shape: {batch_pred.shape},gt shape: {batch.y.shape}')
            gt = np.append(gt, batch.y)
            for i in range(len(threshold_list)):
                tp_list[i] += np.logical_and(batch_pred >= threshold_list[i], batch.y == 1).sum().item()
                fp_list[i] += np.logical_and(batch_pred >= threshold_list[i], batch.y == 0).sum().item()
                fn_list[i] += np.logical_and(batch_pred < threshold_list[i], batch.y == 1).sum().item()
                tn_list[i] += np.logical_and(batch_pred < threshold_list[i], batch.y ==0).sum().item()

        # Summarize the validation epoch
        summary['tp'] = tp_list
        summary['fp'] = fp_list
        summary['fn'] = fn_list
        summary['tn'] = tn_list
        summary['pred'] = prediction
        summary['gt'] = gt
        #self.logger.debug(' Processed %i samples in %i batches',
        #                  len(data_loader.sampler), n_batches)
        #self.logger.info('  Validation acc: %.3f', (summary['tp'] + summary['tn'])/(summary['tp'] + summary['tn'] + summary['fp'] + summary['fn']))
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
