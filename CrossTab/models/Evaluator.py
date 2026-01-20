from typing import Tuple
import os
import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
  classification_report, roc_auc_score, roc_curve


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.eval_datatype == 'imaging':
      self.model = ImagingModel(self.hparams)
    elif self.hparams.eval_datatype == 'multimodal':
      assert self.hparams.strategy == 'tip'
      if self.hparams.finetune_ensemble == True:
        self.model = TIPBackboneEnsemble(self.hparams)
      else:
        self.model = TIPBackbone(self.hparams)
    elif self.hparams.eval_datatype == 'tabular':
        self.model = TabularModel(self.hparams)
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
      if self.hparams.algorithm_name == 'DAFT':
        self.model = DAFT(self.hparams)
      elif self.hparams.algorithm_name in set(['CONCAT','MAX']):
        if self.hparams.strategy == 'tip':
          # use TIP's tabular encoder
          self.model = MultimodalModelTransformer(self.hparams)
        else:
          # use MLP-based tabular encoder
          self.model = MultimodalModel(self.hparams)
      elif self.hparams.algorithm_name == 'MUL':
        self.model = MultimodalModelMUL(self.hparams)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    
    self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

    self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

    self.pre_test = torchmetrics.Precision(task=task, num_classes=self.hparams.num_classes, average='macro')
    self.recall_test = torchmetrics.Recall(task=task, num_classes=self.hparams.num_classes, average='macro')
    self.f1_test = torchmetrics.F1Score(task=task, num_classes=self.hparams.num_classes, average='macro')

    self.criterion = torch.nn.CrossEntropyLoss()
    
    self.best_val_score = 0
    self.test_preds = []
    self.test_labels = []
    self.multi_modal = []

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat = self.model(x)

    # Needed for gradcam
    if len(y_hat.shape)==1:
      y_hat = torch.unsqueeze(y_hat, 0)

    return y_hat

    return y_hat
  
  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    """
    Runs test step
    """
    x, y = batch 
    y_hat = self.forward(x)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    preds = torch.argmax(y_hat, dim=1) if self.hparams.num_classes > 2 else (y_hat > 0.5).long()
    self.test_preds.append(preds.cpu())
    self.test_labels.append(y.cpu())
    # self.multi_modal.append(m.cpu())

    self.acc_test(y_hat, y)
    self.auc_test(y_hat, y)
    self.pre_test(y_hat, y)
    self.recall_test(y_hat, y)
    self.f1_test(y_hat, y)

  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_acc = self.acc_test.compute()
    test_auc = self.auc_test.compute()
    test_pre = self.pre_test.compute()
    test_recall = self.recall_test.compute()
    test_f1 = self.f1_test.compute()
    self.log('test.acc', test_acc)
    self.log('test.auc', test_auc)
    self.log('test.pre', test_pre)
    self.log('test.recall', test_recall)
    self.log('test.f1', test_f1)

    preds = torch.cat(self.test_preds, dim=0)
    labels = torch.cat(self.test_labels, dim=0)

    conf_matrix = confusion_matrix(labels.numpy(), preds.numpy())
    print(conf_matrix)

    # self.dict = {'label': labels, 'embeddings': torch.cat(self.multi_modal, dim=0)}
    # torch.save(self.dict, '/mnt/zt/fyb/BrainMRI-CrossTab-Pretraining/result/runs/adni_vanillav2.pt')

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_train(y_hat, y)
    self.auc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)

    return loss

  def training_epoch_end(self, _) -> None:
    """
    Compute training epoch metrics and check for new best values
    """
    self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
    self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)

    y_hat = torch.softmax(y_hat.detach(), dim=1)
    if self.hparams.num_classes==2:
      y_hat = y_hat[:,1]

    self.acc_val(y_hat, y)
    self.auc_val(y_hat, y)
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_acc_val = self.acc_val.compute()
    epoch_auc_val = self.auc_val.compute()

    self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
    self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
  
    if self.hparams.eval_metric == 'acc':
      self.best_val_score = max(self.best_val_score, epoch_acc_val + epoch_auc_val)
    else:
      self.best_val_score = max(self.best_val_score, epoch_auc_val + epoch_acc_val)
    print('acc', epoch_acc_val, 'auc:', epoch_auc_val)

    # if self.hparams.target == 'dvm':
    #   self.best_val_score = max(self.best_val_score, epoch_acc_val)
    # else:
    #   self.best_val_score = max(self.best_val_score, epoch_acc_val)


    self.acc_val.reset()
    self.auc_val.reset()

  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer
    
    return (
      {
        "optimizer": optimizer, 
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )