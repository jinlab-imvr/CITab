import os 
import sys

import torch
from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DistributedSampler
from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveReconstructImagingAndTabularDataset import ContrastiveReconstructImagingAndTabularDataset
from datasets.ContrastiveReconstructImagingAndTabularHeaderDataset import ContrastiveReconstructImagingAndTabularHeaderDataset
from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset
from datasets.MaskTabularDataset import MaskTabularDataset

from models.MultimodalSimCLR import MultimodalSimCLR
from models.SimCLR import SimCLR
from models.SwAV_Bolt import SwAV
from models.BYOL_Bolt import BYOL
from models.SimSiam_Bolt import SimSiam
from models.BarlowTwins import BarlowTwins
from models.SCARF import SCARF
from models.VIME import VIME
from models.Tips.TipModel3Loss import TIP3Loss
import warnings
warnings.filterwarnings(action='ignore')

def load_datasets(hparams):

  transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
  hparams.transform = transform.__repr__()
  # train_dataset = ContrastiveReconstructImagingAndTabularDataset(
  #   hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
  #   hparams.data_train_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
  #   hparams.field_lengths_tabular, hparams.one_hot,
  #   hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
  # val_dataset = ContrastiveReconstructImagingAndTabularDataset(
  #   hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
  #   hparams.data_val_tabular, hparams.corruption_rate,  hparams.replace_random_rate, hparams.replace_special_rate,
  #   hparams.field_lengths_tabular, hparams.one_hot,
  #   hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)

  train_dataset = ContrastiveReconstructImagingAndTabularHeaderDataset(
    hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
    hparams.data_train_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
    hparams.field_lengths_tabular, hparams.one_hot,
    hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
  val_dataset = ContrastiveReconstructImagingAndTabularHeaderDataset(
    hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
    hparams.data_val_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
    hparams.field_lengths_tabular, hparams.one_hot,
    hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)

  print(hparams.img_size, hparams.target, hparams.augmentation_speedup, hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate,
    hparams.data_train_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
    hparams.field_lengths_tabular, hparams.one_hot,
    hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)

  return train_dataset, val_dataset


def pretrain(hparams, wandb_logger):

  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True, drop_last=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True, drop_last=True)

  print(f"Number of training batches: {len(train_loader)}")
  print(f"Number of validation batches: {len(val_loader)}")
  print(f'Valid batch size: {hparams.batch_size*cuda.device_count()}')

  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  
  model = TIP3Loss(hparams)
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    z_dim =  hparams.multimodal_embedding_dim if hparams.strategy=='tip' else model.pooled_dim
    # z_dim = 2560
    callbacks.append(SSLOnlineEvaluator(z_dim = z_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal'), 
                                        strategy=hparams.strategy))
  
  callbacks.append(
    ModelCheckpoint(
      filename='checkpoint_last_epoch_{epoch:02d}',
      dirpath=logdir,
      save_on_train_epoch_end=True,
      auto_insert_metric_name=False,
      every_n_epochs=50,
      save_top_k=-1  # 每隔 50 个 epoch 保存一次
    )
  )
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))
  print('start training....')
  trainer = Trainer.from_argparse_args(hparams, gpus=cuda.device_count(),
                                       callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, 
                                       limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar,
                                       )
  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)

  else:
    trainer.fit(model, train_loader, val_loader)
