'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/datasets/ContrastiveImagingAndTabularDataset.py
'''
from typing import List, Tuple
import random
import csv
import copy
import nibabel as nib
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
import numpy as np
import torchio as tio

def convert_to_float(x):
  return x.float()

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


class ContrastiveReconstructImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.
  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is masked and replaced with mask_rate and replace_rate
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, replace_random_rate: float, replace_special_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: tuple, live_loading: bool, augmentation_speedup: bool=False) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    img_size = tuple(img_size)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]
    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = tio.Compose([
      tio.Resize(img_size),  # 将图像调整为指定尺寸
      tio.Lambda(convert_to_float)  # 转换为 float 类型
    ])

    # Tabular
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.data_tabular, self.valid_mask = self.read_and_parse_csv(data_path_tabular)
    self.generate_marginal_distributions()
    self.c = corruption_rate
    self.one_hot_tabular = one_hot_tabular
    self.replace_random_rate = replace_random_rate
    self.replace_special_rate = replace_special_rate

    # Classifier
    self.labels = torch.load(labels_path)
    assert len(self.data_imaging) == len(self.data_tabular) == len(self.labels)
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    directly loading from pt file.
    """
    data = torch.load(path_tabular)
    padded_list = []
    mask_list = []

    for arr in data:
      current_len = len(arr)
      padded_array = np.pad(arr, (0, len(self.field_lengths_tabular) - current_len), mode='constant', constant_values=0)
      mask = np.zeros(len(self.field_lengths_tabular), dtype=np.int32)
      mask[:current_len] = 1
      padded_list.append(padded_array)
      mask_list.append(mask)

    return padded_list, mask_list

  def generate_marginal_distributions(self) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data = np.array(self.data_tabular)
    self.marginal_distributions = np.transpose(data)

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def corrupt(self, subject: List[float], valid_len) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    valid_len_matrix = np.sum(np.array(self.valid_mask), axis=1)
    valid_len_indices = np.where(valid_len_matrix == valid_len)[0]

    indices = random.sample(list(range(len(subject[:valid_len]))), int(len(subject[:valid_len])*self.c))
    pick_value_positions = np.random.choice(valid_len_indices.shape[0], size=len(indices))
    subject[indices] = self.marginal_distributions[indices, valid_len_indices[pick_value_positions]]

    return subject
  
  def mask(self, subject: List[float], valid_len) -> List[float]:
    '''
    Create a copy of a subject, selects
    some indices keeping the same
    some indices replacing their values with
    '''
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject[:valid_len]))), round(len(subject[:valid_len])*(self.replace_random_rate + self.replace_special_rate)))
    print(indices)
    num_random = int(len(indices)*self.replace_random_rate/(self.replace_random_rate + self.replace_special_rate))
    num_special = len(indices) - num_random
    # replace some positions with random sample from marginal distribution
    pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=num_random)
    subject[indices[:num_random]] = self.marginal_distributions[indices[:num_random], pick_value_positions]
    mask, mask_random, mask_special = np.zeros_like(subject, dtype=bool), np.zeros_like(subject, dtype=bool), np.zeros_like(subject, dtype=bool)

    mask[indices] = True
    mask_random[indices[:num_random]] = True
    mask_special[indices[num_random:]] = True

    mask[valid_len:] = True
    mask_random[valid_len:] = True
    mask_special[valid_len:] = True

    assert np.sum(mask) == np.sum(mask_special)

    # print(mask_special.sum()/sum(subject.shape), mask_random.sum()/sum(subject.shape), mask.sum()/sum(subject.shape))
    # print('pick value positions: ', pick_value_positions)

    return subject, mask, mask_special, mask_random 

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:

    im = self.data_imaging[index]
    if self.live_loading:
      nii_image = nib.load(im)
      im = nii_image.get_fdata()
      im = (im - np.min(im)) / (np.max(im) - np.min(im))
      im = torch.tensor(im, dtype=torch.float32).unsqueeze(0)

    ims = [self.transform(image=im)['image']] if self.augmentation_speedup else [self.transform(im)]

    if random.random() < self.augmentation_rate:
      ims.append(self.transform(image=im)['image'] if self.augmentation_speedup else self.transform(im))
    else:
      ims.append(self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im))

    orig_im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
    
    return ims, orig_im

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:

    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    # origin   augmented
    if self.c > 0:
      tabular_views = [torch.tensor(self.corrupt(self.data_tabular[index], np.sum(self.valid_mask[index])), dtype=torch.float)]
    else:
      tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float)]

    masked_view, mask, mask_special, mask_random =  self.mask(self.data_tabular[index], np.sum(self.valid_mask[index]))

    tabular_views.append(torch.from_numpy(masked_view).float())
    tabular_views = tabular_views + [torch.from_numpy(mask), torch.from_numpy(mask_special)]

    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]

    label = torch.tensor(self.labels[index], dtype=torch.long)
    unaugmented_tabular = torch.tensor(self.data_tabular[index], dtype=torch.float)  # tabular views: corrupt, raw, mask, mask_specical

    len_mask = self.valid_mask[index] != 0
    tabular_views = tabular_views + [torch.from_numpy(mask) & torch.tensor(len_mask, dtype=torch.bool)]

    return imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular

  def __len__(self) -> int:
    return len(self.data_tabular)
  

