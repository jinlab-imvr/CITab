'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''
from typing import Tuple, List

import torch
from torch import nn

# class ReconstructionLoss(torch.nn.Module):
#   """
#   Loss function for tabular data reconstruction.
#   Loss function for multimodal contrastive learning based off of the CLIP paper.
#
#   Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
#   similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
#   Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
#   """
#   def __init__(self,
#                num_cat: int, num_con: int, cat_offsets: torch.Tensor) -> None:
#     super(ReconstructionLoss, self).__init__()
#
#     self.num_cat = num_cat
#     self.num_con = num_con
#     self.register_buffer('cat_offsets', cat_offsets, persistent=False)
#     self.softmax = nn.Softmax(dim=1)
#
#   def forward(self, out: Tuple, y: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
#
#     B, _, D = out[0].shape
#     # (B*N1, D)
#     out_cat = out[0].reshape(B*self.num_cat, D)
#     # (B, N2)
#     out_con = out[1].squeeze(-1)
#     target_cat = (y[:, :self.num_cat].long()+self.cat_offsets).reshape(B*self.num_cat)
#     target_con = y[:, self.num_cat:]
#     mask_cat = mask[:, :self.num_cat].reshape(B*self.num_cat)
#     mask_con = mask[:, self.num_cat:]
#
#     # cat loss
#     prob_cat = self.softmax(out_cat)
#     onehot_cat = torch.nn.functional.one_hot(target_cat, num_classes=D)
#     loss_cat = -onehot_cat * torch.log(prob_cat+1e-8)
#     loss_cat = loss_cat.sum(dim=1)
#     loss_cat = (loss_cat*mask_cat).sum()/mask_cat.sum()
#
#     # con loss
#     loss_con = (out_con-target_con)**2
#     loss_con = (loss_con*mask_con).sum()/mask_con.sum()
#
#     loss = (loss_cat + loss_con)/2
#
#     return loss, prob_cat, target_cat, mask_cat


class ReconstructionLoss(torch.nn.Module):
  """
  Loss function for tabular data reconstruction.
  Loss function for multimodal contrastive learning based off of the CLIP paper.

  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
  """

  def __init__(self) -> None:
    super(ReconstructionLoss, self).__init__()

  def forward(self, out: Tuple, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

    out_con = out.squeeze(-1)
    loss_con = (out_con - y) ** 2
    loss_con = (loss_con * mask).sum() / mask.sum()

    return loss_con

if __name__ == '__main__':
  # loss_func = ReconstructionLoss()
  # x = (torch.randn(5, 2, 10), torch.randn(5, 6, 1))
  # print(x)
  # y = torch.cat([torch.randint(0,4,(5,2)).float(), torch.randn(5,4) ],dim=1)
  # print(y)
  # mask = torch.tensor([[True, False, False, False, True, True],
  #                      [False, True, False, False, True, True],
  #                      [False, True, False, False, True, True],
  #                      [False, True, False, False, True, True],
  #                      [False, True, False, False, True, True],])
  # print(y.shape)
  # loss = loss_func(x[1], y, mask)
  # print(loss)
  print('-------')