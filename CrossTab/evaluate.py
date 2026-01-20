import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)  # 全局设置 PYTHONHASHSEED

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn

# from datasets.ImageDataset import ImageDataset
# from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
# from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.utils import grab_arg_from_checkpoint, prepend_paths
import sys
from models.Tip_utils.Tip_downstream import TIPBackbone
import logging
from tqdm import tqdm
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed, rank=0):
    """设置全局随机种子，确保所有随机源一致"""
    seed = seed + rank  # 每个进程使用不同种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

def get_dataset(hparams):
    train_dataset = ImagingAndTabularDataset(
        hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, 
        hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_train_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=True, target=hparams.target, corruption_rate=hparams.corruption_rate,
        data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate, 
        augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name
    )
    val_dataset = ImagingAndTabularDataset(
        hparams.data_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, 
        hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_val_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=hparams.corruption_rate,
        data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
        augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name
    )
    
    test_dataset = ImagingAndTabularDataset(
        hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
        hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
        data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
        augmentation_speedup=hparams.augmentation_speedup
    )
    hparams.input_size = train_dataset.get_input_size()
    logger.info("Datasets loaded successfully")
    return train_dataset, val_dataset, test_dataset

import time
def evaluate_model(model, dataloader, device, num_classes=2):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for data_iter_step, data in enumerate(dataloader):
            x, y = data
            time1 = time.time()
            x[0], x[1], x[3], y = x[0].to(device), x[1].to(device), x[3].to(device), y.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1).detach().cpu()
            preds = torch.argmax(probs, dim=1) if num_classes > 2 else (probs[:, 1] > 0.5).long()
            time2 = time.time()
            all_preds.extend(preds.numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(probs.numpy())
            print(x[0].shape, time2 - time1)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_logits, average='macro', multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='macro')
    pre = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    
    metrics = {"acc": acc, "auc": auc, "f1": f1, "precision": pre, "recall": rec}
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

import misc
from torch.utils.data import Subset

def train(hparams):
    """训练主函数"""
    misc.init_dist_pytorch(hparams)
    device = torch.device(hparams.gpu)
    rank = misc.get_rank()
    set_seed(hparams.seed, rank)  
    torch.set_default_dtype(torch.float32)

    base_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'result')
    os.makedirs(base_dir, exist_ok=True)
    save_dir = os.path.join(base_dir, 'runs', 'eval', f"{hparams.exp_name}_{hparams.target}")
    os.makedirs(save_dir, exist_ok=True)


    train_dataset, val_dataset, test_dataset = get_dataset(hparams)
    hparams.world_size = misc.get_world_size()

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=hparams.world_size,
        rank=rank,
        shuffle=False
    )

    data_loader_train = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=hparams.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8, 
        worker_init_fn=worker_init_fn 
    )

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    data_loader_test = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    if rank == 0:
        print(hparams.checkpoint)
    model = TIPBackbone(hparams).to(device)
    hparams.checkpoint = None
    if rank == 0:
        print('hparams', hparams)
    dist.barrier()
    model = DistributedDataParallel(model, static_graph=True)
    model_without_ddp = model.module

    is_main_process = rank == 0
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    
    if is_main_process:
        logger.info(f"[Trainable Params]: {total_trainable:.2f} M")
        logger.info(f"[Frozen Params]: {total_non_trainable:.2f} M")
        logger.info(f"[Total Params]: {(total_trainable + total_non_trainable):.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr_eval, weight_decay=hparams.weight_decay_eval)
    best_score = 0

    for epoch in range(hparams.max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        progress_bar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{hparams.max_epochs}", leave=True)
        for data_iter_step, data in enumerate(progress_bar):
            x, y = data
            x[0], x[1], x[3], y = x[0].to(device), x[1].to(device), x[3].to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Optionally update progress bar with current loss
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if is_main_process:
            avg_loss = total_loss / len(data_loader_train)
            logger.info(f"[Epoch {epoch+1}] Average Training Loss: {avg_loss:.4f}")
            val_metrics = evaluate_model(model.module, data_loader_val, device, num_classes=hparams.num_classes)
            logger.info(f"[Epoch {epoch+1}] Val Acc: {val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            score = val_metrics['acc'] + val_metrics['auc'] + val_metrics['f1']
            if score > best_score:
                best_score = score
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pt"))
                pd.DataFrame([val_metrics]).to_csv(os.path.join(save_dir, "eval_results.csv"), index=False)
                logger.info(f"Saved best model with score {best_score:.4f}")

    if is_main_process:
        state_dict = torch.load(os.path.join(save_dir, "best_model.pt"))
        model.module.load_state_dict(state_dict)
        test_metrics = evaluate_model(model.module, data_loader_test, device, num_classes=hparams.num_classes)
        pd.DataFrame([test_metrics]).to_csv(os.path.join(save_dir, "test_results.csv"), index=False)
        logger.info(f"Test Metrics: {test_metrics}")

    dist.destroy_process_group()

@hydra.main(config_path='./configs', config_name='config_adni_TIP', version_base=None)
def control(args: DictConfig):
    OmegaConf.set_struct(args, False)
    args = prepend_paths(args)
    logger.info("Starting training...")
    train(args)
    logger.info("Training completed")

if __name__ == "__main__":
    control()