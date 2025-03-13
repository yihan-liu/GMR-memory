# train.py

import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from gmr_memory_dataset import GMRMemoryDataset
from model import GMRMemoryModelDualHead, GMRMemoryModelAdapted
from utils import *

def train_phase1(args):
    """
    Phase 1: Train only on circle and square data.
    Labels (e.g. "cscs" and "sccs") do not contain triangle,
    so the dataset returns target sequences with only 2 channels.
    """
    # labels = [k for k in KEY_FRAMES_DICT.keys() if 't' not in k] 
    labels = KEY_FRAMES_DICT.keys()
    print(f'Phase 1 trianing on labels: {labels}')

    num_labels = len(labels)
    samples_per_label = args.num_total_samples // num_labels
    print(f'Allocating {samples_per_label} samples per label (Total: {samples_per_label * num_labels})')

    # combine all datasets
    train_datasets = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            num_samples=samples_per_label,
            run_augment=True,
            cumulation_rate=args.cumulation_rate,
            root=args.root,
            mirror_prob=0.5,
            noise_std=0.2,
            random_high=0.2
        )
        train_datasets.append(dataset)
    combined_train_dataset = ConcatDataset(train_datasets)

    validate_datasets = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            num_samples=500,
            run_augment=False,
            cumulation_rate=args.cumulation_rate,
            root=args.root
        )
        validate_datasets.append(dataset)
    combined_validate_dataset = ConcatDataset(validate_datasets)

    batch_size = args.batch_size
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(combined_validate_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMRMemoryModelDualHead(output_dim=3).to(device)  # NOTE: use output dim 3 now
    print("Total model parameters:", sum(p.numel() for p in model.parameters()))
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_losses = []
    train_r2s = []
    val_losses = []
    val_r2s = []

    # Training loop
    for epoch in tqdm(range(1, args.num_epochs + 1), desc='Phase 1 Training epochs'):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_r2 = 0.0
        total_train = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            features = batch['feature'].to(device)      # [B, 6, 8]
            targets = batch['target'].to(device)        # [B, memory_length, 2, output_dim]
            # Use the last timestamp as ground truth
            targets = targets[:, -1, :].unsqueeze(3)    # [B, 2, output_dim, 1]

            optimizer.zero_grad()
            outputs = model(features)                   # [B, 2, output_dim, 1]

            loss = gmr_loss_dualhead(outputs, targets, alpha=1.5)
            loss.backward()
            optimizer.step()

            current_batch_size = features.size(0)
            epoch_train_loss += loss.item() * current_batch_size
            total_train += current_batch_size

            pred_time = outputs[:, 0, :, :]
            pred_presence = outputs[:, 1, :, :]
            pred_presence = torch.where(pred_presence < 0, -1, 1)  # convert prediction class to only -1 and 1
            pred_signed_time = (pred_time * pred_presence).squeeze(2)

            targets_time = targets[:, 0, :, :]
            targets_presence = targets[:, 1, :, :]
            signed_target = (targets_time * targets_presence).squeeze(2)

            train_preds.append(pred_signed_time.detach().cpu().numpy())
            train_targets.append(signed_target.detach().cpu().numpy())

        # epoch train loss
        epoch_train_loss /= total_train

        # epoch train r2
        train_preds = np.concatenate(train_preds, axis=0).flatten()
        train_targets = np.concatenate(train_targets, axis=0).flatten()
        epoch_train_r2 = gmr_r2(train_targets, train_preds)

        train_losses.append(epoch_train_loss)
        train_r2s.append(epoch_train_r2)

        # validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_r2 = 0.0
        total_val = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in validate_loader:
                features = batch['feature'].to(device)
                targets = batch['target'].to(device)
                targets = targets[:, -1, :].unsqueeze(3)

                outputs = model(features)

                loss = gmr_loss_dualhead(outputs, targets, alpha=1.5)
                current_batch_size = features.size(0)
                epoch_val_loss += loss.item() * current_batch_size
                total_val += current_batch_size

                pred_time = outputs[:, 0, :, :]
                pred_presence = outputs[:, 1, :, :]
                pred_presence = torch.where(pred_presence < 0, -1, 1)  # convert prediction class to only -1 and 1
                pred_signed_time = (pred_time * pred_presence).squeeze(2)

                targets_time = targets[:, 0, :, :]
                targets_presence = targets[:, 1, :, :]
                signed_target = (targets_time * targets_presence).squeeze(2)

                val_preds.append(pred_signed_time.detach().cpu().numpy())
                val_targets.append(signed_target.detach().cpu().numpy())
        
        # epoch validate loss
        epoch_val_loss /= total_val
        
        # epoch validate r2
        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_targets = np.concatenate(val_targets, axis=0).flatten()
        epoch_val_r2 = gmr_r2(val_targets, val_preds)

        val_losses.append(epoch_val_loss)
        val_r2s.append(epoch_val_r2)

        # Every 10 epochs (and at the final epoch), print progress using tqdm.write
        if epoch % 1 == 0 or epoch == args.num_epochs:
            tqdm.write(
                f'Phase 1 - Epoch {epoch}/{args.num_epochs}: '
                f'Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, '
                f'Validate Loss: {epoch_val_loss:.4f}, Validate R2: {epoch_val_r2:.4f}'
            )
    torch.save(model.state_dict(), args.phase1_ckpt)
    print(f'Phase 1 complete. Checkpoint saved to {args.phase1_ckpt}')

    train_losses = np.array(train_losses)
    train_r2s = np.array(train_r2s)
    val_losses = np.array(val_losses)
    val_r2s = np.array(val_r2s)
    phase1_report = np.stack((train_losses, train_r2s, val_losses, val_r2s), axis=1)
    np.save('phase1_report', phase1_report)
    print('Report saved to phase1_report.npy')

def train_phase2(args):
    """
    Phase 2: Adaptation.
    Loads the pretrained base model (which outputs 2 channels), wraps it with the Adaptation module,
    and trains only the prompt (and new head) on data that includes triangle (labels contain 't').
    """
    labels = [k for k in KEY_FRAMES_DICT.keys() if 't' in k] 
    print(f'Phase 2 (Adaptation) training on labels: {labels}')

    num_labels = len(labels)
    samples_per_label = args.num_total_samples // num_labels
    print(f'Allocating {samples_per_label} samples per label (Total: {samples_per_label * num_labels})')

    train_datasets = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            num_samples=samples_per_label,
            run_augment=True,
            cumulation_rate=args.cumulation_rate,
            root=args.root
        )
        train_datasets.append(dataset)
    combined_train_dataset = ConcatDataset(train_datasets)

    validate_datasets = []
    for label in labels:
        dataset = GMRMemoryDataset(
            label=label,
            num_samples=100,
            run_augment=False,
            cumulation_rate=args.cumulation_rate,
            root=args.root
        )
        validate_datasets.append(dataset)
    combined_validate_dataset = ConcatDataset(validate_datasets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = GMRMemoryModelDualHead(output_dim=2).to(device)
    base_model.load_state_dict(torch.load(args.phase1_ckpt, map_location=device))

    batch_size = args.batch_size
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(combined_validate_dataset, batch_size=batch_size, shuffle=False)

    model = GMRMemoryModelAdapted(base_model).to(device)
    prompt_params = list(model.parameters())
    print('Total trainable parameters in Adaption model:',
          sum(p.numel() for p in prompt_params if p.requires_grad))
    print(model)

    optimizer = optim.Adam(prompt_params, lr=1e-2, weight_decay=0)

    train_losses = []
    train_r2s = []
    val_losses = []
    val_r2s = []

    for epoch in tqdm(range(1, args.num_epochs + 1), desc='Phase 2 Training epochs'):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_r2 = 0.0
        total_train = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            features = batch['feature'].to(device)
            targets = batch['target'].to(device)
            targets = targets[:, -1, :].unsqueeze(3)    # [B, 2, new_out_dim, 1]

            optimizer.zero_grad()
            outputs = model(features)                   # [B, 2, new_out_dim, 1]
            
            loss = gmr_loss_dualhead(outputs, targets, alpha=0.1)
            loss.backward()
            optimizer.step()

            current_batch_size = features.size(0)
            epoch_train_loss += loss.item() * current_batch_size
            total_train += current_batch_size

            pred_time = outputs[:, 0, :, :]
            pred_presence = outputs[:, 1, :, :]
            pred_presence = torch.where(pred_presence < 0, -1, 1)  # convert prediction class to only -1 and 1
            pred_signed_time = (pred_time * pred_presence).squeeze(2)

            targets_time = targets[:, 0, :, :]
            targets_presence = targets[:, 1, :, :]
            signed_target = (targets_time * targets_presence).squeeze(2)

            train_preds.append(pred_signed_time.detach().cpu().numpy())
            train_targets.append(signed_target.detach().cpu().numpy())

        # epoch train loss
        epoch_train_loss /= total_train
        
        # epoch train r2
        train_preds = np.concatenate(train_preds, axis=0).flatten()
        train_targets = np.concatenate(train_targets, axis=0).flatten()
        epoch_train_r2 = gmr_r2(train_targets, train_preds)

        train_losses.append(epoch_train_loss)
        train_r2s.append(epoch_train_r2)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_r2 = 0.0
        total_val = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in validate_loader:
                features = batch['feature'].to(device)
                targets = batch['target'].to(device)
                targets = targets[:, -1, :].unsqueeze(2)

                outputs = model(features)

                loss = gmr_loss_dualhead(outputs, targets, alpha=5)
                current_batch_size = features.size(0)
                epoch_val_loss += loss.item() * current_batch_size
                total_val += current_batch_size
                
                pred_time = outputs[:, 0, :, :]
                pred_presence = outputs[:, 1, :, :]
                pred_presence = torch.where(pred_presence < 0, -1, 1)  # convert prediction class to only -1 and 1
                pred_signed_time = (pred_time * pred_presence).squeeze(2)

                targets_time = targets[:, 0, :, :]
                targets_presence = targets[:, 1, :, :]
                signed_target = (targets_time * targets_presence).squeeze(2)

                val_preds.append(pred_signed_time.detach().cpu().numpy())
                val_targets.append(signed_target.detach().cpu().numpy())
        
        # epoch validate loss
        epoch_val_loss /= len(combined_validate_dataset)
        
        # epoch validate r2
        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_targets = np.concatenate(val_targets, axis=0).flatten()
        epoch_val_r2 = gmr_r2(val_targets, val_preds)

        val_losses.append(epoch_val_loss)
        val_r2s.append(epoch_val_r2)

        if epoch % 1 == 0 or epoch == args.num_epochs:
            tqdm.write(
                f'Phase 2 - Epoch {epoch}/{args.num_epochs}: '
                f'Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, '
                f'Validate Loss: {epoch_val_loss:.4f}, Validate R2: {epoch_val_r2:.4f}'
            )
    torch.save(model.state_dict(), args.phase2_ckpt)
    print(f'Phase 2 complete. Checkpoint saved to {args.phase2_ckpt}')

    train_losses = np.array(train_losses)
    train_r2s = np.array(train_r2s)
    val_losses = np.array(val_losses)
    val_r2s = np.array(val_r2s)
    phase1_report = np.stack((train_losses, train_r2s, val_losses, val_r2s), axis=1)
    np.save('phase2_report', phase1_report)
    print('Report saved to phase2_report.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model with GMRMemoryDataset')
    parser.add_argument('-e', '--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-n', '--num-total-samples', type=int, default=10000, help='Total number of samples to use across all labels')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('-c', '--cumulation-rate', type=float, default=0.001, help='Cumulation rate for target values')
    parser.add_argument('--root', type=str, default='./dataset/', help='Root directory for raw data files')
    parser.add_argument('--downsample-factor', type=int, default=1, help='Downsample factor')
    parser.add_argument('--memory-length', type=int, default=1, help='Number of timesteps in each target sample')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1, help='Training phase: 1 for CS-only, 2 for p-tuning')
    parser.add_argument('--phase1-ckpt', type=str, default='phase1_base_model.pth', help='Checkpoint path for phase 1')
    parser.add_argument('--phase2-ckpt', type=str, default='phase2_ptuning_model.pth', help='Checkpoint path for phase 2')
    
    args = parser.parse_args()
    
    gmr_set_seeds(114514)

    if args.phase == 1:
        train_phase1(args)
    else:
        train_phase2(args)