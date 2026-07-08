"""
训练脚本
完整的多模态融合网络训练流程
"""

import os
import sys
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.fusion_net import MultiModalFusionNet, MultiModalFusionNetLite
from models.dataset import create_dataloaders, create_dataloaders_from_splits
from data.prepare_data import MultiModalDataGenerator, split_dataset


class Trainer:
    """训练器"""
    
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        
        # 训练参数
        self.num_epochs = config['train']['num_epochs']
        self.learning_rate = config['train']['learning_rate']
        self.weight_decay = config['train']['weight_decay']
        self.warmup_epochs = config['train']['warmup_epochs']
        self.gradient_clip = config['train']['gradient_clip']
        
        # 损失函数
        class_weights = torch.tensor(config['train']['class_weights'], dtype=torch.float32)
        if config['train']['loss'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=config['train']['label_smoothing']
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # 优化器
        if config['train']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.use_amp = config['train']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # 早停
        self.early_stopping_patience = config['train']['early_stopping']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def _create_scheduler(self):
        """创建学习率调度器"""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs - self.warmup_epochs,
            eta_min=self.config['train']['min_lr']
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_epochs]
        )
        
        return scheduler
    
    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        
        for batch in pbar:
            # 移动数据到设备
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            health = batch['health'].to(self.device)
            medication = batch['medication'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    logits = self.model(video, audio, health, medication)
                    loss = self.criterion(logits, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(video, audio, health, medication)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            
            for batch in pbar:
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                health = batch['health'].to(self.device)
                medication = batch['medication'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(video, audio, health, medication)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, save_dir):
        """完整训练流程"""
        print(f"\n开始训练...")
        print(f"  设备: {self.device}")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  验证样本: {len(val_loader.dataset)}")
        print(f"  批次大小: {self.config['train']['batch_size']}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  训练轮数: {self.num_epochs}")
        print()
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印结果
            print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': dict(self.config),
                }, save_path)
                print(f'  保存最佳模型: {save_path}')
            else:
                self.patience_counter += 1
            
            # 定期保存
            if (epoch + 1) % self.config['train']['save_every'] == 0:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                print(f'\n早停: 验证性能 {self.early_stopping_patience} 轮未提升')
                break
        
        print(f'\n训练完成!')
        print(f'最佳验证准确率: {self.best_val_acc:.2f}%')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
        }


def main():
    parser = argparse.ArgumentParser(description='训练多模态融合网络')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'], help='模型类型')
    parser.add_argument('--num_samples', type=int, default=5000, help='生成样本数')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU')
    args = parser.parse_args()
    
    # 加载配置
    config = get_config()
    
    # 覆盖命令行参数
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        config['train']['learning_rate'] = args.lr
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成或加载数据
    data_path = config['data']['processed_data_dir']
    if not os.path.exists(os.path.join(data_path, 'train')):
        print("数据不存在，生成模拟数据...")
        generator = MultiModalDataGenerator(config)
        dataset = generator.generate_dataset(
            num_samples=args.num_samples,
            save_path=data_path
        )
        
        # 划分数据集
        splits = split_dataset(dataset)
        for split_name, split_data in splits.items():
            split_path = os.path.join(data_path, split_name)
            os.makedirs(split_path, exist_ok=True)
            for key, value in split_data.items():
                np.save(os.path.join(split_path, f'{key}.npy'), value)
            print(f"{split_name}集: {len(split_data['labels'])} 个样本")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 创建模型
    model_config = config['model']
    if args.model == 'full':
        model = MultiModalFusionNet(model_config)
    else:
        model = MultiModalFusionNetLite(model_config)
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 创建训练器
    trainer = Trainer(config, model, device)

    # 训练；full 与 lite 分别保存到子目录，避免互相覆盖（消融实验需对比两个 checkpoint）
    save_dir = config['data']['checkpoint_dir']
    if args.model == 'lite':
        save_dir = os.path.join(save_dir, 'lite')
    history = trainer.train(train_loader, val_loader, save_dir)
    
    # 保存训练历史
    import json
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
