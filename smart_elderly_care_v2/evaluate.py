"""
评估脚本
评估训练好的多模态融合模型
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.fusion_net import MultiModalFusionNet, MultiModalFusionNetLite
from models.dataset import create_dataloaders


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, device, class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names or ['低风险', '中风险', '高风险']
        
    def evaluate(self, test_loader):
        """
        评估模型
        
        Returns:
            dict: 包含各项评估指标的字典
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='评估中'):
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                health = batch['health'].to(self.device)
                medication = batch['medication'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(video, audio, health, medication)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision_macro': precision_score(all_labels, all_preds, average='macro'),
            'recall_macro': recall_score(all_labels, all_preds, average='macro'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted'),
            'recall_weighted': recall_score(all_labels, all_preds, average='weighted'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        }
        
        # 计算每个类别的指标
        per_class_metrics = {}
        for i, name in enumerate(self.class_names):
            per_class_metrics[name] = {
                'precision': precision_score(all_labels, all_preds, labels=[i], average='micro'),
                'recall': recall_score(all_labels, all_preds, labels=[i], average='micro'),
                'f1': f1_score(all_labels, all_preds, labels=[i], average='micro'),
            }
        metrics['per_class'] = per_class_metrics
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        # AUC（多分类）
        try:
            metrics['auc_macro'] = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
        except:
            metrics['auc_macro'] = None
        
        # 预测结果
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['probabilities'] = all_probs
        
        return metrics
    
    def print_report(self, metrics):
        """打印评估报告"""
        print("\n" + "="*60)
        print("模型评估报告")
        print("="*60)
        
        print(f"\n整体指标:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision-Macro): {metrics['precision_macro']:.4f}")
        print(f"  召回率 (Recall-Macro): {metrics['recall_macro']:.4f}")
        print(f"  F1分数 (F1-Macro): {metrics['f1_macro']:.4f}")
        if metrics['auc_macro']:
            print(f"  AUC (Macro): {metrics['auc_macro']:.4f}")
        
        print(f"\n各类别指标:")
        for name, class_metrics in metrics['per_class'].items():
            print(f"  {name}:")
            print(f"    精确率: {class_metrics['precision']:.4f}")
            print(f"    召回率: {class_metrics['recall']:.4f}")
            print(f"    F1分数: {class_metrics['f1']:.4f}")
        
        print(f"\n混淆矩阵:")
        print(metrics['confusion_matrix'])
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """绘制混淆矩阵"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.close()
    
    def plot_metrics(self, metrics, save_path=None):
        """绘制各类别指标对比图"""
        classes = list(metrics['per_class'].keys())
        precision = [metrics['per_class'][c]['precision'] for c in classes]
        recall = [metrics['per_class'][c]['recall'] for c in classes]
        f1 = [metrics['per_class'][c]['f1'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='精确率')
        bars2 = ax.bar(x, recall, width, label='召回率')
        bars3 = ax.bar(x + width, f1, width, label='F1分数')
        
        ax.set_ylabel('分数')
        ax.set_title('各类别评估指标')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"指标对比图已保存: {save_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='评估多模态融合模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='输出目录')
    args = parser.parse_args()
    
    # 配置
    config = get_config()
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model_config = config['model']
    if args.model == 'full':
        model = MultiModalFusionNet(model_config)
    else:
        model = MultiModalFusionNetLite(model_config)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"加载模型: {args.checkpoint}")
    print(f"模型验证准确率: {checkpoint.get('val_acc', 'N/A')}")
    
    # 创建数据加载器
    _, _, test_loader = create_dataloaders(config)
    
    # 评估
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    # 打印报告
    evaluator.print_report(metrics)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制图表
    evaluator.plot_confusion_matrix(
        metrics, 
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    evaluator.plot_metrics(
        metrics,
        save_path=os.path.join(args.output_dir, 'per_class_metrics.png')
    )
    
    # 保存指标到文件
    import json
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k not in ['predictions', 'labels', 'probabilities', 'confusion_matrix']}
    metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
