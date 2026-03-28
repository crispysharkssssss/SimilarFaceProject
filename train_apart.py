import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

# 引入你的模型
from models.mobilefacenet import MobileFaceNet 
from models.metrics import ArcFace

def get_args():
    parser = argparse.ArgumentParser(description='Train ArcFace with MobileFaceNet')
    parser.add_argument('--data_dir', type=str, default='data/casia_aligned', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='weights', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Total epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--use_cbam', type=bool, default=False, help='Turn off CBAM')
    
    # 断点续训参数
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (e.g., weights/checkpoint_latest.pth)')
    
    return parser.parse_args()

def save_checkpoint(state, save_dir, filename='checkpoint_latest.pth'):
    """保存完整的训练状态"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"💾 Checkpoint saved: {path}")

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    print(f"Training on {device} | CBAM: {args.use_cbam}")

    # 2. 数据加载
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        
        # 颜色抖动 (亮度、对比度、饱和度)
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载数据集
    try:
        train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        num_classes = len(train_dataset.classes)
        print(f"Images: {len(train_dataset)} | Classes: {num_classes}")
    except:
        print(f"Cannot find data in {args.data_dir}")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )

    # 3. 初始化模型与组件
    backbone = MobileFaceNet(embedding_size=512, use_cbam=args.use_cbam).to(device)
    # header = ArcFace(in_features=512, out_features=num_classes).to(device)
    header = ArcFace(in_features=512, out_features=num_classes, s=64.0, m=0.3).to(device)
    
    optimizer = optim.SGD([
        {'params': backbone.parameters()},
        {'params': header.parameters()}
    ], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 18, 22], gamma=0.1)
    criterion = LabelSmoothingCrossEntropy(eps=0.1)
    # scaler = torch.amp.GradScaler('cuda') # AMP

    # 4. 断点续训逻辑
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # 恢复权重
        backbone.load_state_dict(checkpoint['backbone'])
        header.load_state_dict(checkpoint['header'])
        
        # 恢复优化器状态 (这很重要，否则学习率和动量会重置)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # scaler.load_state_dict(checkpoint['scaler'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr  # 使用你命令行设定的新LR
        print(f"Force update Learning Rate to: {args.lr}")
        
        # 恢复 Epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Jumping to Epoch {start_epoch + 1}")
    else:
        if args.resume:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")

    # 5. 训练循环
    try:
        for epoch in range(start_epoch, args.epochs):
            backbone.train()
            header.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", mininterval=5.0, ncols=100)
            
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                if labels.max() >= num_classes or labels.min() < 0:
                    print(f"发现非法标签: {labels.max().item()} (最大允许: {num_classes-1})")
                    exit()
                    
                optimizer.zero_grad()
                
                # with torch.amp.autocast('cuda'):
                features = backbone(images)
                outputs = header(features, labels)
                loss = criterion(outputs, labels)
                
                # scaler.scale(loss).backward()
                loss.backward()
                # scaler.step(optimizer)
                optimizer.step()
                # scaler.update()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'Loss': loss.item(), 'Acc': 100.*correct/total})
            
            # Epoch 结束处理
            scheduler.step()
            end_time = time.time()
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            print(f"Epoch {epoch+1} Report: Loss={epoch_loss:.4f} | Acc={epoch_acc:.2f}% | Time={end_time-start_time:.1f}s")
            
            # 保存 checkpoint (包含所有恢复所需信息)
            ckpt_state = {
                'epoch': epoch,
                'backbone': backbone.state_dict(),
                'header': header.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                # 'scaler': scaler.state_dict()
            }
            # 保存两份：一份是最新的断点，一份是当前Epoch的归档
            save_checkpoint(ckpt_state, args.save_dir, 'checkpoint_latest_noCBAM.pth')
            # 也可以每隔几轮存一个固定的
            # if (epoch+1) % 5 == 0:
                # save_checkpoint(ckpt_state, args.save_dir, f'mobilefacenet_cbam_epoch_{epoch+1}.pth')

    except KeyboardInterrupt:
        print("\nDetect Ctrl+C! Saving checkpoint before exit...")
        ckpt_state = {
            'epoch': epoch, # 保存当前正在跑的 epoch (下次从这个epoch开始重新跑一遍)
            'backbone': backbone.state_dict(),
            'header': header.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'scaler': scaler.state_dict()
        }
        save_checkpoint(ckpt_state, args.save_dir, 'checkpoint_interrupted.pth')
        print("Saved to weights/checkpoint_interrupted.pth")
        print("Goodbye!")

if __name__ == "__main__":
    main()