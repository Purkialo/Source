import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.resnet_ssd import create_resnet_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import resnet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--base_net_lr', default=None, type=float, help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float, help='initial learning rate for the layers not in base net and prediction heads.')
# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float, help='T_max value for Cosine Annealing Scheduler.')
# Params for loading pretrained basenet.
parser.add_argument('--base_net', help='Pretrained base model')
# Train params
parser.add_argument('--batch_size', default=21, type=int, help='Batch size for training')
parser.add_argument('--num_epochs', default=150, type=int, help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int, help='the number epochs')
parser.add_argument('--debug_steps', default=10, type=int, help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='models/', help='Directory for saving checkpoint models')

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def train(loader, net, criterion, optimizer, device, debug_steps=10, epoch=-1):
    #设置网络训练标签
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        #反向传播置为零
        optimizer.zero_grad()
        #计算网络得出的目标和位置
        confidence, locations = net(images)
        #计算单步loss
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        #叠加分类损失和位置损失
        loss = regression_loss + classification_loss
        #反向传播
        loss.backward()
        #进行SGD计算
        optimizer.step()
        #统计本epoch训练总损失,总分类损失和总位置损失
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        #输出debug信息
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    #设置网络评价标签
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        #不修改梯度计算各参数
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    timer = Timer()
    creat_net = create_resnet_ssd
    config = resnet_ssd_config
    #数据预处理：归至[-1,1]
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    #预加载训练集
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    #预加载测试集
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform, target_transform=target_transform, is_test=True)
    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    #加载网络
    logging.info("Build network. num_classes:{}".format(num_classes))
    net = creat_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]
    #加载预训练参数
    timer.start("Load Model")
    logging.info(f"Init from base net {args.base_net}")
    net.init_from_base_net(args.base_net)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    #数据移至GPU
    net.to(DEVICE)
    logging.info(f"Net has been built and moved to {DEVICE}")
    #设定损失函数和SGD，并设定参数优化方法
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, " + f"Extra Layers learning rate: {extra_layers_lr}.")
    logging.info("Uses CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    #开始训练
    logging.info(f"Start training from epoch {last_epoch + 1}.")

    for epoch in range(last_epoch + 1, args.num_epochs):
        #优化参数
        scheduler.step()
        #一次epoch训练
        train(train_loader, net, criterion, optimizer, device=DEVICE, epoch=epoch)
        #输出debug信息
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"resnet_ssd-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")