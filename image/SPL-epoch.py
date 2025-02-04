import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import random
import os
import numpy as np
import time

from dataset import get_dataloaders
from data.datasets import input_dataset
from model import get_model, get_last_fc_input
from torch.utils.data import DataLoader, Subset


import argparse

parser = argparse.ArgumentParser('')

#Cifar
parser.add_argument('--dataset', type=str, default='cifar100', help='mnist, cifar10, or cifar100')
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='noisy100')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)

parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--val_batch_size', type=int, default=4000, help='Validation batch size')
parser.add_argument('--test_batch_size', type=int, default=1280, help='Test batch size')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--method', type=str, default='Ours', help='vanilla, SPL, IP, Ghost, output_grads_dot, self_output_grads_dot')
# SPL parameters
parser.add_argument('--v', type=float, default=1, help='SPL weighting parameter')
parser.add_argument('--lambda_', type=float, default=5, help='SPL threshold')

# IP parameters
parser.add_argument('--last_layer', type=bool, default=False, help='Use last layer for SPL')
parser.add_argument('--calculate_val_grad_per_step', type=bool, default=True, help='Calculate val grad per step')

parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='Device (cpu or cuda)')

#sigma
parser.add_argument('--sigma', type=float, default=1, help='sigma for gaussian noise')
args = parser.parse_args()

def set_seed(seed):
    # 固定 Python 的随机数生成器
    random.seed(seed)

    # 固定 NumPy 的随机数生成器
    np.random.seed(seed)

    # 固定 PyTorch 的随机数生成器（CPU 和 GPU）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

    # 确保 cuDNN 的操作是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 确保环境中所有使用随机性的模块都受控
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设定一个种子，比如 42
set_seed(args.seed)

'''
if args.dataset == 'mnist':
    # Load MNIST data
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # 加载训练集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # 划分训练集和验证集（90% 训练集, 10% 验证集）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.val_batch_size, shuffle=False)

    # 加载测试集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # 打印数据集大小
    print('Loading MNIST dataset...')
    print(f'Train set size: {len(train_subset)}')
    print(f'Validation set size: {len(val_subset)}')
    print(f'Test set size: {len(test_dataset)}')

    # Define MLP model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc(x)
            return x
        
    def save_fc_input(module, input, output):
        global last_fc_input
        last_fc_input = input[0].detach()  # 保存输入，input是一个元组
    model = MLP().to(args.device)
    handle = model.fc.register_forward_hook(save_fc_input)

elif args.dataset == 'cifar10':
    # Load CIFAR-10 data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

    # Split training set into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 打印数据集大小
    print('Loading CIFAR-10 dataset...')
    print(f'Train set size: {len(train_subset)}')
    print(f'Validation set size: {len(val_subset)}')
    print(f'Test set size: {len(test_dataset)}')

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 第一层卷积：输入为3通道 (RGB图像)，输出为32个特征图，卷积核大小为3x3
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            # 第二层卷积：输入为32个通道，输出为64个特征图，卷积核大小为3x3
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            # 第三层卷积：输入为64个通道，输出为128个特征图，卷积核大小为3x3
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            
            # 池化层：将特征图尺寸缩小一半
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
            # 全连接层
            self.fc1 = nn.Linear(128 * 4 * 4, 256)  # CIFAR-10 图像经过池化后变为 4x4 大小
            self.fc2 = nn.Linear(256, 10)  # CIFAR-10 有10个类别
        
        def forward(self, x):
            # 卷积层1 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv1(x)))
            # 卷积层2 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv2(x)))
            # 卷积层3 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv3(x)))
            
            # 展平
            x = x.view(-1, 128 * 4 * 4)
            
            # 全连接层1 -> 激活函数ReLU
            x = F.relu(self.fc1(x))
            # 全连接层2 (输出层)
            x = self.fc2(x)
        
            return x
    
    model = CNN().to(args.device)
elif args.dataset == 'cifar100':
    # Load CIFAR-100 data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)

    # Split training set into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # 打印数据集大小
    print('Loading CIFAR-100 dataset...')
    print(f'Train set size: {len(train_subset)}')
    print(f'Validation set size: {len(val_subset)}')
    print(f'Test set size: {len(test_dataset)}')

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # 第一层卷积：输入为3通道 (RGB图像)，输出为32个特征图，卷积核大小为3x3
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            # 第二层卷积：输入为32个通道，输出为64个特征图，卷积核大小为3x3
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            # 第三层卷积：输入为64个通道，输出为128个特征图，卷积核大小为3x3
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            
            # 池化层：将特征图尺寸缩小一半
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
            # 全连接层
            self.fc1 = nn.Linear(128 * 4 * 4, 256)  # CIFAR-100 图像经过池化后变为 4x4 大小
            self.fc = nn.Linear(256, 100)  # CIFAR-100 有10个类别
        
        def forward(self, x):
            # 卷积层1 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv1(x)))
            # 卷积层2 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv2(x)))
            # 卷积层3 -> 激活函数ReLU -> 池化
            x = self.pool(F.relu(self.conv3(x)))
            
            # 展平
            x = x.view(-1, 128 * 4 * 4)
            
            # 全连接层1 -> 激活函数ReLU
            x = F.relu(self.fc1(x))
            # 全连接层2 (输出层)
            x = self.fc(x)
        
            return x
        
    # def save_fc_input(module, input, output):
    #     global last_fc_input
    #     last_fc_input = input[0].detach()  # 保存输入，input是一个元组    
    # model = CNN().to(args.device)
    # handle = model.fc.register_forward_hook(save_fc_input)



    #ResNet
    model = models.resnet18(pretrained=False, num_classes=100).to(args.device)
    # def disable_batchnorm_tracking(module):
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.track_running_stats = False
    #         module.running_mean = None
    #         module.running_var = None

    # 导入预训练的 ResNet-18 模型
    weights = ResNet18_Weights.DEFAULT
    resnet18 = models.resnet18(weights=weights)

    # # 禁用 BatchNorm 的统计跟踪
    # resnet18.apply(disable_batchnorm_tracking)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 100)  # 设定为100类
    model = resnet18.to(args.device)
'''


# noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
# args.noise_type = noise_type_map[args.noise_type]
# # load dataset
# if args.noise_path is None:
#     if args.dataset == 'cifar10':
#         args.noise_path = './data/CIFAR-10_human.pt'
#     elif args.dataset == 'cifar100':
#         args.noise_path = './data/CIFAR-100_human.pt'
#     else: 
#         raise NameError(f'Undefined dataset {args.dataset}')


# train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)

# # 主函数中的相关部分
test_acc = []
for round in range(1):

    # load dataset
    print('Loading dataset...')
    train_dataset, val_dataset, test_dataset, num_classes, num_training_samples, clean_train_subset, noisy_indices = input_dataset(args)
    print(noisy_indices)
    #save noisy_indices with dataset, noise_type
    np.save(f'{args.dataset}_{args.noise_type}_noisy_indices.npy',noisy_indices)
    print(f'clean_train_subset size: {len(clean_train_subset)}')
    # print(f'Train set size: {len(train_dataset)}')
    # print(f'Validation set size: {len(val_dataset)}')
    # print(f'Test set size: {len(test_dataset)}')
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader_no_shuffle = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    #train_loader = DataLoader(clean_train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)


    # 之后可以根据需要使用 train_dataset, val_dataset, test_dataset

    # 假设 args 包含 dataset 和 batch_size 等信息
    #train_loader, val_loader, test_loader = get_dataloaders(args)


    model = get_model(args)
        


    # vanilla SPL function
    def self_paced_learning(losses, v, lambda_):
        """
        根据损失值判断哪些样本参与训练。
        
        参数:
        - losses: 当前batch中每个样本的损失值 (torch.Tensor)
        - lambda_: 自定阈值，低于此阈值的样本将被选中参与训练
        
        返回:
        - weights: 样本选择权重，1表示选中样本，0表示不选中
        """
        # torch.where 判断损失是否小于阈值 lambda_
        weights = torch.where(losses < lambda_, torch.ones_like(losses), torch.zeros_like(losses))
        # print(losses)
        # print(weights)
        # print(lambda_)
        return weights

    # SPL (Self-Paced Learning) function
    def self_paced_learning2(losses, v, lambda_):
        weights = torch.where(losses < lambda_, torch.ones_like(losses), v * (lambda_ / (losses + 1e-10)))
        return weights

    # Inner product self-paced learning
    def self_paced_learningIP(total_grads_per_sample, accumulated_grads_val):
        #IP = torch.dot(total_grads_per_sample, accumulated_grads_val)
        IP = torch.matmul(total_grads_per_sample, accumulated_grads_val)
        # print('IP shape:',IP.shape)
        # print('IP:',IP)
        #weights = torch.where(IP >= 0, torch.ones_like(losses), torch.zeros_like(losses))
        weights = torch.where(IP >= 0, torch.ones_like(IP), torch.zeros_like(IP))
        #weights_negative = torch.where(IP < 0, torch.ones_like(losses), torch.zeros_like(losses))
        # print('weights:',(weights.sum().item()))
        # print('weights_negative:',(weights_negative.sum().item()))
        # print('sum:',(weights.sum().item()+ weights_negative.sum().item()))
        #print('shape of losses:',losses.shape)
        # print('shape of weights:',weights.shape)
        # print('weights:',weights)
        return weights




    # Training parameters
    epochs = args.epochs
    v = args.v  # SPL weighting parameter
    lambda_ = args.lambda_  # SPL threshold
    lr = args.lr

    # Initialize model, loss, and optimizer
    #model = MLP()
    criterion = nn.CrossEntropyLoss(reduction='none')  # We need individual losses for SPL
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    # 定义学习率调度器，每 10 个 epoch 将学习率乘以 0.1
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    if args.last_layer:
        total_params = model.fc.weight.numel() + model.fc.bias.numel()
        # total_params = model.linear.weight.numel() + model.linear.bias.numel()
    else:
        total_params = sum(p.numel() for p in model.parameters())
    #accumulated_grads_val = torch.zeros(total_params).to(args.device)
    # Training loop with SPL

# test_acc = []
# for _ in range(5):
    #score_matrix = torch.zeros((50000, epochs))#.to(args.device)
    score_matrix = np.zeros((50000, epochs))
    print('score_matrix:',score_matrix.shape)
    score_matrix_input_dot_sum = np.zeros((50000, 500))
    score_matrix_grads_dot = np.zeros((50000, 500))

    for epoch in range(epochs):
        scheduler.step(epoch)
        time_start = time.time()
        model.train()
        total_loss = 0
        sample_count = 0
        
        if not args.calculate_val_grad_per_step and args.method == 'IP':
            #caculate the accumulated_grads_val
            accumulated_grads_val = torch.zeros(total_params).to(args.device)
            #with torch.no_grad():  
            model.zero_grad()
            for data2, target2 in val_loader:
                data2, target2 = data2.to(args.device), target2.to(args.device)
                output2 = model(data2)
                pred2 = output2.argmax(dim=1, keepdim=True)
                model.zero_grad()
                losses2 = criterion(output2, target2).mean()
                losses2.backward()
                # 累加本批次的梯度
                if args.last_layer:
                    grads = torch.cat([model.linear.weight.grad.flatten(), model.linear.bias.grad.flatten()])
                else:
                    grads = torch.cat([param.grad.flatten() for param in model.parameters()])
                accumulated_grads_val += grads
            # 计算平均梯度
            accumulated_grads_val /= len(val_loader.dataset)

        for data, target, idx in train_loader:
            #print('idx:',idx)
            if args.method == 'IP' and args.calculate_val_grad_per_step:  #caculate the accumulated_grads_val
                # print('calculate_val_grad_per_step')
                # print(args.calculate_val_grad_per_step)
                accumulated_grads_val = torch.zeros(total_params).to(args.device)
                if args.last_layer:
                    for name, param in model.named_parameters():
                        if 'fc' not in name:  # 这里假设最后一层的名称是'linear'
                            param.requires_grad = False
                #with torch.no_grad():  
                model.zero_grad()
                for data2, target2, _ in val_loader:
                    data2, target2 = data2.to(args.device), target2.to(args.device)
                    output2 = model(data2)
                    pred2 = output2.argmax(dim=1, keepdim=True)
                    model.zero_grad()
                    losses2 = criterion(output2, target2).mean()
                    # print('losses2:',losses2.shape)
                    # print('without mean:',criterion(output2, target2).shape)
                    # print('output2:',output2.shape)
                    # print('target2:',target2.shape)
                    losses2.backward()
                    # 累加本批次的梯度
                    if args.last_layer:
                        grads = torch.cat([model.fc.weight.grad.flatten(), model.fc.bias.grad.flatten()])
                    else:
                        grads = torch.cat([param.grad.flatten() for param in model.parameters()])
                    accumulated_grads_val += grads
                # 计算平均梯度
                accumulated_grads_val /= len(val_loader.dataset)
                if args.last_layer:
                    for name, param in model.named_parameters():
                        param.requires_grad = True  # 解冻所有参数
            #train
            data, target = data.to(args.device), target.to(args.device)
            model.train()
            optimizer.zero_grad()
            
            output = model(data)
            losses = criterion(output, target)  # Compute individual losses for SPL
            weights = weights = torch.ones_like(torch.randn(data.shape[0])).to(args.device)  # Disable SPL #weights = 1.0  # Disable SPL
            model.zero_grad()
            #print(weights)
            # count how many samples are selected
            sample_count += weights.sum().item()
            #print('sample count:', sample_count)
            weighted_loss = (weights * losses).mean()  # Apply SPL weights to losses
            weighted_loss.backward()

            optimizer.step()
        

            total_loss += weighted_loss.item()


        if epoch == 9 or epoch == 29 or epoch == 69 or epoch == 149:
            if args.method == 'Ours':
                ghost_total = 0  # 用于累加每一层的 ghost 结果
                layer_outputs = {}  # 用于存储每一层的输出
                # 定义 hook 函数来获取每一层的输出，并保留梯度
                def forward_hook(module, input, output):
                    #output.retain_grad()  # 保留中间层输出的梯度
                    layer_outputs[module] = (input[0], output)  # 保存输入和输出

                # 为每一层注册 hook
                hooks = []
                for name, layer in model.named_modules():
                    hooks.append(layer.register_forward_hook(forward_hook))

                for data, target, idx in train_loader_no_shuffle:
                    data, target = data.to(args.device), target.to(args.device)
                    model.eval()
                    output = model(data)
                    output.retain_grad()
                    losses = criterion(output, target).mean()
                    output_grads = torch.autograd.grad(outputs=losses.mean(), inputs=output, retain_graph=True)[0]

                    # 计算每一层的 ghost
                    input_dot_sum = 0
                    for layer, (layer_input, layer_output) in layer_outputs.items():
                        if layer_output.grad is None:
                            continue  # 跳过没有梯度的层

                        # # 获取训练和验证部分的输出梯度
                        # output_train_grads = layer_output.grad[:data.shape[0]]
                        # output_val_grads = layer_output.grad[data.shape[0]:]

                        # # 将梯度展平
                        # output_train_grads = output_train_grads.view(data.shape[0], -1)
                        # output_val_grads = output_val_grads.view(data2.shape[0], -1)

                        # 获取训练和验证部分的输入
                        input_train = layer_input[:data.shape[0]].view(data.shape[0], -1)
                        input_val = layer_input[data.shape[0]:].view(data2.shape[0], -1)

                        # 计算点积并累加
                        #grads_dot = torch.mm(output_train_grads, output_val_grads.T)
                        input_dot = torch.mm(input_train, input_val.T)
                        input_dot_sum += input_dot
                        #ghost_total += (grads_dot * input_dot).mean(dim=1)
                    # 将梯度展平
                    output_train_grads = output_train_grads.view(data.shape[0], -1)
                    output_val_grads = output_val_grads.view(data2.shape[0], -1)
                    grads_dot = torch.mm(output_train_grads, output_val_grads.T)
                    ghost_total = (grads_dot * input_dot_sum).mean(dim=1)
                    score_matrix[idx, epoch] = ghost_total.cpu().detach().cpu().numpy()
                    score_matrix_input_dot_sum[idx, :] = input_dot_sum.cpu().detach().cpu().numpy()
                    score_matrix_grads_dot[idx, :] = grads_dot.cpu().detach().cpu().numpy()
                    

                    
                    
                

            
        
        time_end = time.time()
        print('time cost for training 1 epoch', time_end - time_start, 's')
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Validation Accuracy: {accuracy * 100:.2f}%, learning rate: {scheduler.get_last_lr()[0]}')
        print(f'Selected samples: {sample_count}/{len(train_loader.dataset)}')
        

    # Testing loop
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    test_acc.append(accuracy)

    # save score matrix
    if args.method == 'Ours':
        #score_matrix = score_matrix.detach().cpu().numpy()
        np.save(f'score_matrix_{args.dataset}_{args.noise_type}_{round}.npy', score_matrix)
        print('score matrix saved!')
        np.save(f'score_matrix_input_dot_sum_{args.dataset}_{args.noise_type}_{round}.npy', score_matrix_input_dot_sum)
        print('score matrix input dot sum saved!')
        np.save(f'score_matrix_grads_dot_{args.dataset}_{args.noise_type}_{round}.npy', score_matrix_grads_dot)
        print('score matrix grads dot saved!')
    
avg_test_acc = sum(test_acc) / len(test_acc)
std_test_acc = np.std(test_acc)
print(f'Method: {args.method}')
print(f'dataset: {args.dataset}')
#print(f'Average test accuracy: {avg_test_acc * 100:.2f}%')
print(f'Average test accuracy: {avg_test_acc * 100:.2f}%, std: {std_test_acc * 100:.2f}%')
#save method dataset and average test accuracy
with open('result.txt', 'a') as f:
    #f.write(f'Method: {args.method}, dataset: {args.dataset}, Average test accuracy: {avg_test_acc * 100:.2f}%\n')
    f.write(f'Method: {args.method}, dataset: {args.dataset}, noise: {args.noise_type}, Average test accuracy: {avg_test_acc * 100:.2f}%, std: {std_test_acc * 100:.2f}%\n')



