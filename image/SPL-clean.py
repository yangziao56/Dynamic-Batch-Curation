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

from dataset import get_dataloaders
from data.datasets import input_dataset
from model import get_model, get_last_fc_input


import argparse

parser = argparse.ArgumentParser('')

#Cifar
parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, or cifar100')
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='aggre')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=640, help='Batch size')
parser.add_argument('--val_batch_size', type=int, default=12000, help='Validation batch size')
parser.add_argument('--test_batch_size', type=int, default=1000, help='Test batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--method', type=str, default='output_grads_dot', help='vanilla, SPL, IP, Ghost, output_grads_dot')
# SPL parameters
parser.add_argument('--v', type=float, default=1, help='SPL weighting parameter')
parser.add_argument('--lambda_', type=float, default=1e+5, help='SPL threshold')

# IP parameters
parser.add_argument('--last_layer', type=bool, default=False, help='Use last layer for SPL')
parser.add_argument('--calculate_val_grad_per_step', type=bool, default=True, help='Calculate val grad per step')

parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='Device (cpu or cuda)')
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

# load dataset
print('Loading dataset...')
train_dataset, val_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args)
# print(f'Train set size: {len(train_dataset)}')
# print(f'Validation set size: {len(val_dataset)}')
# print(f'Test set size: {len(test_dataset)}')
# print(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


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


# Initialize model, loss, and optimizer
criterion = nn.CrossEntropyLoss(reduction='none')  # We need individual losses for SPL
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training parameters
epochs = args.epochs
v = args.v  # SPL weighting parameter
lambda_ = args.lambda_  # SPL threshold
lr = args.lr

# Initialize model, loss, and optimizer
#model = MLP()
criterion = nn.CrossEntropyLoss(reduction='none')  # We need individual losses for SPL
optimizer = optim.Adam(model.parameters(), lr=lr)


if args.last_layer:
    total_params = model.fc.weight.numel() + model.fc.bias.numel()
    # total_params = model.linear.weight.numel() + model.linear.bias.numel()
else:
    total_params = sum(p.numel() for p in model.parameters())
#accumulated_grads_val = torch.zeros(total_params).to(args.device)
# Training loop with SPL
for epoch in range(epochs):
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

    for data, target in train_loader:
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
            for data2, target2 in val_loader:
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
        

        if args.method == 'vanilla':
            output = model(data)
            losses = criterion(output, target)  # Compute individual losses for SPL
            weights = weights = torch.ones_like(torch.randn(data.shape[0]))  # Disable SPL #weights = 1.0  # Disable SPL
            model.zero_grad()
            #print(weights)
            # count how many samples are selected
            sample_count += weights.sum().item()
            #print('sample count:', sample_count)
            weighted_loss = (weights * losses).mean()  # Apply SPL weights to losses
            weighted_loss.backward()
        elif args.method == 'SPL':
            output = model(data)
            losses = criterion(output, target)  # Compute individual losses for SPL
            weights = self_paced_learning(losses, v, lambda_)

            model.zero_grad()
            #print(weights)
            # count how many samples are selected
            sample_count += weights.sum().item()
            #print('sample count:', sample_count)
            weighted_loss = (weights * losses).mean()  # Apply SPL weights to losses
            weighted_loss.backward()
        elif args.method == 'IP':
            from torch.func import functional_call, vmap, grad
            if args.last_layer:
                last_layer_name = list(model.named_parameters())[-1][0].split('.')[0]  # 获取最后一层的名称
                params = {
                    k: v.detach()
                    for k, v in model.named_parameters()
                    if k.startswith(last_layer_name + '.')
                }

            else:
                params = {k: v.detach() for k, v in model.named_parameters()}
            buffers = {k: v.detach() for k, v in model.named_buffers()}
            # 定义一个函数计算梯度和内积
            def compute_loss(params, buffers, image, label):
                # 确保 image 是四维的
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # 添加批次维度
                logits = functional_call(model, (params, buffers), (image,))
                loss = F.cross_entropy(logits, label.unsqueeze(0))
                return loss
            
            model.zero_grad()
            ft_compute_grad = grad(compute_loss)
            ft_compute_sample_grad = vmap(ft_compute_grad,randomness="same", in_dims=(None, None, 0, 0))
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, target)
            # 首先获取批次大小（batch size），假设任一参数的第一维都是批次大小
            batch_size_temp = next(iter(ft_per_sample_grads.values())).shape[0]
            # 为每个样本初始化一个梯度列表
            grads_per_sample = [torch.cat([g[i].flatten() for g in ft_per_sample_grads.values()]) for i in range(batch_size_temp)]
            # 将列表转换为一个Tensor
            total_grads_per_sample = torch.stack(grads_per_sample)# [batch_size, total_number_of_grads]
            # print('total_grads_per_sample:',total_grads_per_sample.shape)
            # print('accumulated_grads_val:',accumulated_grads_val.shape)
            weights = self_paced_learningIP(total_grads_per_sample, accumulated_grads_val)

            sample_count += weights.sum().item()
            output = model(data)
            losses = criterion(output, target)
            #weighted_loss = (weights * losses).mean()
            # 使用布尔索引过滤掉与权重为 0 对应的损失值
            filtered_losses = losses[weights.bool()]

            # 计算加权损失的平均值
            weighted_loss = filtered_losses.mean()
            
            #backpropagate
            weighted_loss.backward()
            '''
            # Step 1: 计算当前批次的平均梯度
            # ft_per_sample_grads 是字典形式，键为参数名称，值为 [batch_size, param_shape] 的张量
            batch_avg_grads = {name: grads.mean(dim=0) for name, grads in ft_per_sample_grads.items()}  # 对每个参数的梯度取平均

            # Step 2: 将平均梯度赋值到 model.parameters()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in batch_avg_grads:
                        param.grad = batch_avg_grads[name]  # 将平均梯度赋值给模型的参数
            '''
        elif args.method == 'Ghost':
            for data2, target2 in val_loader:
                #stack data and target in batch dimension
                data2, target2 = data2.to(args.device), target2.to(args.device)
                data_stack, target_stack = torch.cat((data, data2)), torch.cat((target, target2))

                output = model(data_stack)
                output.retain_grad()
                losses_total = criterion(output, target_stack)
                losses_total.mean().backward(retain_graph=True)
                output_grads = output.grad
                # print('output_grads:',output_grads.shape)
                output_train_grads, output_val_grads = output_grads[:data.shape[0]], output_grads[data.shape[0]:]
                output_train, output_val = output[:data.shape[0]], output[data.shape[0]:]
                last_fc_input = get_last_fc_input()

                last_fc_input_train, last_fc_input_val = last_fc_input[:data.shape[0]], last_fc_input[data.shape[0]:]
                #multiply the val grads with the train grads
                grads_dot = torch.mm(output_train_grads, output_val_grads.T)
                output_dot = torch.mm(output_train, output_val.T)
                print(output_train.shape)
                last_fc_input_dot = torch.mm(last_fc_input_train, last_fc_input_val.T)
                # print('grads_dot:',grads_dot.shape)
                # print('output_dot:',output_dot.shape)
                ghost = grads_dot * last_fc_input_dot
                #ghost = grads_dot * output_dot
                #计算output dot中的负数个数
                # print(output_dot.shape)
                # print(grads_dot.shape)
                print('negative percentage:',(last_fc_input_dot<0).sum().item()/output_dot.numel())
                #ghost = output_dot
                # avrage the ghost in dim 1
                ghost = ghost.mean(dim=1)
                print('ghost:',ghost.shape)
                
                from scipy.stats import kendalltau
                from scipy.stats import spearmanr
                
                print('kendalltau:',kendalltau(ghost.cpu().numpy(), grads_dot.mean(dim=1).cpu().numpy()))
                print('spearmanr:',spearmanr(ghost.cpu().numpy(), grads_dot.mean(dim=1).cpu().numpy()))

                # weights = torch.where(ghost >= 0, torch.ones_like(ghost), torch.zeros_like(ghost))
                weights = ghost >= 0
                sample_count += weights.sum().item()
                losses = losses_total[:data.shape[0]]
                model.zero_grad()
                filtered_losses = losses[weights.bool()]

                # 计算加权损失的平均值
                weighted_loss = filtered_losses.mean()
                
                #backpropagate
                weighted_loss.backward()
        elif args.method == 'output_grads_dot':
            for data2, target2 in val_loader:
                #stack data and target in batch dimension
                data2, target2 = data2.to(args.device), target2.to(args.device)
                data_stack, target_stack = torch.cat((data, data2)), torch.cat((target, target2))

                output = model(data_stack)
                output.retain_grad()
                losses_total = criterion(output, target_stack)
                losses_total.mean().backward(retain_graph=True)
                output_grads = output.grad
                # print('output_grads:',output_grads.shape)
                output_train_grads, output_val_grads = output_grads[:data.shape[0]], output_grads[data.shape[0]:]
                
                #multiply the val grads with the train grads
                mean_val_grads = output_val_grads.mean(dim=0)      # 形状: [D]
                # 计算每个训练梯度与验证梯度均值的点积
                ghost = torch.matmul(output_train_grads, mean_val_grads)  # 形状: [N_train]

                weights = ghost >= 0
                print('negative percentage:',(ghost<0).sum().item()/ghost.numel())
                sample_count += weights.sum().item()
                losses = losses_total[:data.shape[0]]
                model.zero_grad()
                filtered_losses = losses[weights.bool()]

                # 计算加权损失的平均值
                weighted_loss = filtered_losses.mean()
                
                #backpropagate
                weighted_loss.backward()
                




            

        optimizer.step()

        total_loss += weighted_loss.item()
    # Validation
    model.eval()
    correct = 0
    accumulated_grads_val = torch.zeros(total_params).to(args.device)
    #with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        model.zero_grad()
        losses = criterion(output, target).mean()
        losses.backward()
        # 累加本批次的梯度
        if args.last_layer:
            grads = torch.cat([model.fc.weight.grad.flatten(), model.fc.bias.grad.flatten()])
        else:
            grads = torch.cat([param.grad.flatten() for param in model.parameters()])
        accumulated_grads_val += grads
    # 计算平均梯度
    accumulated_grads_val /= len(val_loader.dataset)

    accuracy = correct / len(val_loader.dataset)
    if epoch == 0:
        best_acc = accuracy
        best_model = model
        best_epoch = epoch
    else:
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_epoch = epoch

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Validation Accuracy: {accuracy * 100:.2f}%')
    print(f'Selected samples: {sample_count}/{len(train_loader.dataset)}')

# Testing loop
best_model = model
best_model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Bset Epoch: {best_epoch + 1}')
