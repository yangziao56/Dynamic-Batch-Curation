import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

# 定义全局变量或类来存储钩子捕获的输入
last_fc_input = None

def get_last_fc_input():
    return last_fc_input

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

class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x

def get_model(args):
    global last_fc_input  # 声明使用全局变量

    if args.dataset == 'mnist':
        #return MLP().to(args.device)
        #return CNN_CIFAR(num_classes=10).to(args.device)
        model = MLP().to(args.device)
        
    

        # 定义钩子函数，用于捕获 fc 层的输入
        def hook_fn(module, input, output):
            global last_fc_input
            last_fc_input = input[0].detach().clone()  # 保存输入并进行 detach 和 clone 操作以避免梯度追踪

        # 给 model.fc 注册前向钩子
        hook = model.fc.register_forward_hook(hook_fn)

        # 如果需要在其他地方移除钩子，可以返回 hook 或在适当的时候调用 hook.remove()
        # 这里假设钩子在模型生命周期内一直有效

        return model
    elif args.dataset == 'cifar10':
        #return CNN_CIFAR(num_classes=10).to(args.device)
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
        
        # 修改最后一层的全连接层，使其适应CIFAR-100
        model.fc = torch.nn.Linear(model.fc.in_features, 10).to(args.device)

        # 定义钩子函数，用于捕获 fc 层的输入
        def hook_fn(module, input, output):
            global last_fc_input
            last_fc_input = input[0].detach().clone()  # 保存输入并进行 detach 和 clone 操作以避免梯度追踪

        # 给 model.fc 注册前向钩子
        hook = model.fc.register_forward_hook(hook_fn)

        # 如果需要在其他地方移除钩子，可以返回 hook 或在适当的时候调用 hook.remove()
        # 这里假设钩子在模型生命周期内一直有效

        return model
    elif args.dataset == 'cifar100':
        #model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
        #model = models.resnet152(weights='IMAGENET1K_V1').to(args.device)
        #不加载预训练权重
        model = models.resnet18(pretrained=False).to(args.device)
        model = models.resnet152(pretrained=False).to(args.device)

        
        # 修改最后一层的全连接层，使其适应CIFAR-100
        model.fc = torch.nn.Linear(model.fc.in_features, 100).to(args.device)

        # 定义钩子函数，用于捕获 fc 层的输入
        def hook_fn(module, input, output):
            global last_fc_input
            last_fc_input = input[0].detach().clone()  # 保存输入并进行 detach 和 clone 操作以避免梯度追踪

        # 给 model.fc 注册前向钩子
        hook = model.fc.register_forward_hook(hook_fn)

        # 如果需要在其他地方移除钩子，可以返回 hook 或在适当的时候调用 hook.remove()
        # 这里假设钩子在模型生命周期内一直有效

        return model
    else:
        raise ValueError(f"Model for dataset {args.dataset} is not defined.")
