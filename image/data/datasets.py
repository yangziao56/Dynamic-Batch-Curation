import numpy as np 
import torchvision.transforms as transforms

from .cifar import CIFAR10, CIFAR100
from torch.utils.data import random_split, Subset
import torch
import os


# 定义噪声类型映射
noise_type_map = {
    'clean': 'clean_label',
    'worst': 'worse_label',
    'aggre': 'aggre_label',
    'rand1': 'random_label1',
    'rand2': 'random_label2',
    'rand3': 'random_label3',
    'clean100': 'clean_label',
    'noisy100': 'noisy_label'
}

'''
# 定义数据增强和预处理
train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
'''

# CIFAR-10 数据增强和预处理
train_cifar10_transform = transforms.Compose([
    transforms.RandAugment(2, 14),  # 使用 RandAugment 增强
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-100 数据增强和预处理
train_cifar100_transform = transforms.Compose([
    transforms.RandAugment(2, 14),  # 使用 RandAugment 增强
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def input_dataset(args):
    # 检查并映射 noise_type
    if args.noise_type not in noise_type_map:
        raise ValueError(f"Undefined noise type {args.noise_type}")
    mapped_noise_type = noise_type_map[args.noise_type]
    
    # 设置 noise_path
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            noise_path = './data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            noise_path = './data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
    else:
        noise_path = args.noise_path
    
    args.is_human = False
    
    # 选择数据集类型和相应的变换
    if args.dataset == 'cifar10':
        transform_train = train_cifar10_transform
        transform_test = test_cifar10_transform
        DatasetClass = CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        transform_train = train_cifar100_transform
        transform_test = test_cifar100_transform
        DatasetClass = CIFAR100
        num_classes = 100
    else:
        raise NameError(f'Undefined dataset {args.dataset}')
    
    # 加载有噪声的完整训练集
    full_noisy_train_dataset = DatasetClass(
        root=os.path.expanduser('~/data/'),
        download=True,  
        train=True, 
        transform=transform_train,
        noise_type=mapped_noise_type,
        noise_path=noise_path,
        is_human=args.is_human
    )
    
    # 加载干净的完整训练集
    full_clean_train_dataset = DatasetClass(
        root=os.path.expanduser('~/data/'),
        download=True,  
        train=True, 
        transform=transform_test,
        noise_type='clean_label',
        noise_path=noise_path,  # 可以保持与有噪声训练集相同的 noise_path
        is_human=args.is_human
    )
    
    # 定义训练集和验证集的大小
    num_training_samples = 50000
    num_val_samples = 5000
    num_train_samples = num_training_samples - num_val_samples  # 45000
    
    # 生成划分索引
    if args.dataset == 'cifar100':
        generator = torch.Generator().manual_seed(42)
    else:
        generator = torch.Generator().manual_seed(42)  # 保证划分的可重复性 42 for cifar100，
    indices = list(range(num_training_samples))  # 生成完整索引列表
    train_indices, val_indices = random_split(
        indices, 
        [num_train_samples, num_val_samples],
        generator=generator
    )

    # 从 random_split 返回的 Subset 对象中提取索引
    train_indices = train_indices.indices
    val_indices = val_indices.indices

    # 创建 Subset 数据集
    noisy_train_subset = Subset(full_noisy_train_dataset, train_indices)
    clean_val_subset = Subset(full_clean_train_dataset, val_indices)

    
    # 加载测试集，noise_type 设置为 'clean_label'
    test_dataset = DatasetClass(
        root=os.path.expanduser('~/data/'),
        download=False,  
        train=False, 
        transform=transform_test,
        noise_type='clean_label',
        noise_path=noise_path,
        is_human=args.is_human
    )

    # 假设 full_noisy_train_dataset 是加载后的含噪声训练数据集
    # 获取干净样本的索引
    clean_indices = [i for i, (label, noisy_label) in enumerate(zip(full_noisy_train_dataset.train_labels, full_noisy_train_dataset.train_noisy_labels)) if label == noisy_label]

    # 只保留干净样本的 Subset 数据集
    clean_train_subset = Subset(full_noisy_train_dataset, clean_indices)

    # 获取噪声样本的索引
    noisy_indices = [i for i in range(len(full_noisy_train_dataset)) if i not in clean_indices]


    # 假设 full_noisy_train_dataset 是加载后的含噪声训练数据集
    # 获取干净样本的索引和标签
    correct_labels = np.array(full_noisy_train_dataset.train_labels)  # 真实标签
    noisy_labels = np.array(full_noisy_train_dataset.train_noisy_labels)  # 含噪声标签

    # 初始化一个大小为50000的数组，默认值为-1
    label_array = np.full(len(full_noisy_train_dataset), -1)

    # 遍历每个样本
    for i in range(len(full_noisy_train_dataset)):
        if correct_labels[i] == noisy_labels[i]:
            label_array[i] = -1  # 正确的样本设为-1
        else:
            label_array[i] = correct_labels[i]  # 含噪声样本设为正确标签

    # 保存为 .npy 文件
    np.save('noisy_labels_mask.npy', label_array)

    print("50000的npy文件已生成，包含正确和错误标签信息。")


    # # 获取所有 50000 样本的索引和标签信息 (添加这部分代码)
    # all_indices = []
    # all_labels = []
    # for i in range(len(full_noisy_train_dataset)):
    #     _, label, index = full_noisy_train_dataset[i]
    #     all_indices.append(index)
    #     all_labels.append(label)
    
    # # 将索引和标签合并为一个数组
    # combined_array = np.column_stack((all_indices, all_labels))
    
    # # 保存为 .npy 文件
    # save_path = os.path.join(os.getcwd(), f'{args.dataset}_index_label.npy')
    # np.save(save_path, combined_array)
    # print(f"Saved 50000 index-label mapping to {save_path}, shape: {combined_array.shape}")

    


    

    return noisy_train_subset, clean_val_subset, test_dataset, num_classes, num_train_samples, clean_train_subset, noisy_indices
