# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split
# import torch.nn.functional as F
# import torchvision.models as models
# from torchvision.models import ResNet18_Weights
# import random
# import os
# import numpy as np
# import time

# from dataset import get_dataloaders
# from data.datasets import input_dataset
# from model import get_model, get_last_fc_input

# import argparse

# parser = argparse.ArgumentParser('')

# # CIFAR
# parser.add_argument('--dataset', type=str, default='cifar10', help='mnist, cifar10, or cifar100')
# parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='aggre')
# parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)

# parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
# parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
# parser.add_argument('--val_batch_size', type=int, default=12000, help='Validation batch size')
# parser.add_argument('--test_batch_size', type=int, default=1000, help='Test batch size')
# parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
# parser.add_argument('--method', type=str, default='vanilla', help='vanilla, SPL, IP, Ghost, output_grads_dot, self_output_grads_dot')
# # SPL parameters
# parser.add_argument('--v', type=float, default=1, help='SPL weighting parameter')
# parser.add_argument('--lambda_', type=float, default=1e+5, help='SPL threshold')

# # IP parameters
# parser.add_argument('--last_layer', type=bool, default=False, help='Use last layer for SPL')
# parser.add_argument('--calculate_val_grad_per_step', type=bool, default=True, help='Calculate val grad per step')

# parser.add_argument('--seed', type=int, default=42, help='Random seed')
# parser.add_argument('--device', type=str, default='cuda:0', help='Device (cpu or cuda)')
# args = parser.parse_args()

# def set_seed(seed):
#     # 固定 Python 的随机数生成器
#     random.seed(seed)

#     # 固定 NumPy 的随机数生成器
#     np.random.seed(seed)

#     # 固定 PyTorch 的随机数生成器（CPU 和 GPU）
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

#     # 确保 cuDNN 的操作是确定的
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

#     # 确保环境中所有使用随机性的模块都受控
#     os.environ['PYTHONHASHSEED'] = str(seed)

# # 设定一个种子，比如 42
# set_seed(args.seed)

# # Load dataset
# print('Loading dataset...')
# train_dataset, val_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args)

# # Create DataLoaders with batch_size=1 for per-sample gradients
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

# # Load model
# model = get_model(args)

# # Load saved model parameters
# checkpoint_path = f'./{args.dataset}_{args.noise_type}_best_model.pth'
# model.load_state_dict(torch.load(checkpoint_path))
# model.eval().to(args.device)
# correct = 0
# with torch.no_grad():
#     for data, target, _ in test_loader:
#         data, target = data.to(args.device), target.to(args.device)
#         output = model(data)
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()

# accuracy = correct / len(test_loader.dataset)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')



# # Initialize loss and optimizer
# criterion = nn.CrossEntropyLoss(reduction='none')  # We need individual losses

# total_params_last_layer = model.fc.weight.numel() + model.fc.bias.numel()
# total_params = sum(p.numel() for p in model.parameters())

# # Set model to evaluation mode
# model.eval()
# model.to(args.device)

# # Compute average gradient on validation set
# print("Computing average gradient on validation set...")
# model.zero_grad()

# for data, target, _ in test_loader:
#     data, target = data.to(args.device), target.to(args.device)
#     output = model(data)
#     loss = criterion(output, target)
#     avg_loss = torch.mean(loss)
# # Enable gradient computation

#     avg_loss.backward()

# # Collect validation gradients (all parameters)
# val_grads = []
# for param in model.parameters():
#     if param.grad is not None:
#         val_grads.append(param.grad.detach().cpu().view(-1))
# val_grads = torch.cat(val_grads)


# # Collect validation gradients (last layer parameters)
# val_grads_last_layer = []
# for param in model.fc.parameters():
#     if param.grad is not None:
#         val_grads_last_layer.append(param.grad.detach().cpu().view(-1))
# val_grads_last_layer = torch.cat(val_grads_last_layer)



# print("Computing per-sample gradients and inner products...")

# train_inner_products = []
# train_inner_products_last_layer = []

# for data, target, _ in train_loader:
#     data, target = data.to(args.device), target.to(args.device)
#     model.zero_grad()
#     output = model(data)
#     loss = criterion(output, target)
#     loss.backward()
#     # Collect sample gradients (all parameters)
#     sample_grads = []
#     for param in model.parameters():
#         if param.grad is not None:
#             sample_grads.append(param.grad.detach().cpu().view(-1))
#     sample_grads = torch.cat(sample_grads)
#     # Compute inner product (all parameters)
#     inner_product = torch.dot(sample_grads, val_grads)
#     train_inner_products.append(inner_product.item())
#     # Collect sample gradients (last layer)
#     sample_grads_last_layer = []
#     for param in model.fc.parameters():
#         if param.grad is not None:
#             sample_grads_last_layer.append(param.grad.detach().cpu().view(-1))
#     sample_grads_last_layer = torch.cat(sample_grads_last_layer)
#     # Compute inner product (last layer)
#     inner_product_last_layer = torch.dot(sample_grads_last_layer, val_grads_last_layer)
#     train_inner_products_last_layer.append(inner_product_last_layer.item())

# # Optionally, save the inner products to files
# np.save('train_inner_products.npy', np.array(train_inner_products))
# np.save('train_inner_products_last_layer.npy', np.array(train_inner_products_last_layer))
# print("shape of train_inner_products and train_inner_products_last_layer ", np.array(train_inner_products).shape, np.array(train_inner_products_last_layer).shape)
# print("Done computing inner products.")


# # calculate the confusion matrix btw the train_inner_products and train_inner_products_last_layer, using 0 as the threshold
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

# threshold = 0
# train_inner_products = np.array(train_inner_products)
# train_inner_products_last_layer = np.array(train_inner_products_last_layer)
# train_inner_products_last_layer = np.where(train_inner_products_last_layer > threshold, 1, 0)
# train_inner_products = np.where(train_inner_products > threshold, 1, 0)
# cm = confusion_matrix(train_inner_products, train_inner_products_last_layer)

# plt.imshow(cm, cmap='binary')
# plt.colorbar()
# plt.show()
# print(cm)


# #读入train_inner_products 和 train_inner_products_last_layer算 Pearson 相关系数和kendell相关系数
#load
import numpy as np

train_inner_products = np.load('train_inner_products.npy')
train_inner_products_last_layer = np.load('train_inner_products_last_layer.npy')
from scipy.stats import pearsonr, kendalltau
pearson_corr = pearsonr(train_inner_products, train_inner_products_last_layer)
kendall_corr = kendalltau(train_inner_products, train_inner_products_last_layer)
print(f"Pearson correlation: {pearson_corr}")
print(f"Kendall correlation: {kendall_corr}")

#用linear regression拟合train_inner_products 和 train_inner_products_last_layer，计算R^2并print y=kx+b
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
train_inner_products = train_inner_products.reshape(-1, 1)
train_inner_products_last_layer = train_inner_products_last_layer.reshape(-1, 1)
#normalize
train_inner_products = train_inner_products / np.linalg.norm(train_inner_products)
train_inner_products_last_layer = train_inner_products_last_layer / np.linalg.norm(train_inner_products_last_layer)

reg = LinearRegression().fit(train_inner_products, train_inner_products_last_layer)
r2 = reg.score(train_inner_products, train_inner_products_last_layer)
print(f"R^2: {r2}")
print(f"y = {reg.coef_[0][0]}x + {reg.intercept_[0]}")



#画出图像，横坐标是train_inner_products，纵坐标是train_inner_products_last_layer
import matplotlib.pyplot as plt
plt.scatter(train_inner_products, train_inner_products_last_layer, s=1)
plt.xlabel('train_inner_products')
plt.ylabel('train_inner_products_last_layer')
plt.show()



