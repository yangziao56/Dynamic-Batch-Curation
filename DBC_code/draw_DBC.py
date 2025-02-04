import os
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import copy
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

del font_manager.weight_dict['roman']
# font_manager._rebuild()
rcParams['font.family'] = 'Times New Roman'


scale = 0.2 / 0.55
FONT_SIZE = 38 * scale

score_matrix = np.load('score_matrix_cifar10_worst_0.npy')[:,10:]
labels = np.load('cifar10_worst_index_label.npy')[:,1]
print(score_matrix.shape, labels.shape)

print(labels[:10], np.max(labels), np.min(labels))


score_sum = np.sum(np.abs(score_matrix), axis=1)


train_idx = np.where(score_sum != 0)[0]
print(train_idx.shape)
score_matrix = score_matrix[train_idx]
labels = labels[train_idx]


score_matrix[score_matrix > 1e-4] = 0.99e-4
alpha = 0.7
# ==================== sample ====================
xs = np.arange(len(score_matrix[0]))
plot_idx = np.arange(0,len(score_matrix[0]), 2)
fig, ax = plt.subplots()
plt.scatter(xs[plot_idx], score_matrix[4][plot_idx], lw=1, alpha=alpha, color='tab:blue', label='Sample 1')
plt.scatter(xs[plot_idx], score_matrix[5][plot_idx], lw=1, alpha=alpha, color='tab:orange', label='Sample 2')
plt.scatter(xs[plot_idx], score_matrix[6][plot_idx], lw=1, alpha=alpha, color='tab:green', label='Sample 3')

# plt.hlines(0, -9999, 9999, color='tab:red', lw=3, alpha=0.8)
for i in plot_idx:
    plt.plot([xs[i], xs[i]], [0, score_matrix[4,i]], alpha=alpha, color='tab:blue', linestyle='-', lw=0.6)
    plt.plot([xs[i], xs[i]], [0, score_matrix[5,i]], alpha=alpha, color='tab:orange', linestyle='-', lw=0.6)
    plt.plot([xs[i], xs[i]], [0, score_matrix[6,i]], alpha=alpha, color='tab:green', linestyle='-', lw=0.6)

plt.text(len(xs)*0.05, 1e-4 * 0.9, 'C', fontsize=FONT_SIZE*1.2, fontweight='bold')
plt.xlabel('Epoch', fontsize=FONT_SIZE)
plt.ylabel('Influence Score', fontsize=FONT_SIZE)
plt.xticks([1, 50, 100, 150, 200, 250, 300])
plt.xlim(1, len(xs))
plt.ylim(0, 1e-4)
ax.tick_params(labelsize=FONT_SIZE)
ax.ticklabel_format(style='sci', axis='y', scilimits=(-5,-5))
ax.yaxis.get_offset_text().set(size=FONT_SIZE)
for spine in ax.spines.values():
    spine.set_linewidth(0.3)
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig.set_size_inches(11 * scale, 10 * scale)
plt.legend(loc='upper right', fontsize=FONT_SIZE*0.8)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('DBC_figures/cifar10_sample.pdf')
plt.show()



# ==================== histgram ====================
fig, ax = plt.subplots()
sns.histplot(score_matrix[:,1], kde=False, alpha=alpha, bins=100, binrange=(-1e-4, 1e-4), label='Epoch 1', edgecolor="white", linewidth=0.5)
sns.histplot(score_matrix[:,100], kde=False, alpha=alpha, bins=100, binrange=(-1e-4, 1e-4), label='Epoch 100', edgecolor="white", linewidth=0.5)

plt.text(-1e-4*0.9, 3500 * 0.9, 'A', fontsize=FONT_SIZE*1.2, fontweight='bold')
plt.xlabel('Influence Score', fontsize=FONT_SIZE)
plt.ylabel('#Sample', fontsize=FONT_SIZE)
ax.tick_params(labelsize=FONT_SIZE)
ax.ticklabel_format(style='sci', axis='x', scilimits=(-5,-5))
ax.xaxis.get_offset_text().set(size=FONT_SIZE)
for spine in ax.spines.values():
    spine.set_linewidth(0.3)
plt.xlim(-1e-4, 1e-4)
plt.ylim(0, 3500)
fig.set_size_inches(11 * scale, 10 * scale)
plt.legend(loc='upper right', fontsize=FONT_SIZE*0.8)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('DBC_figures/cifar10_histgram.pdf')
plt.show()








# # # ==================== stack plot ====================
# num_per_class = {}
# count_sum = None
# for i in range(10):
#     cur_idx = np.where(labels == i)[0]

#     cur_score_matrix = score_matrix[cur_idx]

#     cur_score_matrix[cur_score_matrix<=0] = 0
#     cur_score_matrix[cur_score_matrix>0] = 1
#     cur_score_matrix = np.sum(cur_score_matrix, axis=0)
#     num_per_class[i] = cur_score_matrix.copy()


#     if count_sum is None:
#         count_sum = cur_score_matrix
#     else:
#         count_sum += cur_score_matrix


# for i in range(10):
#     num_per_class[i] = num_per_class[i] / count_sum



# fig, ax = plt.subplots()
# plt.stackplot(np.arange(len(num_per_class[0])) + 1, num_per_class[0], num_per_class[1], num_per_class[2], num_per_class[3], num_per_class[4], num_per_class[5], num_per_class[6], num_per_class[7], num_per_class[8], num_per_class[9],
#               labels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'], alpha=0.7)

# handles, labels = plt.gca().get_legend_handles_labels()
# legend = plt.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(1.3, 0.8), title="Classes", fontsize=FONT_SIZE)
# legend.get_title().set_fontsize(FONT_SIZE)
# plt.xlabel('Epoch', fontsize=FONT_SIZE)
# plt.ylabel('#Sample', fontsize=FONT_SIZE)
# plt.xlim(1, len(score_matrix[0]))
# plt.ylim(0, 1)
# ax.tick_params(labelsize=FONT_SIZE)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# fig.set_size_inches(13, 10)
# plt.tight_layout()
# plt.show()













# score_matrix[score_matrix<=0] = 0
# score_matrix[score_matrix>0] = 1
# print(np.sum(score_matrix, axis=0))
# print(np.min(score_matrix), np.max(score_matrix), np.mean(score_matrix))




# def count_number(matrix, min_value, max_value):
#     count_per_column = ((matrix > min_value) & (matrix <= max_value)).sum(axis=0)
#     return count_per_column

# df = pd.DataFrame({
#     'Epoch': np.arange(len(score_matrix[0])),
#     'neg 1': count_number(score_matrix, -1, -4e-5),
#     'neg 2': count_number(score_matrix, -4e-5, -2e-5),
#     'neg 3': count_number(score_matrix, -2e-5, 0),
#     'pos 1': count_number(score_matrix, 0, 2e-5),
#     'pos 2': count_number(score_matrix, 2e-5, 4e-5),
#     'pos 3': count_number(score_matrix, 4e-5, 1),
# })


# fig, ax = plt.subplots()
# plt.stackplot(df['Epoch'], df['neg 1'], df['neg 2'], df['neg 3'], df['pos 1'], df['pos 2'], df['pos 3'],
#               labels=['(4e-5, +$\infty$]', '(2e-5, 4e-5]', '(0, 2e-5]', '(-2e-5, 0]', '(-4e-5, -2e-5]', '(-$\infty$ -4e-5]'][::-1], alpha=0.8)

# handles, labels = plt.gca().get_legend_handles_labels()
# legend = plt.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(1.01, 0.55), title="Influence Score", fontsize=FONT_SIZE)
# legend.get_title().set_fontsize(FONT_SIZE)
# plt.xlabel('Epoch', fontsize=FONT_SIZE)
# plt.ylabel('#Sample', fontsize=FONT_SIZE)
# plt.xlim(0, len(score_matrix[0]) - 1)
# plt.ylim(0, 50000)
# ax.tick_params(labelsize=FONT_SIZE)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# fig.set_size_inches(10, 10)
# plt.tight_layout()
# plt.show()




















# # # ==================== heatmap plot ====================
# def one_sort(matrix, step, max_step, direction):
#     if step == max_step:
#         matrix = matrix[matrix[:, step].argsort()[::direction]]
#         return matrix
#     else:
#         subset = copy.deepcopy(matrix)
#         sorted_subset = subset[subset[:, step].argsort()[::direction]]
#         split_idx = int(np.sum(subset[:, step]))
#         if direction == 1:
#             split_idx = len(subset) - split_idx
#         matrix = sorted_subset
#         matrix[:split_idx] = one_sort(matrix[:split_idx], step + 1, max_step, direction)
#         matrix[split_idx:] = one_sort(matrix[split_idx:], step + 1, max_step, -1 * direction)
#     return matrix

# def my_sort(matrix):
#     num_steps = len(matrix[0])
#     matrix = one_sort(matrix, 0, num_steps - 1, 1)
#     return matrix


# score_matrix[score_matrix<=0] = 0
# score_matrix[score_matrix>0] = 1

# num_steps = 6
# skip = len(score_matrix[0]) / num_steps
# xs = [int(x) for x in np.arange(num_steps) * skip]
# xs[0] = 1
# xs.append(299)
# plot_matrix = score_matrix[:, xs]
# plot_matrix = my_sort(plot_matrix)
# print(plot_matrix.shape)


# cmap = ListedColormap(["white", "tab:blue"])

# fig, ax = plt.subplots()
# sns.heatmap(plot_matrix, cmap=cmap, linewidths=0, cbar=False, annot=False, alpha=0.7)

# plt.text(len(xs)*0.05, len(score_matrix) * 0.1, 'B', fontsize=FONT_SIZE*1.2, fontweight='bold')
# plt.xlabel('Epoch', fontsize=FONT_SIZE)
# plt.ylabel('Sample Index', fontsize=FONT_SIZE)

# # plt.axis('off')
# ax.set_xticks(np.arange(len(xs)) + 0.5)
# ax.set_xticklabels(xs[:-1] + [300], fontsize=FONT_SIZE)
# ax.tick_params(labelsize=FONT_SIZE)
# for spine in ax.spines.values():
#     spine.set_linewidth(0.3)
# plt.gca().tick_params(
#     axis='both',
#     which='both',
#     bottom=False,
#     left=False,
#     labelbottom=True,
#     labelleft=False
# )
# plt.gca().spines['top'].set_visible(True)
# plt.gca().spines['right'].set_visible(True)
# plt.gca().spines['left'].set_visible(True)
# plt.gca().spines['bottom'].set_visible(True)

# fig.set_size_inches(10 * scale, 10 * scale)
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.savefig('DBC_figures/cifar10_heatmap.pdf')
# plt.show()
