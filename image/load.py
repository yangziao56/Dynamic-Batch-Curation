import numpy as np


# Load the newly uploaded file
input_dot_sum_data = np.load("score_matrix_input_dot_sum_cifar100_noisy100_0.npy")
noisy_indices = np.load("cifar100_noisy100_noisy_indices.npy")
score_matrix_grads_dot_cifar10_worst_0 = np.load("score_matrix_grads_dot_cifar10_worst_0.npy")

# Extract rows at noisy indices for the new matrix
input_dot_sum_noisy_rows = input_dot_sum_data[noisy_indices]

# Summarize characteristics of these rows (mean and standard deviation)
input_dot_sum_summary = {
    "mean": input_dot_sum_noisy_rows.mean(axis=1),
    "std_dev": input_dot_sum_noisy_rows.std(axis=1)
}

#input_dot_sum_noisy_rows有多少负数的比例
negatives = input_dot_sum_noisy_rows < 0
negatives_ratio = negatives.sum() / negatives.size
print(negatives_ratio)

#input_dot_sum_noisy_rows有多少数处于-1到1之间比例
between_neg1_pos1 = (input_dot_sum_noisy_rows >= -1) & (input_dot_sum_noisy_rows <= 1)
between_neg1_pos1_ratio = between_neg1_pos1.sum() / between_neg1_pos1.size
print(between_neg1_pos1_ratio)

print(input_dot_sum_summary)


#支出score_matrix_grads_dot_cifar10_worst_0中 最大值和最小值
max_value = score_matrix_grads_dot_cifar10_worst_0.max()
min_value = score_matrix_grads_dot_cifar10_worst_0.min()
print(max_value)
print(min_value)
