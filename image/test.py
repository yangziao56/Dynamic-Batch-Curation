import numpy as np

# 示例参数和数组
args = type('args', (object,), {'points_to_delete': 3})()
I2 = np.array([10, 20, 5, 3, 40, 25])

# 定义函数
def find_trimming_points(args, I2):

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    return indices_to_delete

# 执行函数
indices_to_delete = find_trimming_points(args, I2)

# 打印删除的元素和它们的值
deleted_elements = I2[indices_to_delete]
print("Indices to delete:", indices_to_delete)
print("Deleted elements:", deleted_elements)



import numpy as np

# 定义函数
def find_trimming_points(args, I2):
    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()
    print("# Indices to Delete ==> ", len(indices_to_delete))
    return indices_to_delete

# 示例数据和参数
class Args:
    def __init__(self, points_to_delete):
        self.points_to_delete = points_to_delete

# 创建示例数据
I2 = np.array([10, 50, 30, 20, 40])
args = Args(points_to_delete=2)

# 调用函数并打印结果
indices_to_delete = find_trimming_points(args, I2)
print("Indices to delete:", indices_to_delete)

# 验证删除的值
print("Values to delete:", I2[indices_to_delete])


import numpy as np

# 定义函数
def find_trimming_points(args, I2):

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    return indices_to_delete

# 示例数据和参数
class Args:
    def __init__(self, points_to_delete):
        self.points_to_delete = points_to_delete

# 创建示例数据
I2 = np.array([10, 50, 30, 20, 40])
args = Args(points_to_delete=2)

# 调用函数并打印结果
indices_to_delete = find_trimming_points(args, I2)
print("Indices to delete:", indices_to_delete)

# 验证删除的值
print("Values to delete:", I2[indices_to_delete])
