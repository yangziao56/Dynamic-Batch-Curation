import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 清除缓存并重新生成
mpl.font_manager._rebuild()

# =============== 新增全局字体设置 =============== #
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
# ============================================= #

# 全局参数：指定绘图的 Episode 范围和点数
START_EPISODE = 1200  # 起始 Episode
END_EPISODE = 2000    # 结束 Episode
N_POINTS = 12         # 等距离选择的点数

def find_all_npy_files():
    """
    在当前目录下（plot/）搜索所有 *.npy 文件。
    返回文件名列表。
    """
    all_files = os.listdir('.')
    npy_files = [f for f in all_files if f.endswith('.npy')]
    return npy_files

def parse_filename(filename):
    """
    解析文件名格式，例如：
    'vanilla_rewards_run1.npy' -> method='vanilla', metric='rewards', run=1
    'Ours_qvals_run3.npy'     -> method='Ours', metric='qvals', run=3
    """
    pattern = r'^(vanilla|Ghost|Ours|Ours_last_layer)_(rewards|losses|qvals)_run(\d+)\.npy$'
    match = re.match(pattern, filename)
    if match:
        method, metric, run_str = match.groups()
        run_idx = int(run_str)
        return method, metric, run_idx
    return None

def load_data_by_method_and_metric(npy_files):
    """
    加载所有 npy 文件数据，并按照以下结构存储：
    { method: {metric: { run_i: data_array } } }
    """
    data_dict = {}
    for f in npy_files:
        parsed = parse_filename(f)
        if parsed is None:
            continue
        method, metric, run_idx = parsed
        arr = np.load(f)

        if method not in data_dict:
            data_dict[method] = {}
        if metric not in data_dict[method]:
            data_dict[method][metric] = {}
        data_dict[method][metric][run_idx] = arr

    return data_dict

def plot_metric_subplot(ax, data_dict, metric='rewards', methods=None, title='', ylabel=''):
    """
    绘制指定 metric 的平均曲线及其 std 阴影到指定的 subplot，限定 Episode 范围为 [START_EPISODE, END_EPISODE]
    """
    if methods is None:
        methods = ['vanilla', 'Ghost', 'Ours_last_layer', 'Ours',]

    # 定义友好名称映射（仅用于标签）
    name_mapping = {
        'vanilla': 'Vanilla',
        'Ghost': 'DBC-C',
        'Ours_last_layer': 'DBC-LL',
        'Ours': 'DBC-LE'
        
    }

    # 颜色和线型配置（确保使用有效参数）
    colors = {
        'vanilla': 'blue',
        'Ghost': 'green',
        'Ours': 'red',
        'Ours_last_layer': 'orange'
    }

    linestyles = {  # 确保值都是有效线型符号（如 '-', '--', 等）
        'vanilla': '-',
        'Ghost': '-',
        'Ours': '-',
        'Ours_last_layer': '-'
    }

    # 新增线宽配置字典
    linewidths = {
        'vanilla': 1.3,
        'Ghost': 1.3,
        'Ours': 2.0,          # 重点加粗
        'Ours_last_layer': 1.3
    }


    for method in methods:
        if method not in data_dict:
            continue
        if metric not in data_dict[method]:
            continue

        runs_data = data_dict[method][metric]
        max_len = max(len(arr) for arr in runs_data.values())
        aligned = []
        for run_i, arr in runs_data.items():
            if len(arr) < max_len:
                pad = np.full((max_len - len(arr),), arr[-1])
                arr = np.concatenate([arr, pad])
            aligned.append(arr)
        aligned = np.array(aligned)

        mean_ = aligned.mean(axis=0)
        std_ = aligned.std(axis=0)
        x = np.arange(max_len)

        # 限制绘图范围
        start = max(0, START_EPISODE)
        end = min(max_len, END_EPISODE)
        x = x[start:end]
        mean_ = mean_[start:end]
        std_ = std_[start:end]

        # 等距离选择点
        indices = np.linspace(0, len(x)-1, N_POINTS, dtype=int)
        x_sampled = x[indices]
        mean_sampled = mean_[indices]
        std_sampled = std_[indices]

        # 确保颜色和线型参数使用原始方法名，标签使用友好名称
        ax.plot(
            x_sampled, 
            mean_sampled, 
            label=name_mapping.get(method, method),  # 标签用友好名称
            color=colors.get(method, 'black'),       # 颜色用原始方法名
            linestyle=linestyles.get(method, '-'),   # 线型用原始方法名
            linewidth=linewidths.get(method, 1.0),  # 动态获取线宽，默认1.0
        )
        ax.fill_between(
            x_sampled,
            mean_sampled - std_sampled,
            mean_sampled + std_sampled,
            alpha=0.1,
            color=colors.get(method, 'black')
        )

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)


def main():
    npy_files = find_all_npy_files()
    data_dict = load_data_by_method_and_metric(npy_files)

    # 创建一个包含三个子图的正方形网格
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    
    # 子图 1: Rewards
    plot_metric_subplot(axes[0], data_dict, metric='rewards', title='Rewards', ylabel='Reward')
    
    # 子图 2: Losses
    plot_metric_subplot(axes[1], data_dict, metric='losses', title='Losses', ylabel='Loss')
    
    # 子图 3: Q-Values
    plot_metric_subplot(axes[2], data_dict, metric='qvals', title='Q-Values', ylabel='Q-Value')

    # 在每张子图左上角添加标签
    for i, ax in enumerate(axes):
        ax.text(0.05, 0.95, 
                chr(65 + i),  # 65是'A'的ASCII码，依次生成ABC
                transform=ax.transAxes,
                fontsize=14,
                weight='bold',
                verticalalignment='top',
                backgroundcolor='white')

    # =============== 新增PDF字体嵌入设置 =============== #
    plt.rcParams['pdf.fonttype'] = 42
    # =============================================== #
    
    # 保存整张图
    plt.savefig(f'metrics_plot_{START_EPISODE}_{END_EPISODE}_{N_POINTS}_points.pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()