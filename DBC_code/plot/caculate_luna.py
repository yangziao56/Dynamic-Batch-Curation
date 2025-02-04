import os
import re
import numpy as np

def find_all_npy_files():
    """搜索当前目录下所有.npy文件"""
    return [f for f in os.listdir('.') if f.endswith('.npy')]

def parse_filename(filename):
    """解析文件名结构"""
    pattern = r'^(vanilla|Ghost|Ours|Ours_last_layer)_(rewards|losses|qvals)_run(\d+)\.npy$'
    match = re.match(pattern, filename)
    if match:
        return match.groups()  # (method, metric, run_idx)
    return None

def load_data_by_method_and_metric(npy_files):
    """加载数据到结构化字典"""
    data_dict = {}
    for f in npy_files:
        parsed = parse_filename(f)
        if not parsed:
            continue
        method, metric, run_idx = parsed
        run_idx = int(run_idx)
        
        if method not in data_dict:
            data_dict[method] = {}
        if metric not in data_dict[method]:
            data_dict[method][metric] = {}
        
        data_dict[method][metric][run_idx] = np.load(f)
    return data_dict

def compute_statistics(data_dict):
    """计算并打印关键指标"""
    # 定义方法显示顺序和友好名称
    method_order = ['vanilla', 'Ghost', 'Ours_last_layer', 'Ours']
    name_map = {
        'vanilla': 'Vanilla',
        'Ghost': 'DBC-C',
        'Ours_last_layer': 'DBC-LL',
        'Ours': 'DBC-LE'
    }
    
    # 结果收集字典
    results = {}
    
    for method in method_order:
        if method not in data_dict or 'rewards' not in data_dict[method]:
            continue
            
        rewards_data = data_dict[method]['rewards']
        last_100_avgs = []
        success_rates = []
        
        # 遍历每个运行的奖励数据
        for run_id, rewards in rewards_data.items():
            # 最后100个episode的平均奖励
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            last_100_avgs.append(np.mean(last_100))
            
            # 成功率计算（奖励>=200的比例）
            success_rate = np.mean(np.array(rewards) >= 200)
            success_rates.append(success_rate)
        
        # 计算均值和标准差
        results[method] = {
            'last_100': (np.mean(last_100_avgs), np.std(last_100_avgs)),
            'success': (np.mean(success_rates)*100, np.std(success_rates)*100)
        }
    
    # 打印格式化结果
    print(f"{'Method':<10} | {'Last 100 Avg (mean±std)':^25} | {'Success Rate % (mean±std)':^25}")
    print('-' * 65)
    for method in method_order:
        if method not in results:
            continue
        data = results[method]
        print(f"{name_map[method]:<10} | "
              f"{data['last_100'][0]:>6.2f} ± {data['last_100'][1]:<5.2f} | "
              f"{data['success'][0]:>6.2f} ± {data['success'][1]:<5.2f}")


def main():
    # 数据加载
    npy_files = find_all_npy_files()
    if not npy_files:
        print("No .npy files found in current directory!")
        return
    
    data_dict = load_data_by_method_and_metric(npy_files)
    
    # 计算并打印统计指标
    compute_statistics(data_dict)

if __name__ == '__main__':
    main()