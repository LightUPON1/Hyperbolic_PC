import re
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
log_file_path = '/home/ubuntu/CodeSpace/Hyperbolic_PC/experiments/base/pretrain/meru_debug/20241027_135708.log'
# log_file_path = '/home/ubuntu/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru_debug/20241026_095602.log'
with open(log_file_path, 'r') as file:
    log_content = file.read()

plt.figure(figsize=(10, 6))
# 使用正则表达式提取 pts = () 内部的数值
pattern_txt = r'txt\s*=\s*\(([^)]+)'
pattern_pts = r'pts\s*=\s*\(([^)]+)'
pattern_img = r'img\s*=\s*\(([^)]+)'
color_ = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
patterns = [pattern_txt, pattern_pts, pattern_img, color_]
j = 0

# 创建图形和轴
fig, ax1 = plt.subplots()

# 绘制第一条曲线，使用左边的Y轴
for pattern in [pattern_txt, pattern_pts, pattern_img]:
    matches = re.findall(pattern, log_content)
    # 将提取的数值转换为浮点数列表
    data = []
    for match in matches:
        values = list(map(float, match.split(' ')))
        data.append(values)

    # 提取均值和方差
    means = [point[0] for point in data]
    stds = [point[1] for point in data]

    # 为每个箱线图添加误差棒，表示方差
    for i, (mean, std_dev) in enumerate(zip(means, stds)):
        if i == 0:
            ax1.errorbar(i, mean, yerr=std_dev, fmt='o', capsize=5, color=color_[j], ecolor=color_[j], elinewidth=2, label=f'{patterns[j][0:3]}')
        else:
            ax1.errorbar(i, mean, yerr=std_dev, fmt='o', capsize=5, color=color_[j], ecolor=color_[j], elinewidth=2)
    j += 1

ax1.legend(loc='upper left')
ax1.set_title('Boxplot with Variance as Error Bars')
ax1.set_xlabel('Epochss')
ax1.set_ylabel('Values')

# 创建第二个Y轴，共享X轴
ax2 = ax1.twinx()

# 匹配 svm acc = 9.22 这样的浮点数
pattern_acc = r'svm acc\s*=\s*(\d+\.\d+)'
matches_acc = re.findall(pattern_acc, log_content)
acc = [float(match) for match in matches_acc]

# 绘制 acc 数据，使用右边的Y轴
ax2.plot(range(len(acc)), acc, color='y', marker='*', label='acc')
ax2.set_ylabel('Accuracy (%)', color='y')
ax2.tick_params(axis='y', labelcolor='y')
ax2.set_ylim(0, 100)  # 设置右边的 Y 轴范围为 0 到 100

ax2.legend(loc='upper right')

# 显示图表
plt.show()