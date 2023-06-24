# import pandas as pd
# import matplotlib.pyplot as plt

# data = [
#     [32.20, 24.85, 36.05, 38.91, 45.15],
#     [54.64, 47.43, 53.31, 38.91, 50.65],
#     [41.90, 47.12,  46.44, 42.85, 38.97],
#     [59.40, 54.05, 55.63, 55.29, 63.19],
#     [65.67, 64.14, 62.79, 60.24, 67.48]
# ]

# df = pd.DataFrame(data)
# print(df.describe())
# df.plot.box(title="Box Chart")
# plt.grid(linestyle="--", alpha=0.3)
# plt.ylabel("mIoU (%)")
# plt.xticks([1,2,3,4,5], ['MiB', 'PLOP', 'UCD', 'RCIL', 'Ours'])
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#### ==============mIoU_curve==================
<<<<<<< HEAD
# fig, x_axis = plt.subplots(figsize=(8,5))
# plt.rcParams['xtick.direction'] = 'in'#将x轴刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# plt.rcParams.update({'font.size': 14})#字号
# x_axis = [15,16,17,18,19,20]
# plt.ylim(20, 80)     # 设置y轴刻度范围
# labels = ['MiB', 'PLOP', 'UCD', 'RCIL', 'Ours']
# colors = ['gray', 'gold', 'blue', 'green', 'darkred']
# #设置坐标轴
# plt.grid(linestyle="--", alpha=0.3)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.ylabel('mIoU(%)', size=14)
# plt.xlabel('number of learned classes', size=14)


# IoU_MiB = [75.57, 67.44, 56.50, 51.72, 37.60, 29.70]
# IoU_PLOP = [75.57, 70.86, 65.10, 61.22, 58.28, 53.00]
# IoU_UCD = [75.57, 70.32, 66.89, 59.34, 50.67, 41.90]
# IoU_RCIL = [75.57, 72.99, 69.38, 67.22, 63.27, 59.40]
# IoU_Ours = [75.57, 72.53, 70.92, 68.54, 67.04, 66.08]

# #MiB
# plt.plot(x_axis, IoU_MiB, color=colors[0], alpha=0.7, label=labels[0], marker='d', linestyle='-', linewidth=3)
# #PLOP
# plt.plot(x_axis, IoU_PLOP, color=colors[1], alpha=0.7, label=labels[1], marker='*', linestyle='--', linewidth=3)
# #UCD
# plt.plot(x_axis, IoU_UCD, color=colors[2], alpha=0.7, label=labels[2], marker='o', linestyle=':', linewidth=3)
# #RCIL
# plt.plot(x_axis, IoU_RCIL, color=colors[3], alpha=0.7, label=labels[3], marker='^', linestyle='-.', linewidth=3)
# #Ours
# plt.plot(x_axis, IoU_Ours, color=colors[4], alpha=0.7, label=labels[4], marker='s', linestyle='-', linewidth=3)

# # 添加图例
# plt.legend(loc=0)

# plt.show()
# fig.savefig('mIoU_curve.pdf',dpi=600,format='pdf')
=======
plt.rcParams['xtick.direction'] = 'in'#将x轴刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.rcParams.update({'font.size': 14})#字号
x_axis = [15,16,17,18,19,20]
plt.ylim(20, 80)     # 设置y轴刻度范围
labels = ['MiB', 'PLOP', 'UCD', 'RCIL', 'Ours']
colors = ['gray', 'gold', 'blue', 'green', 'darkred']
#设置坐标轴
plt.grid(linestyle="--", alpha=0.3)
plt.ylabel('mIoU(%)')
plt.xlabel('number of learned classes')


IoU_MiB = [75.57, 67.44, 56.50, 51.72, 37.60, 29.70]
IoU_PLOP = [75.57, 70.86, 65.10, 61.22, 58.28, 53.00]
IoU_UCD = [75.57, 70.32, 66.89, 59.34, 50.67, 41.90]
IoU_RCIL = [75.57, 72.99, 69.38, 67.22, 63.27, 59.40]
IoU_Ours = [75.57, 72.53, 70.92, 68.54, 67.04, 65.67]

#MiB
plt.plot(x_axis, IoU_MiB, color=colors[0], alpha=0.7, label=labels[0], marker='d', linestyle='-', linewidth=3)
#PLOP
plt.plot(x_axis, IoU_PLOP, color=colors[1], alpha=0.7, label=labels[1], marker='*', linestyle='--', linewidth=3)
#UCD
plt.plot(x_axis, IoU_UCD, color=colors[2], alpha=0.7, label=labels[2], marker='o', linestyle=':', linewidth=3)
#RCIL
plt.plot(x_axis, IoU_RCIL, color=colors[3], alpha=0.7, label=labels[3], marker='^', linestyle='-.', linewidth=3)
#Ours
plt.plot(x_axis, IoU_Ours, color=colors[4], alpha=0.7, label=labels[4], marker='s', linestyle='-', linewidth=3)

# 添加图例
plt.legend(loc=0)

plt.show()

>>>>>>> ecae318e6dc743ca5dadc27edce5539b03438991




#### ==============mIoU_class_orders==================
<<<<<<< HEAD
fig, x_axis = plt.subplots(figsize=(9,5))
plt.rcParams['xtick.direction'] = 'in'#将x轴刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.rcParams.update({'font.size': 14})#字号

x_axis = np.arange(5)
y_axis = [0, 10, 20, 30, 40, 50, 60, 70]

# 设置均值和误差范围
mean_value = [35.43, 52.96, 43.46, 57.51, 64.146]
std_value = [7.57202879, 4.264219741, 3.363098274, 3.75, 2.52791]

# 设置误差棒属性
err_attri = dict(elinewidth=1, ecolor='black', capsize=4)

# 设置柱标签名
labels = ['MiB', 'PLOP', 'UCD', 'RCIL', 'Ours']
colors = ['gray', 'gold', 'blue', 'green', 'darkred']

#生成
plt.bar(x_axis, mean_value, color=colors, width=0.75, align='center', alpha=0.5, yerr=std_value, error_kw=err_attri, tick_label=labels)

#设置坐标轴
plt.grid(linestyle="--", alpha=0.3)
plt.xticks(size=14)
plt.yticks(size=14)
plt.ylabel('mIoU(%)', size=16)
# plt.title('')

plt.show()
fig.savefig('mIoU_class_orders.pdf',dpi=600,format='pdf')
=======
# # plt.rcParams['xtick.direction'] = 'in'#将x轴刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# plt.rcParams.update({'font.size': 14})#字号

# x_axis = np.arange(5)
# y_axis = [0, 10, 20, 30, 40, 50, 60, 70]

# # 设置均值和误差范围
# mean_value = [35.43, 52.96, 43.46, 57.51, 64.06]
# std_value = [7.57202879, 4.264219741, 3.363098274, 3.75, 2.761345686]

# # 设置误差棒属性
# err_attri = dict(elinewidth=1, ecolor='black', capsize=4)

# # 设置柱标签名
# labels = ['MiB', 'PLOP', 'UCD', 'RCIL', 'Ours']
# colors = ['gray', 'gold', 'blue', 'green', 'darkred']

# #生成
# plt.bar(x_axis, mean_value, color=colors, width=0.75, align='center', alpha=0.5, yerr=std_value, error_kw=err_attri, tick_label=labels)

# #设置坐标轴
# plt.grid(linestyle="--", alpha=0.3)
# plt.ylabel('mIoU(%)')
# # plt.title('')

# plt.show()
>>>>>>> ecae318e6dc743ca5dadc27edce5539b03438991
