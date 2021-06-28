import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_mean = pd.read_csv('../results/bs_var_avg_full.csv')
    data_mean = data_mean.values.tolist()

    data_max = []
    data_min = []
    data_avg = []
    data_std = []

    for index, row in enumerate(data_mean):
        row = row[1:]
        maximum = max(row)
        minimum = min(row)
        avg = sum(row) / len(row)
        std = np.std(row)

        data_std.append(std)
        data_max.append(maximum)
        data_min.append(minimum)
        data_avg.append(avg)

    ax = plt.axes()

    for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    # batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    length = np.arange(len(batch_sizes))

    plt.bar(length, data_avg, yerr=data_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(length, batch_sizes, fontsize=11)
    plt.yticks(fontsize=11)

    plt.xlabel('Different Batch Sizes', fontsize=14, weight='bold')
    plt.ylabel('MSE', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('./plots/error_bar_var_full.png')
    plt.show()


if __name__ == '__main__':
    main()
