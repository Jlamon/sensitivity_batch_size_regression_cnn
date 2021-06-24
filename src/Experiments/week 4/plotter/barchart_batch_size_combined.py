import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    data = pd.read_csv('../results/results_batch_size.csv')

    prev_bs = 16
    y = []
    curr_y = 0
    x = []
    curr_x = 0

    for index, row in data.iterrows():
        if row['BS'] != prev_bs:
            y.append(curr_y)
            x.append(curr_x)
            prev_bs = row['BS']
            curr_x = 0

        # curr_y = math.log10(row['MSE'])
        curr_y = row['MSE']
        curr_x += row['TIME']

    y.append(curr_y)
    x.append(curr_x)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    length = np.arange(len(batch_sizes))
    width = 0.3

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Different Batch Sizes', fontsize=12, weight='bold')
    ax1.set_ylabel('MSE', fontsize=12, color=color, weight='bold')
    ax1.bar(length - width/2, np.array(y), width=width, color=color)
    ax1.tick_params(axis='y', colors=color, labelsize=12)
    # ax1.tick_params(axis='x', colors='white', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('TIME(s)', fontsize=12, color=color, weight='bold')  # we already handled the x-label with ax1
    ax2.bar(length + width/2, np.array(x), width=width, color=color)
    ax2.tick_params(axis='y', colors=color, labelsize=12)
    # ax2.spines['bottom'].set_color('white')
    # ax2.spines['top'].set_color('white')
    # ax2.spines['right'].set_color('white')
    # ax2.spines['left'].set_color('white')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xticks(length, batch_sizes)

    # plt.savefig('./plots/barchart_batch_combined.png', transparent=True)
    plt.tight_layout()
    plt.savefig('./plots/barchart_batch_combined.png')
    plt.show()


if __name__ == '__main__':
    main()
