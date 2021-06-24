import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    data = pd.read_csv('../results/results_batch_size.csv')

    prev_bs = 16
    time = []
    curr_time = 0

    for index, row in data.iterrows():
        if row['BS'] != prev_bs:
            time.append(curr_time)
            prev_bs = row['BS']
            curr_time = 0

        curr_time += row['TIME']

    time.append(curr_time)

    ax = plt.axes()

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    length = np.arange(len(batch_sizes))

    plt.bar(length, time, align='center')
    plt.xticks(length, batch_sizes)

    plt.title('Time per Batch Size', fontsize=12, color='white')
    plt.xlabel('Batch Sizes', fontsize=12)
    plt.ylabel('Time', fontsize=12)

    plt.savefig('./plots/barchart_batch_time.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
