import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    data = pd.read_csv('../results/results_batch_size.csv')

    prev_bs = 16
    y = []
    curr_y = 0

    for index, row in data.iterrows():
        if row['BS'] != prev_bs:
            y.append(curr_y)
            prev_bs = row['BS']

        # curr_y = math.log10(row['MSE'])
        curr_y = row['MSE']

    y.append(curr_y)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    length = np.arange(len(batch_sizes))

    plt.bar(length, y, align='center')
    plt.xticks(length, batch_sizes)

    plt.title('MSE of last epoch', fontsize=12)
    plt.xlabel('Batch Sizes', fontsize=12)
    plt.ylabel('MSE', fontsize=12)

    plt.savefig('./plots/barchart_batch.png')
    plt.show()


if __name__ == '__main__':
    main()
