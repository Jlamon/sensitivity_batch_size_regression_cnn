import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_mean = pd.read_csv('../results/batch_size_mean.csv')
    data_median = pd.read_csv('../results/batch_size_median.csv')
    data_std = pd.read_csv('../results/batch_size_std.csv')
    data_var = pd.read_csv('../results/batch_size_var.csv')

    prev_bs = 2
    mean = []
    median = []
    std = []
    var = []
    curr_mean = float('inf')
    curr_median = float('inf')
    curr_std = float('inf')
    curr_var = float('inf')

    for index, row in data_mean.iterrows():
        if row['BS'] != prev_bs:
            mean.append(curr_mean)
            prev_bs = row['BS']
            curr_mean = float('inf')

        loss = row['MSE']

        if loss < curr_mean:
            curr_mean = loss

    prev_bs = 2
    for index, row in data_median.iterrows():
        if row['BS'] != prev_bs:
            median.append(curr_median)
            prev_bs = row['BS']
            curr_median = float('inf')

        loss = row['MSE']

        if loss < curr_median:
            curr_median = loss
    prev_bs = 2
    for index, row in data_std.iterrows():
        if row['BS'] != prev_bs:
            std.append(curr_std)
            prev_bs = row['BS']
            curr_std = float('inf')

        loss = row['MSE']

        if loss < curr_std:
            curr_std = loss

    prev_bs = 2
    for index, row in data_var.iterrows():
        if row['BS'] != prev_bs:
            var.append(curr_var)
            prev_bs = row['BS']
            curr_var = float('inf')

        loss = row['MSE']

        if loss < curr_var:
            curr_var = loss

    mean.append(curr_mean)
    median.append(curr_median)
    std.append(curr_std)
    var.append(curr_var)

    # batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    length = np.arange(len(batch_sizes))

    plt.plot(mean, marker='o', label='Mean')
    plt.plot(median, marker='o', label='Median')
    plt.plot(std, marker='o', label='Standard Deviation')
    # plt.plot(var, marker='o', label='Variance')
    plt.xticks(length, batch_sizes)

    plt.xlabel('Different Batch Sizes', fontsize=12, weight='bold')
    plt.ylabel('MSE', fontsize=12, weight='bold')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plots/scatter_sensitivity_extreme.png')
    plt.show()


if __name__ == '__main__':
    main()
