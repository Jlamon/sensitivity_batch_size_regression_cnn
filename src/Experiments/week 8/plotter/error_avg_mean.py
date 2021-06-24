import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_mean = pd.read_csv('../results/bs_mean_avg_seed.csv')
    data_mean = data_mean.values.tolist()

    data_max = []
    data_min = []
    data_avg = []

    for index, row in enumerate(data_mean):
        row = row[1:]
        maximum = max(row)
        minimum = min(row)
        avg = sum(row) / len(row)

        data_max.append(maximum)
        data_min.append(minimum)
        data_avg.append(avg)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    length = np.arange(len(batch_sizes))

    plt.plot(data_max, label='Maximum', color='red')
    plt.plot(data_avg, label='Mean')
    plt.plot(data_min, label='Minimum', color='green')
    plt.xticks(length, batch_sizes)

    plt.fill_between(length, data_max, data_avg, color='red', alpha=0.1)
    plt.fill_between(length, data_avg, data_min, color='green', alpha=0.1)

    plt.xlabel('Different Batch Sizes', fontsize=12, weight='bold')
    plt.ylabel('MSE', fontsize=12, weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/error_mean_seed.png')
    plt.show()


if __name__ == '__main__':
    main()
