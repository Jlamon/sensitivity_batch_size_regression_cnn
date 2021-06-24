import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    data = pd.read_csv('../results/results_learning_rate.csv')

    epoch = 0
    prev_lr = 0.000001
    y = []
    curr_y = 0

    for index, row in data.iterrows():
        if row['LR'] != prev_lr:
            y.append(curr_y)
            prev_lr = row['LR']
            epoch = 0

        # curr_y = math.log10(row['MSE'])
        curr_y = row['MSE']
        epoch += 1

    y.append(curr_y)

    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    length = np.arange(len(learning_rates))
    width = 0.3

    plt.bar(length, y, width=width, align='center')
    plt.xticks(length, learning_rates)

    plt.title('MSE of last epoch', fontsize=12)
    plt.xlabel('Learning Rates', fontsize=12)
    plt.ylabel('MSE', fontsize=12)

    plt.savefig('./plots/barchart_learning.png')
    plt.show()


if __name__ == '__main__':
    main()
