import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def main():
    data = pd.read_csv('../results/results_learning_rate.csv')

    prev_lr = 0.000001
    time = []
    curr_time = 0

    for index, row in data.iterrows():
        if row['LR'] != prev_lr:
            time.append(curr_time)
            prev_lr = row['LR']
            curr_time = 0

        curr_time += row['TIME']

    time.append(curr_time)

    learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    length = np.arange(len(learning_rates))

    plt.bar(length, time, align='center')
    plt.xticks(length, learning_rates)

    plt.title('Time per learning rates', fontsize=12)
    plt.xlabel('Learning Rates', fontsize=12)
    plt.ylabel('Time', fontsize=12)

    plt.savefig('./plots/barchart_learning_time.png')
    plt.show()


if __name__ == '__main__':
    main()
