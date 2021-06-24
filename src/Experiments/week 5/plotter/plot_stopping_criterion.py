import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    data = pd.read_csv('../results/stopping_criterion.csv')

    prev_batch = 16
    x = []
    last_epoch = 0

    for index, row in data.iterrows():
        if row['BS'] != prev_batch:
            x.append(last_epoch + 1)
            last_epoch = 0
            prev_batch = row['BS']
        else:
            last_epoch = row['EPOCH']

    last = data.iloc[-1]['EPOCH']
    x.append(last + 1)

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    length = np.arange(len(batch_sizes))

    plt.bar(length, x, align='center')
    plt.xticks(length, batch_sizes)

    plt.title('Number of epoch per Batch Size', fontsize=12) #, color='white')
    plt.xlabel('Different Batch Sizes', fontsize=12)
    plt.ylabel('# of Epochs', fontsize=12)

    plt.savefig('plots/barchart_stopping_criterion_test.png') #, transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
