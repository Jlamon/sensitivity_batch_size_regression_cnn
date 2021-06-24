import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv('../results/batch_size_median.csv')

    epoch = 0
    prev_batch = 16
    x = []
    y = []
    curr_x = []
    curr_y = []

    for index, row in data.iterrows():
        if row['BS'] != prev_batch:
            x.append(curr_x)
            y.append(curr_y)
            curr_y = []
            curr_x = []
            prev_batch = row['BS']
            epoch = 0

        curr_y.append(row['MSE'])
        curr_x.append(epoch)

        epoch += 1

    x.append(curr_x)
    y.append(curr_y)

    # ax = plt.axes()
    #
    # ax.spines['bottom'].set_color('white')
    # ax.spines['top'].set_color('white')
    # ax.spines['right'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    # ax.tick_params(axis='x', colors='white')
    # ax.tick_params(axis='y', colors='white')

    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    for idx in range(len(batch_sizes)):
        label = "Batch Size: " + str(batch_sizes[idx])
        plt.plot(x[idx], y[idx], marker='o', label=label)

    # plt.yscale('log')

    plt.xlabel('Epochs', fontsize=12, weight='bold')
    plt.ylabel('Result', fontsize=12, weight='bold')
    plt.legend(loc="best", framealpha=0.3)

    plt.tight_layout()
    # plt.savefig('plots/batch_size_median.png', transparent=True)
    plt.savefig('plots/batch_size_median.png')
    plt.show()


if __name__ == '__main__':
    main()
