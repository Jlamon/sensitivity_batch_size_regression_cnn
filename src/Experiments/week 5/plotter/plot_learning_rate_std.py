import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv('../results/learning_rate_std.csv')

    epoch = 0
    prev_fold = 0.00025
    x = []
    y = []
    curr_x = []
    curr_y = []

    for index, row in data.iterrows():
        if row['LR'] != prev_fold:
            x.append(curr_x)
            y.append(curr_y)
            curr_y = []
            curr_x = []
            prev_fold = row['LR']
            epoch = 0

        curr_y.append(row['MSE'])
        curr_x.append(epoch)

        epoch += 1

    x.append(curr_x)
    y.append(curr_y)

    ax = plt.axes()

    for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    # ax.spines['bottom'].set_color('white')
    # ax.spines['top'].set_color('white')
    # ax.spines['right'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    # ax.tick_params(axis='x', colors='white', labelsize=12)
    # ax.tick_params(axis='y', colors='white', labelsize=12)

    # learning_rates = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    learning_rates = [0.00025, 0.0005, 0.00075 , 0.001, 0.0015, 0.00175, 0.002]
    for idx in range(len(learning_rates)):
        label = "Learning Rate: " + str(learning_rates[idx])
        plt.plot(x[idx], y[idx], marker='o', label=label)

    # plt.yscale('log')

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)

    plt.xlabel('Epochs', fontsize=14, weight='bold')
    plt.ylabel('MSE', fontsize=14, weight='bold')
    plt.legend(loc="best", framealpha=0.3)

    plt.tight_layout()
    # plt.savefig('./plots/learning_rates_std.png', transparent=True)
    plt.savefig('./plots/learning_rates_std.png')
    plt.show()


if __name__ == '__main__':
    main()
