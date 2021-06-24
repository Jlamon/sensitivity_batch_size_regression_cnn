import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv('../Experiments/week 3/results/results_linechart_kfold.csv')

    epoch = 0
    prev_fold = 0
    x = []
    y = []
    curr_x = []
    curr_y = []

    for index, row in data.iterrows():
        if row['FOLDS'] != prev_fold:
            x.append(curr_x)
            y.append(curr_y)
            curr_y = []
            curr_x = []
            prev_fold = row['FOLDS']
            epoch = 0

        curr_y.append(row['MSE'])
        curr_x.append(epoch)

        epoch += 1

    x.append(curr_x)
    y.append(curr_y)

    for idx in range(len(x)):
        label = "Fold " + str(idx)
        plt.plot(x[idx], y[idx], marker='o', label=label)

    plt.title('Convergence of MSE lr=0.0001', fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Result', fontsize=12)
    plt.legend(loc="best")
    # plt.savefig('./plots/mse_epochs_folds.png')
    plt.show()


if __name__ == '__main__':
    main()
