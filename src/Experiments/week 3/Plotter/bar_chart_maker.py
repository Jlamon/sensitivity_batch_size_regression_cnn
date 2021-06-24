import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv('../Experiments/week 3/results/results_barchart_kfold.csv')

    length = np.arange(len(data))
    width = 0.3

    plt.bar(length, data['MSE'], width=width, align='center')
    # plt.bar(length + width, data['R_SQUARED'], width=width)
    plt.bar(length + width, data['MINIMUM'], width=width, align='center')
    plt.bar(length - width, data['MAXIMUM'], width=width, align='center')

    plt.title('Average, Minimum, Maximum', fontsize=12)
    plt.xlabel('Folds', fontsize=12)
    plt.ylabel('Result', fontsize=12)
    plt.legend(loc="best")
    plt.savefig('./plots/mse_avg_max_min_folds.png')
    plt.show()


if __name__ == '__main__':
    main()
