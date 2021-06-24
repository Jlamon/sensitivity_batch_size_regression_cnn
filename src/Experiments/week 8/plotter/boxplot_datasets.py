import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mean = np.genfromtxt('../datasets/train_mean.csv', delimiter=',')
    median = np.genfromtxt('../datasets/train_median.csv', delimiter=',')
    std = np.genfromtxt('../datasets/train_std.csv', delimiter=',')
    var = np.genfromtxt('../datasets/train_var.csv', delimiter=',')
    data = [mean, median, std]

    plt.boxplot(data, labels=("Mean","Median","Std"))
    # plt.boxplot(median)
    plt.title('Box plot of mean, median, and std')
    plt.savefig('./plots/boxplot_datasets.png')
    plt.show()
