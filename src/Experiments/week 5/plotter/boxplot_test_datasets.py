import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mean = np.genfromtxt('../datasets/test_mean.csv', delimiter=',')
    median = np.genfromtxt('../datasets/test_median.csv', delimiter=',')
    std = np.genfromtxt('../datasets/test_std.csv', delimiter=',')
    data = [mean, median, std]

    plt.boxplot(data, labels=("Mean","Median","Std"))
    # plt.boxplot(median)
    plt.title('Box plot of mean, median, and std')

    plt.show()
