import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mean = np.genfromtxt('../datasets/train_mean.csv', delimiter=',')
    median = np.genfromtxt('../datasets/train_median.csv', delimiter=',')
    std = np.genfromtxt('../datasets/train_std.csv', delimiter=',')
    var = np.genfromtxt('../datasets/train_var.csv', delimiter=',')
    data = [mean, median, std]

    ax = plt.axes()

    for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)

    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off

    plt.boxplot(data, labels=("Mean","Median","Std"))
    # plt.xlabel("Variance", fontsize=13, weight='bold')
    # plt.boxplot(var)

    plt.tight_layout()
    plt.savefig('./plots/boxplot_datasets.png')
    plt.show()
