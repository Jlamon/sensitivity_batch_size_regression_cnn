import matplotlib.pyplot as plt
import numpy as np


def main():
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]

    for bs in batch_sizes:
        train_counter = np.genfromtxt(f'../losses/bs_mean/train_counter_mean_bs_{bs}.csv', delimiter=',')
        train_losses = np.genfromtxt(f'../losses/bs_mean/train_losses_mean_bs_{bs}.csv', delimiter=',')
        test_counter = np.genfromtxt(f'../losses/bs_mean/test_counter_mean_bs_{bs}.csv', delimiter=',')
        test_losses = np.genfromtxt(f'../losses/bs_mean/test_losses_mean_bs_{bs}.csv', delimiter=',')

        ax = plt.axes()

        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white', labelsize=12)
        ax.tick_params(axis='y', colors='white', labelsize=12)

        plt.plot(train_counter, train_losses, color='blue', zorder=1)
        plt.scatter(test_counter, test_losses, color='red', zorder=2)
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right', framealpha=0.3)

        plt.title(f'MSE for a lr of 0.01, batch size of {bs}', fontsize=12, color='white', weight='bold')
        plt.xlabel('Number of Training Examples Seen', fontsize=12, weight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('MSE Loss', fontsize=12, weight='bold')

        plt.tight_layout()
        plt.savefig(f'./plots/bs_mean_loss/bs_{bs}_loss_.png', transparent=True)
        plt.show()


if __name__ == '__main__':
    main()
