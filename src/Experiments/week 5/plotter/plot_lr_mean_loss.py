import matplotlib.pyplot as plt
import numpy as np


def main():
    train_counter = np.genfromtxt('../losses/lr/train_counter_lr_0.01.csv', delimiter=',')
    train_losses = np.genfromtxt('../losses/lr/train_losses_lr_0.01.csv', delimiter=',')
    test_counter = np.genfromtxt('../losses/lr/test_counter_lr_0.01.csv', delimiter=',')
    test_losses = np.genfromtxt('../losses/lr/test_losses_lr_0.01.csv', delimiter=',')

    ax = plt.axes()

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right', framealpha=0.3)

    plt.title('MAE for a lr of 0.01, batch size of 64', fontsize=12, color='white', weight='bold')
    plt.xlabel('Number of Training Examples Seen', fontsize=12, weight='bold')
    plt.ylabel('MAE Loss', fontsize=12, weight='bold')
    plt.savefig('./plots/mean_train_test_loss_mae.png', transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
