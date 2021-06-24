import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_mae = pd.read_csv('../results/mae_loss_std.csv')
    data_mse = pd.read_csv('../results/mse_loss_std.csv')

    mae = data_mae['MAE']
    mse = data_mse['MSE']

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

    plt.plot(mae, label='MAE')
    plt.plot(mse, label='MSE')

    # plt.yscale('log')

    # plt.title('Different Batch Sizes', fontsize=12, color='white')
    plt.title('MSE vs MAE', fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Result', fontsize=12)
    plt.legend(loc="best", framealpha=0.3)
    # plt.savefig('plots/batch_size_var.png', transparent=True)
    plt.savefig('plots/differences_std.png')
    plt.show()


if __name__ == '__main__':
    main()
