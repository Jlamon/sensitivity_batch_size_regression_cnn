from src.networks.network_mnist import Net
from src.networks.mnist_mean_dataset import MNISTMeanDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from numpy import savetxt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from sklearn.metrics import mean_squared_error

# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
# momentum is omitted in this example

n_epochs = 10
learning_rate = 0.01
log_interval = 10

# For repeatable Experiments we have to set random seeds for anything using random number generation (numpy and random)
# cuDNN uses nondeterministic algorithms which can be disabled setting it to False
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main():
    # batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [32]

    train = datasets.MNIST(root='../../../data', train=True, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                           , download=True)

    test = datasets.MNIST(root='../../../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    train_dataset = MNISTMeanDataset(train.data, transform=transforms.ToTensor)
    test_dataset = MNISTMeanDataset(test.data, transform=transforms.ToTensor)

    for bs in batch_sizes:
        print(f'Batch Size {bs}')
        print('--------------------------------')

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=bs,
                                  shuffle=True)  # shuffle=True
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=bs,
                                 shuffle=True)  # shuffle=True

        network = Net()
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        # MSE Loss
        criterion = nn.MSELoss()

        # Checking if GPU is available
        if torch.cuda.is_available():
            print("Cuda is available")
            network = network.cuda()
            criterion = criterion.cuda()

        train_losses = []
        test_losses = []
        train_counter = []
        test_counter = []

        def train(epoch):
            network.train()
            for batch_idx, (data, means) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    means = means.cuda()

                optimizer.zero_grad()
                outputs = network(data.unsqueeze(1).float())
                loss = criterion(outputs, means.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * bs) + ((epoch - 1) * len(train_loader)))

        def test(loader):
            test_predicted = []
            test_actual_means = []

            network.eval()
            with torch.no_grad():
                for data, means in loader:
                    if torch.cuda.is_available():
                        data = data.cuda()
                        means = means.cuda()

                    output = network(data.unsqueeze(1).float())
                    test_predicted.append(output.cpu().numpy())
                    test_actual_means.append(means.cpu().numpy())

            test_predicted = [a.squeeze().tolist() for a in test_predicted]
            flat_predicted = [item for sublist in test_predicted for item in sublist]

            test_actual_means = [a.squeeze().tolist() for a in test_actual_means]
            flat_actual_means = [item for sublist in test_actual_means for item in sublist]

            mse = mean_squared_error(flat_actual_means, flat_predicted)
            print(f'The Mean Squared Error: {mse}')
            return mse

        test_losses.append(test(test_loader))
        for epoch in range(1, n_epochs + 1):
            print(f'Starting epoch {epoch}')
            train(epoch)
            test_loss = test(test_loader)
            test_losses.append(test_loss)

        test_counter = [i * len(train_loader) for i in range(n_epochs + 1)]

        savetxt(f'losses/test_counter_mean_bs_{bs}.csv', test_counter, delimiter=',')
        savetxt(f'losses/test_losses_mean_bs_{bs}.csv', test_losses, delimiter=',')
        savetxt(f'losses/train_losses_mean_bs_{bs}.csv', train_losses, delimiter=',')
        savetxt(f'losses/train_counter_mean_bs_{bs}.csv', train_counter, delimiter=',')


if __name__ == '__main__':
    main()
