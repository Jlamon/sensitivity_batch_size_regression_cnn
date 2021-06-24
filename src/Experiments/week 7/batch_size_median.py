from src.networks.network_mnist import Net
from src.networks.mnist_median_dataset import MNISTMedianDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
import time
from numpy import savetxt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from sklearn.metrics import mean_squared_error

# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
# momentum is omitted in this example

n_epochs = 10
learning_rate = 0.1
log_interval = 10

# For repeatable Experiments we have to set random seeds for anything using random number generation (numpy and random)
# cuDNN uses nondeterministic algorithms which can be disabled setting it to False
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main():
    arr = []
    # batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    train = datasets.MNIST(root='../../../data', train=True, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                           , download=True)

    test = datasets.MNIST(root='../../../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    train_dataset = MNISTMedianDataset(train.data, transform=transforms.ToTensor)
    test_dataset = MNISTMedianDataset(test.data, transform=transforms.ToTensor)

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
        criterion = nn.MSELoss()

        # Checking if GPU is available
        if torch.cuda.is_available():
            print("Cuda is available")
            network = network.cuda()
            criterion = criterion.cuda()

        def train():
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

        for epoch in range(1, n_epochs + 1):
            start = time.time()
            print(f'Starting epoch {epoch}')
            train()
            test_loss = test(test_loader)

            process = time.time() - start
            arr.append([bs, test_loss,  process])

    return arr


if __name__ == '__main__':
    data = main()

    with open('results/batch_size_median.csv', mode='w') as file:
        file_writer = csv.writer(file)

        file_writer.writerow(['BS', 'MSE', 'TIME'])

        for ep in data:
            file_writer.writerow([ep[0], ep[1], ep[2]])

        file.close()
