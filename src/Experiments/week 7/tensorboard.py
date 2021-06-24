from src.networks.network_mnist import Net
from src.networks.mnist_mean_dataset import MNISTMeanDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
import time
from datetime import date
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from sklearn.metrics import mean_squared_error

# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
# momentum is omitted in this example

n_epochs = 10
log_interval = 10

# For repeatable Experiments we have to set random seeds for anything using random number generation (numpy and random)
# cuDNN uses nondeterministic algorithms which can be disabled setting it to False
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main():
    today = date.today()
    arr = []
    learning_rates = [0.01]
    # learning_rates = [0.0025, 0.005, 0.0075 , 0.01, 0.015, 0.0175, 0.02]

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

    for lr in learning_rates:
        print(f'Learning Rate {lr}')
        print('--------------------------------')

        writer = SummaryWriter(f'runs/lr_{lr}_{today.strftime("%b_%d_%Y")}')

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=64,
                                  shuffle=True)  # shuffle=True
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=64,
                                 shuffle=True)  # shuffle=True

        network = Net()
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Checking if GPU is available
        if torch.cuda.is_available():
            print("Cuda is available")
            network = network.cuda()
            criterion = criterion.cuda()

        train_losses = []
        test_losses = []
        train_counter = []

        def train(epoch):
            network.train()

            running_loss = 0.0
            for batch_idx, (data, means) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    means = means.cuda()

                optimizer.zero_grad()
                outputs = network(data.unsqueeze(1).float())
                loss = criterion(outputs, means.unsqueeze(1).float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % log_interval == 0:
                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / log_interval,
                                      epoch * len(train_loader) + batch_idx)

                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(train_loader)))
                    running_loss = 0.0

        def test(loader, epoch):
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

            writer.add_scalar('test_loss', mse, epoch * len(train_loader))

            print(f'The Mean Squared Error: {mse}')
            return mse

        for epoch in range(n_epochs):
            start = time.time()
            print(f'Starting epoch {epoch}')
            train(epoch)
            test_loss = test(test_loader, epoch)
            test_losses.append(test_loss)

            process = time.time() - start
            arr.append([lr, test_loss,  process])

        # Call flush() method to make sure that all pending events have been written to disk.
        writer.flush()
        writer.close()

    return arr


if __name__ == '__main__':
    data = main()
