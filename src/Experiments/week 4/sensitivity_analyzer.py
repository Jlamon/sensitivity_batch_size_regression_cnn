from src.networks.network_mnist import Net
from src.networks.mnist_mean_dataset import CustomMNISTDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from sklearn.metrics import mean_squared_error
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
# momentum is omitted in this example

n_epochs = 10
log_interval = 10

# For repeatable Experiments we have to set random seeds for anything using random number generation (numpy and random)
# cuDNN uses nondeterministic algorithms which can be disabled setting it to False
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main(arr):
    train = datasets.MNIST(root='../../../data', train=True, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                           , download=True)

    test = datasets.MNIST(root='../../../data', train=False, transform=Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                          , download=True)

    train_dataset = CustomMNISTDataset(train.data, transform=transforms.ToTensor)
    test_dataset = CustomMNISTDataset(test.data, transform=transforms.ToTensor)

    # Print
    print('STARTING')
    print('--------------------------------')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=int(arr[1]),
                              shuffle=True)  # shuffle=True
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=int(arr[1]),
                             shuffle=True)  # shuffle=True

    network = Net()
    optimizer = optim.Adam(network.parameters(), lr=arr[0])
    criterion = nn.MSELoss()

    # Checking if GPU is available
    if torch.cuda.is_available():
        print("Cuda is available")
        network = network.cuda()
        criterion = criterion.cuda()

    train_losses = []
    train_counter = []
    network.train()

    def train(epoch):
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
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

    def test():
        test_predicted = []
        test_actual_means = []

        network.eval()
        with torch.no_grad():
            for data, means in test_loader:
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
        print(f'Starting epoch {epoch}')
        train(epoch)
        test()

    # torch.save(network.state_dict(), './models/model.pth')
    return test()


if __name__ == '__main__':
    problem = {
        'num_vars': 2,
        'names': ['lr', 'bs'],
        'bounds': [[0.000001, 0.1],
                   [50, 2050]]
    }

    param_values = saltelli.sample(problem, N=2)

    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        Y[i] = main(X)

    print('---------------------')
    print('ANALYZING')

    Si = sobol.analyze(problem, Y, print_to_console=True)

