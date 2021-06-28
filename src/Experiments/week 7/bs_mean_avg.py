from src.networks.network_mnist import Net
from src.networks.mnist_mean_dataset import MNISTMeanDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import Compose
from sklearn.metrics import mean_squared_error

# Code from: https://nextjournal.com/gkoehler/pytorch-mnist
# momentum is omitted in this example

n_epochs = 9
learning_rate = 0.0015
log_interval = 10

# For repeatable Experiments we have to set random seeds for anything using random number generation (numpy and random)
# cuDNN uses nondeterministic algorithms which can be disabled setting it to False
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)


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

    train_dataset = MNISTMeanDataset(train.data, transform=transforms.ToTensor)
    test_dataset = MNISTMeanDataset(test.data, transform=transforms.ToTensor)

    for bs in batch_sizes:
        collect = [bs]

        for run in range(10):
            random_seed = run + bs
            torch.backends.cudnn.enabled = False
            torch.manual_seed(random_seed)

            print(f'Batch Size {bs} and run {run}')
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
                print(f'Starting epoch {epoch}')
                train()
                test(test_loader)
            collect.append(test(test_loader))
        arr.append(collect)
    return arr


if __name__ == '__main__':
    data = main()

    with open('results/bs_mean_avg_seed_full.csv', mode='w') as file:
        file_writer = csv.writer(file)

        file_writer.writerow(['BS', 'RUN_0', 'RUN_1', 'RUN_2', 'RUN_3', 'RUN_4', 'RUN_5', 'RUN_6', 'RUN_7', 'RUN_8', 'RUN_9'])

        for ep in data:
            file_writer.writerow([ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], ep[6], ep[7], ep[8], ep[9], ep[10]])

        file.close()
