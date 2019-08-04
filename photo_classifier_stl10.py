import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import functools


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 7)
        self.pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)  # reshape to row vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


def timer(func):
    """timing decorator for training and testing methods"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        print('Finished %s in %.2f seconds'
              % (func.__name__, (end_time - start_time)))
        return value
    return wrapper_timer


def disp_img(img):
    img = img/2 + 0.5  # take from [-1,1] to original [0,1]
    numpy_img = img.numpy()   # convert to np array

    # convert from (channels, xsize, yzise) to (xsize, ysize, channel)
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()


def disp_sample_images(train_loader, classes, batch_size):
    dataiter = iter(train_loader)

    images, labels = dataiter.next()

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # show images
    disp_img(torchvision.utils.make_grid(images))


@timer
def prep_data(batch=4):
    print('Beginning data prep')
    # convert to Tensor and normalize from [0,1] to [-1,1]
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # create a training set and loader from CIFAR-10 data
    train_set = torchvision.datasets.STL10(download=True, root='./data',
                                           split='train', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch,
                                               num_workers=2, shuffle=True)

    # create a testing set and loader from CIFAR-10 data
    test_set = torchvision.datasets.STL10(download=True, root='/.data',
                                          split='test', transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch,
                                              num_workers=2, shuffle=True)

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse',
               'monkey', 'ship', 'truck')

    print('Completed data prep')
    return train_set, train_loader, test_set, test_loader, classes


def loss_and_criterion():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


@timer
def training(train_loader, epochs=2, verbose=False):
    if verbose:
        print('Beginning training')
    criterion, optimizer = loss_and_criterion()

    for epoch in range(epochs):  # loop over the dataset multiple times
        if verbose:
            print('Started Epoch %d' % (epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                if verbose:
                    print('[Epoch %d, Iteration%5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    if verbose:
        print('Finished Training')


@timer
def testing(test_loader, verbose=False):
    if verbose:
        print('Beginning Testing')
    criterion, optimizer = loss_and_criterion()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if verbose:
        print('Finished Testing')
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))


def sample_guesses(test_loader, classes, batch_size):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    print('Truth: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    outputs = net(images)

    energy, index = torch.max(outputs, 1)

    print('Predicted: ',
          ' '.join('%5s' % classes[index[j]] for j in range(batch_size)))

    disp_img(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    _batch_size = 4
    _train_set, _train_loader, _test_set, _test_loader, _classes = prep_data(
        batch=_batch_size)

    training(train_loader=_train_loader, epochs=5, verbose=True)
    # sample_guesses(test_loader=_test_loader, classes=_classes,
    #                batch_size=_batch_size)
    testing(test_loader=_test_loader, verbose=True)