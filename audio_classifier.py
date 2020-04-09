import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import datetime
import functools
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)  
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(16*64*64, 120) # 256 / 2 / 2 because of 2 maxpool2d(2) calls
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*64*64)  # reshape to row vector
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
        print('Finished {} in {} seconds'.format(func.__name__, round((end_time - start_time),3)))
        return value
    return wrapper_timer


def make_and_split(train_ratio=0.75):

    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(root='./raw_spectrograms/', transform=transform)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size 

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_set, test_set 



def disp_img(img):
    img = img/2 + 0.5  # take from [-1,1] to original [0,1]
    numpy_img = img.numpy()   # convert to np array

    # convert from (channels, xsize, yzise) to (xsize, ysize, channel)
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()


@timer
def prep_data(batch_size=10):
    # convert to Tensor and normalize from [0,1] to [-1,1]
    train_set, test_set = make_and_split()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               num_workers=2, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              num_workers=2, shuffle=True)

    classes = ('a', 'b', 'c', 'd', 'e', 
               'f', 'g', 'h', 'i', 'j')

    return train_set, train_loader, test_set, test_loader, classes

def loss_and_criterion():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer


@timer
def training(epochs=2):
    criterion, optimizer = loss_and_criterion()

    for epoch in range(epochs):  # loop over the dataset multiple times
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
            if i % 100 == 99:    # print every 100 minibatches
                print('[epoch {}, minibatch count {}] loss: {}'.format(
                    epoch + 1, i + 1, round(running_loss / 100, 3)))
                running_loss = 0.0

    print('Finished Training')


@timer
def testing():
    print('Starting Testing')

    correct = Counter({i:0 for i in range(len(classes))}) # to support inc on common keys
    total = Counter({i:0 for i in range(len(classes))})

    true_labels = []
    predicted_labels = []
    with torch.no_grad(): # performance - dont need to compute loss on testing
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels = true_labels + list(labels) 
            predicted_labels = predicted_labels + list(predicted)
            total_inc = Counter(map(lambda x: int(x), labels))

            correct_list = (predicted == labels)
            correct_with_labels = list(filter(lambda x: x[1] == True, (zip(labels, correct_list))))
            correct_inc = Counter([int(x[0]) for x in correct_with_labels])

            correct = correct + correct_inc
            total = total + total_inc 

    print('Finished Testing')

    for x in classes:
        i = classes.index(x)
        print('For class \'{}\', the classifier answered {} correct out of {} total for an accuracy of {}%'
            .format(x, correct[i], total[i], 100*correct[i]/total[i]))
    
    print('\nOverall, the classifier answered {} correct out of {} total for an accuracy of {}%'
        .format(sum(correct.values()), 
                sum(total.values()), 
                round(100 * sum(correct.values()) / sum(total.values()),3)))


    cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classes)

    disp = disp.plot(include_values=True,
                 cmap='viridis', 
                 ax=None, 
                 xticks_rotation='horizontal',
                 values_format='d')

    # plt.savefig('pres_images/confusion_matrices/raw_'+str(datetime.datetime.now())+'.png')
    plt.show()
    plt.close()

    return None 

def sample_guesses():
    test_loader, classes = prep_data()[3], prep_data()[4]
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    print('Truth: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

    outputs = net(images)

    _, index = torch.max(outputs, 1)

    print('Predicted: ',
          ' '.join('%5s' % classes[index[j]] for j in range(len(labels))))

    # keep the output width manageable
    disp_img(torchvision.utils.make_grid(images, nrow=min(len(images), 15)))


if __name__ == '__main__':
    train_set, train_loader, test_set, test_loader, classes = prep_data()
    training(epochs=10)
    testing()
    sample_guesses()
