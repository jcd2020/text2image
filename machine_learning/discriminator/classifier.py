# -*- coding: utf-8 -*-
"""
Training a Classifier
=====================
This is it. You have seen how to define neural networks, compute loss and make
updates to the weights of the network.
Now you might be thinking,
What about data?
----------------
Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.
-  For images, packages such as Pillow, OpenCV are useful
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful
Specifically for vision, we have created a package called
``torchvision``, that has data loaders for common datasets such as
Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz.,
``torchvision.datasets`` and ``torch.utils.data.DataLoader``.
This provides a huge convenience and avoids writing boilerplate code.
For this tutorial, we will use the CIFAR10 dataset.
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,
‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
.. figure:: /_static/img/cifar10.png
   :alt: cifar10
   cifar10
Training an image classifier
----------------------------
We will do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
import torch
import torchvision
from torchvision import datasets, transforms
import os
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.
BATCH_SIZE = 16
epochs = 50
no_cuda = True
seed = 1
log_interval = 50
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)


device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# ## 2. Data loaders

os.chdir("/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Training'
val_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Test'


train_set = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_root, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F

CATEGORIES = len(train_set.class_to_idx)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*22*22, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, CATEGORIES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*22*22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net.pth'

criterion = nn.CrossEntropyLoss()

def train_network():
    net = Net()

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.
    max_test_acc = 0.

    for epoch in range(0, epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        total_correct = 0.0
        total_predicted = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_predicted += labels.size()[0]
            total_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f, running train accuracy: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000, total_correct/total_predicted))
                total_correct = 0.
                total_predicted = 0.
                running_loss = 0.0
        print('Epoch {} done training'.format(epoch+1))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Epoch %d; Accuracy of the network on the %d test images: %d %%' % (
            epoch+1, total, 100 * correct / total))

        if correct / total > max_test_acc:
            print('Test accuracy increased; saving network')
            max_test_acc = correct/total
            torch.save(net.state_dict(), PATH)


def evaluate_model():
    PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net_30_epochs.pth'
    classes = {v: k for k, v in train_set.class_to_idx.items()}

    net = Net()
    net.load_state_dict(torch.load(PATH))



    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: %d %%' % (
        100 * correct / total))

    ########################################################################
    # That looks way better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(CATEGORIES))
    class_total = list(0. for i in range(CATEGORIES))

    confusion_matrix = list({} for i in range(CATEGORIES))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = int(labels[i])
                prediction = int(predicted[i])
                if label != prediction:
                    if prediction not in confusion_matrix[label]:
                        confusion_matrix[label][prediction] = 0.
                    confusion_matrix[label][prediction] += 1
                class_correct[label] += c[i].item()
                class_total[label] += 1



    import heapq
    rank_entries = [() for i in range(CATEGORIES)]
    for i in range(CATEGORIES):
        top_confusions = heapq.nlargest(5, confusion_matrix[i], key=confusion_matrix[i].__getitem__)
        confused_classes = [classes[j] for j in top_confusions]
        confused_pcts = [confusion_matrix[i][j] / class_total[i] for j in top_confusions]
        zipped_confusions = list(zip(confused_classes, confused_pcts))
        confusions = ', '.join([x[0] + ': ' + str(round(x[1], 3)) for x in zipped_confusions])
        cls_name = classes[i]
        accuracy = round(100 * class_correct[i] / class_total[i], 3)
        rank_entries[i] = (cls_name, accuracy, confusions)

    confusion_matrix = sorted(rank_entries, key=lambda x: x[1], reverse=True)

    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/classifier_results.csv', 'w+') as f:
        for entry in confusion_matrix:
            entry = (entry[0], str(entry[1]), entry[2])
            f.write(', '.join(entry) + '\n')

if __name__ == "__main__":
    train_network()