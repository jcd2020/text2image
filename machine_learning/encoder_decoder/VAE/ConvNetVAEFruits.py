#!/usr/bin/env python
# coding: utf-8

# ## 0. Imports

import matplotlib.pyplot as plt
import os
from machine_learning.discriminator import classifier
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

# ## 1. Vars

batch_size = 128
epochs = 50
no_cuda = False
seed = 1
log_interval = 50
ZDIM = 256
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)


device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


train_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Training'
val_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Test'

# In[ ]:

os.chdir('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE')

train_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

val_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_root, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


# ## 3. Model

# In[ ]:


class VAE_CNN(nn.Module):
    '''Code taken from: https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd'''
    def __init__(self):
        super(VAE_CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())
        # Encoder
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(25 * 25 * 16, 2048)
        self.fc_bn1 = nn.BatchNorm1d(2048)
        self.fc21 = nn.Linear(2048, ZDIM)
        self.fc22 = nn.Linear(2048, ZDIM)

        # Sampling vector
        self.fc3 = nn.Linear(ZDIM, 2048)
        self.fc_bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 25 * 25 * 16)
        self.fc_bn4 = nn.BatchNorm1d(25 * 25 * 16)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def encode(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3).view(-1, 16*25*25)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 25, 25)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 3, 100, 100)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[ ]:


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        return loss_MSE + 1.5*loss_KLD


model = VAE_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


loss_mse = customLoss()

# ## Train

val_losses = []
train_losses = []


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader_food):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_food.dataset)))
    train_losses.append(train_loss / len(train_loader_food.dataset))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader_food):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_mse(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 3, 100, 100)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)
    return test_loss


def train_loop():
    min_test_loss = 100000
    for epoch in range(1, epochs + 1):
        train(epoch)
        loss = test(epoch)
        if loss < min_test_loss:
            min_test_loss = loss
            print('New min test loss {}'.format(min_test_loss))
            torch.save(model.state_dict(), 'VAEModels/model.pt')
        with torch.no_grad():
            sample = torch.randn(64, ZDIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 100, 100),
                       'results/sample_' + str(epoch) + '.png')

    plt.figure(figsize=(15, 10))
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title("Validation loss and loss per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.show()


    torch.save(model.state_dict(), 'VAEModels/model.pt')
    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space


def test_classifier():
    PATH = '../../discriminator/result/fruits_net_30_epochs.pth'
    device = torch.device("cuda")

    discriminator = classifier.Net()
    discriminator.load_state_dict(torch.load(PATH))
    discriminator.to(device)
    discriminator.eval()

    model = VAE_CNN()
    model.load_state_dict(torch.load('VAEModels/model.pt'))
    model.to(device)
    model.eval()
    total = 0.
    correct = 0.
    avg_loss = 0.
    batches = 0.
    for data in val_loader_food:
        batches += 1
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        fake_images, mu, logvar = model(images)
        outputs = discriminator(fake_images)
        _, predicted = torch.max(outputs, 1)
        loss = classifier.criterion(outputs, labels)
        avg_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Downstream classifier accuracy and loss: %d %d %%' % (
            100 * correct / total, avg_loss / batches))


if __name__ == "__main__":
    test_classifier()
