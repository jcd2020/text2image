import random

import matplotlib.pyplot as plt
from machine_learning.discriminator import classifier

import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import pickle
import numpy as np
import math
from ConvNetVAEFruits import VAE_CNN, ZDIM
from TextEncoder import TextEncoder, generate_instance, ENC_SIZE, EMBED_SIZE, RANDOM_SIZE, BATCH_SIZE, TEST_FRUITS, generate_text_instances
N_SAMPLES = 50
N_SAMPLES = 50
RAND_SIZE = 8
DEVICE = torch.device("cuda")


class TextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self):
        super(TextCNN, self).__init__()

        first_size = 100
        next_size = 512
        self.dropout = nn.Dropout(0.80)  # a dropout layer

        self.fc1_txt = nn.Sequential(nn.Linear(EMBED_SIZE + RANDOM_SIZE, next_size),
                                     nn.BatchNorm1d(next_size),
                                     nn.ReLU(),
                                     self.dropout)

        self.fc2_txt = nn.Sequential(nn.Linear(next_size, next_size),
                                     nn.BatchNorm1d(next_size),
                                     nn.ReLU(),
                                     self.dropout)

        self.fc3_txt = nn.Sequential(nn.Linear(next_size, next_size),
                                     nn.BatchNorm1d(next_size),
                                     nn.ReLU(),
                                     self.dropout)

        self.fc4_txt = nn.Sequential(nn.Linear(next_size, next_size),
                                     nn.BatchNorm1d(next_size),
                                     nn.ReLU(),
                                     self.dropout)

        # two different convolutional layers

        # Latent vectors mu and sigma
        self.fc21_txt = nn.Linear(next_size, int(ENC_SIZE / 2))
        self.fc22_txt = nn.Linear(next_size, int(ENC_SIZE / 2))

        self.relu = nn.ReLU()

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


    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def encode(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        x1 = self.fc1_txt(inputs)
        x2 = self.fc2_txt(x1)
        x3 = self.fc3_txt(x2)
        x4 = self.fc4_txt(x3)

        r1 = self.fc21_txt(x4)
        r2 = self.fc22_txt(x4)

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

kwargs = {'num_workers': 1, 'pin_memory': True}
train_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Training'
train_set = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
train_loader_food = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

def prep_tensors():
    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}
    train_data = []
    test_data = []

    test_tmp = []
    train_tmp = []
    for (imgs, labels) in train_loader_food:
        for idx, label in enumerate(labels):
            fruit_name = rev_classes[label.item()]
            encoding = generate_instance(fruit_name)[0]
            instn = (encoding, imgs[idx], label)
            if label.item() in TEST_FRUITS:
                test_tmp.append(instn)
                if len(test_tmp) == BATCH_SIZE:
                    unzipped_batch = list(zip(*test_tmp))
                    batch_encs = list(unzipped_batch[0])
                    batch_imgs = list(unzipped_batch[1])
                    batch_labels = list(unzipped_batch[2])

                    batch_encs = torch.stack(batch_encs)
                    batch_labels = torch.stack(batch_labels)
                    batch_imgs = torch.stack(batch_imgs)

                    instn = (batch_encs, batch_imgs, batch_labels)
                    test_data.append(instn)
                    test_tmp = []
            else:
                train_tmp.append(instn)
                if len(train_tmp) == BATCH_SIZE:
                    unzipped_batch = list(zip(*train_tmp))
                    batch_encs = list(unzipped_batch[0])
                    batch_imgs = list(unzipped_batch[1])
                    batch_labels = list(unzipped_batch[2])

                    batch_encs = torch.stack(batch_encs)
                    batch_labels = torch.stack(batch_labels)
                    batch_imgs = torch.stack(batch_imgs)

                    instn = (batch_encs, batch_imgs, batch_labels)
                    train_data.append(instn)
                    train_tmp = []

    return train_data, test_data



def prep_data(tensors):
    classes = train_set.class_to_idx
    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}
    train_data = []
    test_data = []
    if True:
        fruits = list(map(lambda x : classes[x], list(tensors.keys())))
        random.shuffle(fruits)
        train_fruits = [x for x in fruits if x not in TEST_FRUITS]

        test = []
        for fruit_id in TEST_FRUITS:
            fruit_name = rev_classes[fruit_id]
            label = torch.tensor(np.asarray([fruit_id]))
            for tensor in tensors[fruit_name]:
                test.append((tensor, label))

        random.shuffle(test)
        num_batches = int(len(test)/BATCH_SIZE)
        for i in range(num_batches):
            batch = test[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            unzipped_batch = list(zip(*batch))
            batch_tensors = list(unzipped_batch[0])
            batch_labels = list(unzipped_batch[1])

            batch_tensors = torch.stack(batch_tensors)
            batch_labels = torch.cat(batch_labels)
            test_data.append((batch_tensors, batch_labels))

        train = []
        for fruit_id in train_fruits:
            fruit_name = rev_classes[fruit_id]
            label = torch.tensor(np.asarray([fruit_id]))
            for tensor in tensors[fruit_name]:
                train.append((tensor, label))

        random.shuffle(train)
        num_batches = int(len(train) / BATCH_SIZE)
        for i in range(num_batches):
            batch = train[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
            unzipped_batch = list(zip(*batch))
            batch_tensors = list(unzipped_batch[0])
            batch_labels = list(unzipped_batch[1])

            batch_tensors = torch.stack(batch_tensors)
            batch_labels = torch.cat(batch_labels)
            train_data.append((batch_tensors, batch_labels))

    return train_data, test_data



def update_params_model(text_cnn, VAE=True):
    if VAE:
        model = VAE_CNN()
        model.load_state_dict(torch.load('VAEModels/model.pt'))
        model.to(DEVICE)
    else:
        model = TextEncoder()
        model.load_state_dict(torch.load('TextEnc/model_individual_sentences.pt'))
        model.to(DEVICE)

    pre_trained_state_dict = model.state_dict()

    text_state_dict = text_cnn.state_dict()

    # Fiter out unneccessary keys
    filtered_dict = {k: v for k, v in pre_trained_state_dict.items() if k in text_state_dict}

    text_state_dict.update(filtered_dict)
    text_cnn.load_state_dict(text_state_dict)

    # if VAE:
    #     print("Freezing updates for decoder layers")
    #     idx = 0
    #     for param in text_cnn.parameters():
    #         if idx > 19:
    #             param.requires_grad = False
    #         idx += 1

    return text_cnn

def load_model():
    text_cnn = TextCNN()
    text_cnn.to(DEVICE)


    DISCRIMNATOR_PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net_30_epochs.pth'

    discriminator = classifier.Net()
    discriminator.load_state_dict(torch.load(DISCRIMNATOR_PATH))
    discriminator.to(DEVICE)
    discriminator.eval()


    return text_cnn, discriminator



class customDiscriminatorLoss(nn.Module):
    def __init__(self, d):
        super(customDiscriminatorLoss, self).__init__()
        self.discriminator = d
        self.cross_ent = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, x_recon, imgs, labels, mu, logvar):
        predictions = self.discriminator(x_recon)
        _, predicted = torch.max(predictions.data, 1)
        total_predicted = labels.size()[0]
        total_correct = (predicted == labels).sum().item()


        loss_DISCRIM = self.cross_ent(predictions, labels)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # # Normalise by same number of elements as in reconstruction

        loss_MSE = self.mse(x_recon, imgs)

        return loss_MSE + loss_DISCRIM/200, total_correct, total_predicted




def train(model,train_data,  optimizer, discrim_loss, epoch, losses):
    model.train()
    train_loss = 0
    total_correct = 0
    total_predicted = 0
    for batch_idx, (imgs, input_vecs, _, labels) in enumerate(train_data):
        input_vecs = input_vecs.to(DEVICE)
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        model.zero_grad()
        recon_batch, mu, logvar = model(input_vecs)
        loss, correct, predicted = discrim_loss(recon_batch, imgs, labels, mu, logvar)

        total_correct += correct
        total_predicted += predicted
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                epoch, batch_idx * len(input_vecs), len(train_data)*BATCH_SIZE,
                       100. * batch_idx / len(train_data),
                       loss.item() / len(input_vecs),
                        total_correct/total_predicted))

    print('====> Epoch: {} Average loss: {:.8f}'.format(
        epoch, train_loss / (len(train_data)*BATCH_SIZE)))
    losses.append(train_loss / (len(train_data)*BATCH_SIZE))
    return total_correct/total_predicted

def test(model, test_data, discrim_loss, epoch, losses):
    model.eval()
    test_loss = 0
    img_map = {}
    total_correct = 0
    total_predicted = 0
    with torch.no_grad():
        for i,  (imgs, input_vecs, _, labels) in enumerate(test_data):
            input_vecs = input_vecs.to(DEVICE)
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            recon_batch, mu, logvar = model(input_vecs)
            loss, correct, predicted = discrim_loss(recon_batch, imgs, labels, mu, logvar)
            total_correct += correct
            total_predicted += predicted
            test_loss += loss.item()
            for idx, label in enumerate(labels):
                img_map[label.item()] = recon_batch[idx]


        comparison = torch.stack(list(img_map.values())).view(len(img_map), 3, 100, 100)
        save_image(comparison.cpu(), 'text_decoding_results/reconstruction_' + str(epoch) + '.png', nrow=len(img_map))
                

    test_loss /= len(test_data)*BATCH_SIZE
    print('====> Test set loss: {:.8f} accuracy: {:.4f}'.format(test_loss, total_correct/total_predicted))
    losses.append(test_loss)
    return test_loss, total_correct/total_predicted


def train_loop(text_cnn, discriminator):
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/train_individual_sentences.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/test_individual_sentences.pkl', 'rb') as f:
        test_data = pickle.load(f)

    epochs = 350
    learning_rate = 5e-5

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, text_cnn.parameters()), lr=learning_rate)

    discrim_loss = customDiscriminatorLoss(discriminator)

    # ## Train

    val_losses = []
    train_losses = []


    min_test_loss = 100000
    max_train_acc = 0
    max_test_acc = 0
    for epoch in range(1, epochs + 1):
        acc = train(text_cnn, train_data, optimizer, discrim_loss, epoch, train_losses)
        loss, test_acc = test(text_cnn, test_data, discrim_loss, epoch, val_losses)
        if loss < min_test_loss:
            min_test_loss = loss
            print('New min test loss {}'.format(min_test_loss))
            torch.save(text_cnn.state_dict(), 'TextVAEModels/model_test_loss_opt_individual_sents.pt')
        if acc > max_train_acc:
            max_train_acc = acc
            print('New max train acc {}'.format(max_train_acc))
            torch.save(text_cnn.state_dict(), 'TextVAEModels/model_train_acc_opt_individual_sents.pt')
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            print('New max test acc {}'.format(max_test_acc))
            torch.save(text_cnn.state_dict(), 'TextVAEModels/model_test_acc_opt_individual_sents.pt')
        with torch.no_grad():
            sample = torch.randn(64, ZDIM).to(DEVICE)
            sample = text_cnn.decode(sample).cpu()
            save_image(sample.view(64, 3, 100, 100),
                       'text_decoding_results/latent_space_sample_' + str(epoch) + '.png')


    plt.figure(figsize=(15, 10))
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title("Validation loss and loss per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.show()


    torch.save(text_cnn.state_dict(), 'TextVAEModels/model_individual_sents.pt')
    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space

def display_images(img_list, loc, cols=3):
    """
    Display images in img_list
    """

    num_images = len(img_list)
    num_rows = int(math.ceil(num_images/cols))


    f, axarr = plt.subplots(num_rows, cols)

    [axi.set_axis_off() for axi in axarr.ravel()]
    for idx, images in enumerate(img_list):
        image = images[0]
        image_file_name = images[1]
        if cols > 1:
            i = int(idx / cols)
            j = int(idx % cols)
            axarr[i, j].imshow(image)
            axarr[i, j].set_title(image_file_name)
        else:
            axarr[idx].imshow(image)
            axarr[idx].set_title(image_file_name)

    plt.savefig(loc)
    plt.close(f)

fruit_name_to_desc = {}
for fruit in os.listdir('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/TextDescriptions'):
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/TextDescriptions/'
              + fruit + '/description.txt', 'r') as f:
        fruit_name_to_desc[fruit] = ''.join(f.readlines())

def eval_text_decoding(model, discriminator):
    model.eval()
    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}

    train_fruit_names = [x for x in rev_classes.keys() if x not in TEST_FRUITS]
    train_fruit_names = list(map(lambda x: rev_classes[x], train_fruit_names))

    random.shuffle(train_fruit_names)
    train_fruit_names = ['Tamarillo', 'Apple Golden 2', 'Pepper Green', 'Quince', 'Pepper Yellow', 'Apple Granny Smith', 'Cherry 2', 'Apple Braeburn', 'Mangostan']
    images = []
    for name in train_fruit_names:
        print('Generating instance of {}'.format(name))
        print('Description: {}'.format(fruit_name_to_desc[name].strip()))
        instn = generate_instance(name).to(DEVICE)
        output_img, mu, logvar = model(instn)
        img = classifier.imshow(output_img[0].cpu().detach())
        images.append((img, name))
        classification = discriminator(output_img)
        top_k = classification.topk(90)[1][0]
        rank = 90
        for idx in range(len(top_k)):
            if top_k[idx].item() == train_set.class_to_idx[name]:
                rank = idx + 1
        print('Rank of true label: {}'.format(rank))
        _, predicted = torch.max(classification, 1)
        predicted = rev_classes[predicted[0].item()]
        print('Classifier predicted {}\n\n'.format(predicted))
    display_images(images, '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/IndividualSentences/train_fruits.png')
    test_fruit_names = [x for x in rev_classes.keys() if x in TEST_FRUITS]
    test_fruit_names = list(map(lambda x: rev_classes[x], test_fruit_names))

    images = []
    for name in test_fruit_names:
        print('Generating instance of {}'.format(name))
        print('Description: {}'.format(fruit_name_to_desc[name].strip()))
        instn = generate_instance(name).to(DEVICE)
        output_img, mu, logvar = model(instn)
        img = classifier.imshow(output_img[0].cpu().detach())
        images.append((img, name))
        classification = discriminator(output_img)
        top_k = classification.topk(90)[1][0]
        rank = 90
        for idx in range(len(top_k)):
            if top_k[idx].item() == train_set.class_to_idx[name]:
                rank = idx + 1
        print('Rank of true label: {}'.format(rank))
        _, predicted = torch.max(classification, 1)
        predicted = rev_classes[predicted[0].item()]
        print('Classifier predicted {}\n\n'.format(predicted))
    display_images(images, '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/IndividualSentences/test_fruits.png')


def run_train():
    text_cnn, discriminator = load_model()
    text_cnn = update_params_model(text_cnn, VAE=True)
    text_cnn = update_params_model(text_cnn, VAE=False)


    train_loop(text_cnn, discriminator)


def interpolate(model, descriptions, path):
    text_cnn = TextCNN()
    text_cnn.load_state_dict(torch.load(model))
    text_cnn.to(DEVICE)
    text_cnn.eval()

    descs = generate_text_instances(descriptions).to(DEVICE)

    mus, logvars = text_cnn.encode(descs)

    mu0 = mus[0].view(1, 256)
    mu1 = mus[1].view(1, 256)

    logvar0 = logvars[0].view(1, 256)
    logvar1 = logvars[1].view(1, 256)

    images = []
    for i in range(0, 6):
        mu = i/5 * mu0 + (1-i/5)*mu1
        logvar = i/5 * logvar0 + (1-i/5)*logvar1
        img = text_cnn.decode(text_cnn.reparameterize(mu, logvar))[0]
        img = classifier.imshow(img.cpu().detach())
        images.append((img, '{}% {}, {}%, {}'.format(int(100*i/5),
                                                     descriptions[0],
                                                     int(100*(1-i/5)),
                                                     descriptions[1])))
    display_images(images, path, cols=1)


def evaluate(model):
    text_cnn = TextCNN()
    text_cnn.load_state_dict(torch.load(model))
    text_cnn.to(DEVICE)
    text_cnn.eval()

    DISCRIMNATOR_PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net_30_epochs.pth'

    discriminator = classifier.Net()
    discriminator.load_state_dict(torch.load(DISCRIMNATOR_PATH))
    discriminator.to(DEVICE)
    discriminator.eval()

    eval_text_decoding(text_cnn, discriminator)

def evaluate_novel_descriptions(descriptions, model):
    text_cnn = TextCNN()
    text_cnn.load_state_dict(torch.load(model))
    text_cnn.to(DEVICE)
    text_cnn.eval()

    DISCRIMNATOR_PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net_30_epochs.pth'

    discriminator = classifier.Net()
    discriminator.load_state_dict(torch.load(DISCRIMNATOR_PATH))
    discriminator.to(DEVICE)
    discriminator.eval()

    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}

    images = []
    embeds = generate_text_instances(descriptions)
    embeds = embeds.to(DEVICE)
    imgs, _, _ = text_cnn(embeds)
    classifications = discriminator(imgs)
    for idx, output_img in enumerate(imgs):
        img = classifier.imshow(output_img.cpu().detach())
        images.append((img, descriptions[idx]))
        top_k = classifications[idx].topk(90)[1]
        print('Description {}'.format(descriptions[idx]))
        for idx in range(3):
            print('Prediction number {}: {}'.format(idx+1, rev_classes[top_k[idx].item()]))
        print('\n')
    display_images(images, '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/IndividualSentences/novel_descriptions.png', cols=1)

def evaluate_repl(model):
    import warnings
    warnings.filterwarnings("ignore")
    text_cnn = TextCNN()
    text_cnn.load_state_dict(torch.load(model))
    text_cnn.to(DEVICE)
    text_cnn.eval()

    DISCRIMNATOR_PATH = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/discriminator/result/fruits_net_30_epochs.pth'

    discriminator = classifier.Net()
    discriminator.load_state_dict(torch.load(DISCRIMNATOR_PATH))
    discriminator.to(DEVICE)
    discriminator.eval()

    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}

    images = []
    counter = 0
    while True:
        desc = input('Type in a description of a fruit and press enter. Press q to quit.')
        if desc == 'q':
            break
        else:
            embeds = generate_text_instances([desc])
            embeds = embeds.to(DEVICE)
            imgs, _, _ = text_cnn(embeds)
            classifications = discriminator(imgs)
            for idx, output_img in enumerate(imgs):
                img = classifier.imshow(output_img.cpu().detach())
                top = classifications[idx].topk(90)[1][0].item()
                print('The classifier thought your fruit looked like a {}'.format(rev_classes[top]))
                f, axarr = plt.subplots(1, 1)
                axarr.axis('off')
                axarr.imshow(img)
                axarr.set_title(desc)
                plt.savefig('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/FuglyFruits/fugly_fruit_{}.png'.format(counter))
                plt.show()
                counter += 1


if __name__ == '__main__':
    # run_train()
    model = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEModels/model_test_loss_opt_individual_sents.pt'
    # interpolate(model, ('red', 'purple'), '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/WithRandomness/interpolations_red_purple.png')
    # interpolate(model, ('long', 'flat'), '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/WithRandomness/interpolations_long_flat.png')

    # interpolate(model, ('long', 'round'))
    evaluate(model)
    evaluate_novel_descriptions(['red',
                                 'yellow long smooth',
                                 'shiny purple round',
                                 'small spiky spherical',
                                 'coarse spotted white',
                                 'oblong light',
                                 'purple'],
                                model)
    # evaluate_repl(model)