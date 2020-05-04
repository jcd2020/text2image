import random

import matplotlib.pyplot as plt
import itertools

import torch
import os
from torch import nn, optim
from torchvision import datasets, transforms
import pickle
import numpy as np
from ConvNetVAEFruits import VAE_CNN
from machine_learning.embeddings.embed_fruits import  embed_bow

EMBED_SIZE = 255
N_SAMPLES = 50
UNIQUE_RANDOM_VECS = 400
ENC_SIZE = 512
DEVICE = torch.device("cuda")
BATCH_SIZE = 512
RANDOM_SIZE = 5
# RANDOM_SIZE = 0
TEST_FRUITS = [48, 22, 20, 65, 8]

# ## 1. Vars

batch_size = 1
epochs = 50
no_cuda = False
seed = 1
log_interval = 200
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)


device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# ## 2. Data loaders

os.chdir("/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE")



class TextEncoder(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self):
        super(TextEncoder, self).__init__()

        first_size = 100
        next_size = 512
        self.dropout = nn.Dropout(0.65)  # a dropout layer

        self.fc1_txt = nn.Sequential(nn.Linear(EMBED_SIZE+RANDOM_SIZE, next_size),
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


    def encode(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        x1 = self.fc1_txt(inputs)
        x2 = self.fc2_txt(x1)
        x3 = self.fc3_txt(x2)
        x4 = self.fc4_txt(x3)


        r1 = self.fc21_txt(x4)
        r2 = self.fc22_txt(x4)

        return torch.cat([r1, r2], dim=1)


    def forward(self, x):
        return self.encode(x)



train_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Training'
val_root = '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Test'


train_set = datasets.ImageFolder(train_root, transform=transforms.ToTensor())
train_loader_food = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size, shuffle=True, **kwargs)

val_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_root, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}

with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Embeddings/BoW/embeddings_short.pkl', 'rb') as f:
    text_embeddings = pickle.load(f)
    EMBED_SIZE = len(text_embeddings['Kiwi'])
    print("Embedding size: {}".format(EMBED_SIZE))


with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Data/Fruits/fruit_data/Embeddings/BoW/individual_sentence_embeddings.pkl', 'rb') as f:
    individual_sentence_embeddings = pickle.load(f)



def load_model():
    text_cnn = TextEncoder()
    text_cnn.to(DEVICE)

    return text_cnn

np.random.seed(seed)
random_appended_vectors = []
# CHOOSE some fixed random vectors, in order to ensure reuse of vectors
for i in range(UNIQUE_RANDOM_VECS):
    random_appended_vectors.append(np.random.rand(1, RANDOM_SIZE))


def generate_instances(labels, idx_to_class):
    ret = []
    for label in labels:
        fruit_name = idx_to_class[label.item()]
        ret.append(generate_instance(fruit_name))
    return torch.cat(ret)


def generate_instance(fruit_name):
    rand = random.choice(random_appended_vectors)
    arr = np.asarray([text_embeddings[fruit_name]])
    arr = np.concatenate((rand, arr), axis=1)
    return torch.from_numpy(arr.astype(np.float32))


def generate_text_instances(descriptions):
    ret = []
    for desc in descriptions:
        ret.append(generate_text_instance(desc))
    return torch.cat(ret)


def generate_text_instance(description):
    embedding = embed_bow(description)
    rand = random.choice(random_appended_vectors)
    arr = np.asarray([embedding])
    arr = np.concatenate((rand, arr), axis=1)
    return torch.from_numpy(arr.astype(np.float32))


def make_data(conv):
    with torch.no_grad():
        train_data = []
        batch = []
        i = 0
        idx = 0
        for img, label in train_loader_food:
            if i % BATCH_SIZE == 0 and i != 0:
                unzipped_batch = list(zip(*batch))
                batch_images = torch.cat(list(unzipped_batch[0]))
                batch_encs = torch.cat(list(unzipped_batch[1]))
                batch_labels = torch.cat(list(unzipped_batch[2]))

                train_data.append((batch_images, batch_encs, batch_labels))
                batch = []

            img = img.to(DEVICE)
            r1, r2 = conv.encode(img)
            conv.zero_grad()
            enc = torch.cat([r1, r2], dim=1)
            enc = enc.to(torch.device('cpu'))
            img = img.to(torch.device('cpu'))
            batch.append((img, enc, label))
            i += 1
            idx += 1
            if idx % 1000 == 0:
                print(idx)
        test_data = []
        batch = []
        i = 0
        idx = 0
        for img, label in val_loader_food:
            if i % BATCH_SIZE == 0 and i != 0:
                unzipped_batch = list(zip(*batch))
                batch_images = torch.cat(list(unzipped_batch[0]))
                batch_encs = torch.cat(list(unzipped_batch[1]))
                batch_labels = torch.cat(list(unzipped_batch[2]))

                test_data.append((batch_images, batch_encs, batch_labels))
                batch = []
            img = img.to(DEVICE)
            r1, r2 = conv.encode(img)
            conv.zero_grad()
            enc = torch.cat([r1, r2], dim=1)
            enc = enc.to(torch.device('cpu'))
            img = img.to(torch.device('cpu'))
            batch.append((img, enc, label))

            i += 1
            idx += 1
            if idx % 1000 == 0:
                print(idx)
        return train_data, test_data

def train(model, train_data, optimizer, loss_func, train_losses, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (_, input_vecs, target_encs, _) in enumerate(train_data):
        input_vecs = input_vecs.to(DEVICE)
        target_encs = target_encs.to(DEVICE)
        model.zero_grad()
        predicted_enc = model(input_vecs)
        loss, var_loss = loss_func(predicted_enc, target_encs)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(target_encs), len(train_data)*BATCH_SIZE,
                       100. * batch_idx / len(train_data),
                       loss.item() / len(target_encs)))

    print('====> Epoch: {} Average loss: {:.8f}'.format(
        epoch, train_loss / (len(train_data) * BATCH_SIZE)))
    train_losses.append(train_loss / len(train_data) * BATCH_SIZE)


def test(model, test_data, loss_func, test_losses, epoch):
    model.eval()
    test_loss = 0
    var_loss_sum = 0
    with torch.no_grad():
        for i,(_, input_vecs, target_encs, _)in enumerate(test_data):
            input_vecs = input_vecs.to(DEVICE)
            target_encs = target_encs.to(DEVICE)

            predicted_enc = model(input_vecs)
            loss, var_loss = loss_func(predicted_enc, target_encs)
            var_loss_sum += var_loss
            test_loss += loss

    test_loss /= (len(test_data) * BATCH_SIZE)
    var_loss_sum /= (len(test_data) * BATCH_SIZE)
    print('====> Test set loss: {:.8f} and var loss {:8f}'.format(test_loss, var_loss_sum))
    test_losses.append(test_loss)
    return test_loss



class customEncLoss(nn.Module):
    def __init__(self):
        super(customEncLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, predicted_encs, target_encs):
        std1 = target_encs.std(dim=0)
        std2 = predicted_encs.std(dim=0)

        var_loss = self.l1(std1, std2)

        mse_loss = self.mse(predicted_encs, target_encs)

        return mse_loss, var_loss

def train_loop(train_data, test_data, text_cnn):
    epochs = 500
    learning_rate = 5e-5
    optimizer = optim.Adam(text_cnn.parameters(), lr=learning_rate)

    loss_func = customEncLoss()

    # ## Train

    val_losses = []
    train_losses = []

    min_test_loss = 100000
    for epoch in range(1, epochs + 1):
        train(text_cnn, train_data, optimizer, loss_func, train_losses, epoch)
        loss = test(text_cnn, test_data, loss_func, val_losses, epoch)
        if loss < min_test_loss:
            min_test_loss = loss
            print('New min test loss {}'.format(min_test_loss))
            torch.save(text_cnn.state_dict(), 'TextEnc/model_individual_sentences.pt')

    plt.figure(figsize=(15, 10))
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title("Validation loss and loss per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.show()


def hold_out_random_classes(data0, data1):

    train_data = []
    train_batch = []
    test_data = []
    test_batch = []

    data0.extend(data1)
    for batch_idx, (img, enc, labels) in enumerate(data0):
        for i in range(len(labels)):
            img_i = img[i]
            enc_i = enc[i]
            label_i = labels[i]
            if label_i.item() not in TEST_FRUITS:
                train_batch.append((img_i, enc_i, label_i))
                if len(train_batch) == BATCH_SIZE:
                    unzipped_batch = list(zip(*train_batch))
                    batch_imgs = list(unzipped_batch[0])
                    batch_encs = list(unzipped_batch[1])
                    batch_labels = list(unzipped_batch[2])

                    batch_imgs = torch.stack(batch_imgs)
                    batch_encs = torch.stack(batch_encs)
                    batch_labels = torch.stack(batch_labels)
                    train_data.append((batch_imgs, batch_encs, batch_labels))
                    train_batch = []
            else:
                test_batch.append((img_i, enc_i, label_i))
                if len(test_batch) == BATCH_SIZE:
                    unzipped_batch = list(zip(*test_batch))
                    batch_imgs = list(unzipped_batch[0])
                    batch_encs = list(unzipped_batch[1])
                    batch_labels = list(unzipped_batch[2])

                    batch_imgs = torch.stack(batch_imgs)
                    batch_encs = torch.stack(batch_encs)
                    batch_labels = torch.stack(batch_labels)
                    test_data.append((batch_imgs, batch_encs, batch_labels))
                    test_batch = []

    return train_data, test_data, TEST_FRUITS


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_individual_embeddings(fruit_name, k):
    rand = random.choice(random_appended_vectors)[0]
    lst = list(chunks(individual_sentence_embeddings[fruit_name], k))
    res = []
    for entries in lst:
        arr = np.sum(entries, axis=0)
        arr = np.concatenate((rand, arr), axis=0)
        res.append(arr)
    return res


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def save_data(data, f, individual_sentences=False, k=1):
    data_new = []

    for batch_idx, (imgs, target_encs, labels) in enumerate(data):
        new_img_batch = []
        new_enc_batch = []
        new_label_batch = []
        new_sentence_embedding_batch = []
        if not individual_sentences:
            input_vecs = generate_instances(labels, idx_to_class)
            batch = (imgs, input_vecs, target_encs, labels)
            data_new.append(batch)
        else:
            for idx in range(len(imgs)):
                img = imgs[idx]
                enc = target_encs[idx]
                label = labels[idx]
                sentence_embeddings = generate_individual_embeddings(idx_to_class[label.item()], k)
                for embed  in sentence_embeddings:
                    new_img_batch.append(img)
                    new_label_batch.append(label)
                    new_enc_batch.append(enc)
                    new_sentence_embedding_batch.append(torch.from_numpy(embed.astype(np.float32)))

            new_img_batch = list(map(lambda x: torch.stack(x), list(chunks(new_img_batch, BATCH_SIZE))))[:-1]
            new_sentence_embedding_batch = list(map(lambda x: torch.stack(x), list(chunks(new_sentence_embedding_batch, BATCH_SIZE))))[:-1]
            new_enc_batch = list(map(lambda x: torch.stack(x), list(chunks(new_enc_batch, BATCH_SIZE))))[:-1]
            new_label_batch = list(map(lambda x: torch.stack(x), list(chunks(new_label_batch, BATCH_SIZE))))[:-1]

            bs = list(zip(new_img_batch, new_sentence_embedding_batch, new_enc_batch, new_label_batch))
            data_new.extend(bs)
    print('dumping data')
    pickle.dump(data_new, f)
    return data_new

def run_train():
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/train_short.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/test_short.pkl', 'rb') as f:
        test_data = pickle.load(f)
    train_data, test_data, held_out_indices = hold_out_random_classes(train_data, test_data)
    print(held_out_indices)
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/train_individual_sentences.pkl', 'wb+') as f:
        train_data = save_data(train_data, f, individual_sentences=True, k=3)

    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/test_individual_sentences.pkl', 'wb+') as f:
        test_data = save_data(test_data, f, individual_sentences=False)

    text_cnn = load_model()
    print(count_parameters(text_cnn))
    train_loop(train_data, test_data, text_cnn)


def run_make_data():
    model = VAE_CNN()
    model.load_state_dict(torch.load('VAEModels/model.pt'))
    model.to(DEVICE)
    model.eval()
    train_data, test_data = make_data(model)
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/train_short.pkl', 'wb+') as f:
        pickle.dump(train_data, f)
    with open('/hdd/home/Documents/Research/DecodingDefinitionsToObjects/machine_learning/encoder_decoder/VAE/TextVAEDataPickles/test_short.pkl', 'wb+') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    # run_make_data()
    run_train()
