from Text2PicVAE import load_model, fruit_name_to_desc, train_set, DEVICE, generate_instance, update_params_model, display_images
import torch
from machine_learning.discriminator import classifier
from TextEncoder import TextEncoder, TEST_FRUITS
from ConvNetVAEFruits import VAE_CNN
import matplotlib.pyplot as plt

def eval_text_decoding(model, discriminator, enc):
    model.eval()
    rev_classes = {v: k for k, v in train_set.class_to_idx.items()}

    train_fruit_names = [x for x in rev_classes.keys() if x in TEST_FRUITS]
    train_fruit_names = list(map(lambda x: rev_classes[x], train_fruit_names))
    images = []
    for name in train_fruit_names:
        print('Generating instance of {}'.format(name))
        print('Description: {}'.format(fruit_name_to_desc[name]))
        instn = generate_instance(name).to(DEVICE)
        output_img, mu, logvar = model(instn)
        encoding = enc(instn)
        # print('img: {}'.format((str(output_img))))
        # print('Mu: {}'.format(str(mu)))
        # print('logvar: {}'.format(str(logvar)))
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
    display_images(images, '/hdd/home/Documents/Research/DecodingDefinitionsToObjects/Results/Pretraining/test_fruits.png')

def interpolate():
    model = VAE_CNN()
    model.load_state_dict(torch.load('VAEModels/model.pt'))
    model.to(DEVICE)


if __name__ == "__main__":
    model, discrim = load_model()
    enc = TextEncoder()
    enc.load_state_dict(torch.load('TextEnc/BestResults/model_short_embed.pt'))
    enc.to(DEVICE)
    enc.eval()

    model = update_params_model(model, VAE=True)
    model = update_params_model(model, VAE=False)

    eval_text_decoding(model, discrim, enc)