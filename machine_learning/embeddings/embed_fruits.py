import pickle
import os
import nltk.tokenize as tokenize
import numpy as np


def get_tokens_and_bigrams(text):
    tokens = list(map(lambda x: x.lower(), tokenize.word_tokenize(text)))


    unique_toks = {}
    bigrams = {}

    for token in tokens:
        if token in unique_toks:
            unique_toks[token] = unique_toks[token] + 1
        else:
            unique_toks[token] = 1

    for i in range(len(tokens) - 1):
        bigram = tokens[i] + ' ' + tokens[i + 1]
        if bigram in bigrams:
            bigrams[bigram] = bigrams[bigram] + 1
        else:
            bigrams[bigram] = 1


    return unique_toks, bigrams

FILE_LOC = os.getcwd().split('DecodingDefinitionsToObjects')[0] + 'DecodingDefinitionsToObjects/'
with open(FILE_LOC + 'Data/Fruits/fruit_data/Embeddings/BoW/longer_words.txt', 'r') as f:
    word_list = set(map(lambda x: x.strip(), f.readlines()))

def bag_of_words():
    with open(FILE_LOC + '/Data/Fruits/fruit_data/AllTextDescriptions/description.txt', 'r') as f:
        lines = f.readlines()

        all_toks, all_bigrams = get_tokens_and_bigrams('\n'.join(lines))



        gram_to_idf = {}
        for line in lines:
            line_toks, line_bigrams = get_tokens_and_bigrams(line)
            for tok in line_toks.keys():
                if tok in gram_to_idf:
                    gram_to_idf[tok] = gram_to_idf[tok] + 1
                else:
                    gram_to_idf[tok] = 1
            # for tok in line_bigrams.keys():
            #     if tok in gram_to_idf:
            #         gram_to_idf[tok] = gram_to_idf[tok] + 1
            #     else:
            #         gram_to_idf[tok] = 1
        gram_idf_dup = {}
        with open(FILE_LOC + '/Data/Fruits/fruit_data/Embeddings/BoW/all_words.txt', 'w') as f:
            for gram in gram_to_idf:
                if gram_to_idf[gram] < 4:
                    if len(gram.split(' ')) == 2:
                        del all_bigrams[gram]
                    else:
                        del all_toks[gram]
                else:
                    f.write(gram + '\n')
                    if gram in word_list:
                        gram_idf_dup[gram] = gram_to_idf[gram]



        i = 0
        gram_to_idx = {}
        for token in gram_idf_dup.keys():
            gram_to_idx[token] = i
            i += 1

        return gram_to_idx, gram_idf_dup



GRAM_TO_IDX, GRAM_TO_DOC_COUNT = bag_of_words()



N = 90



def embed_bow(text):
    all_toks, all_bigrams = get_tokens_and_bigrams(text)
    all_toks.update(all_bigrams)
    max_val = max(all_toks.values())
    vector = np.zeros(len(GRAM_TO_IDX))
    for gram in all_toks:
        if gram in GRAM_TO_IDX and gram in word_list:
            idx = GRAM_TO_IDX[gram]
            tf = 0.5 + (0.5 * all_toks[gram]) / max_val
            idf = np.log(N / GRAM_TO_DOC_COUNT[gram])
            vector[idx] = tf * idf
    return vector

def make_bow_embedings():
    text_path = 'Data/Fruits/fruit_data/TextDescriptions'
    classes = {}
    for folder in os.listdir(text_path):
        class_name = folder
        with open(text_path + '/' + folder + '/description.txt', 'r') as desc:
            vector = embed_bow('\n'.join(desc.readlines()))
            classes[class_name] = vector

    path = 'Data/Fruits/fruit_data/Embeddings/BoW/embeddings_short.pkl'
    with open(path, 'wb+') as f:
        pickle.dump(classes, f)
    print('created embeddings at {}'.format(path))

if __name__ == "__main__":
    make_bow_embedings()