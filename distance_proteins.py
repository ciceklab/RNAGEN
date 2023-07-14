import numpy as np
from nltk import ngrams
import pdb
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str, required=False, default='./input/proteins.csv')
parser.add_argument('-t', type=str, required=False, default='./data/model/trained_gan.ckpt.meta')
parser.add_argument('-lr', type=int, required=False ,default=1)
parser.add_argument('-gpu', type=str, required=False ,default='-1')

args = parser.parse_args()


def convertTuple(tup):
    mystr = ''.join(tup)
    return mystr
#threegram embeddings by prot2vec

three_gram_embeddings = np.loadtxt("./data/protVec_100d_3grams.csv",dtype=str)
threegramnames = three_gram_embeddings[:,0]
threegramnames = [x[1:] for x in threegramnames]
threegram_vecs = three_gram_embeddings[:,1:]
for item in threegram_vecs:
    item[-1] = item[-1][:-1]
threegram_vecs = threegram_vecs.astype(np.float32)

embedding_dict = {}
for name, n in zip(threegramnames, range(len(threegramnames))):
    embedding_dict[name] = threegram_vecs[n]


def seq2vec(seq):
    vec1 = np.zeros(100)
    threegrams = ngrams(seq, 3)
    threegrams = [convertTuple(grams) for grams in threegrams]
    for threegram in threegrams:
        prot2vec3gram = embedding_dict[threegram]
        vec1 = vec1 + prot2vec3gram
    return vec1

if __name__ == '__main__':

    df = pd.read_csv(args.i)

    protein_names = df['protein_name'].to_numpy()
    protein_sequences = df['seq'].to_numpy()

    #Target protein
    target_protein = protein_names[0]
    target_seq = protein_sequences[0]
    target_vec = seq2vec(target_seq)

    for i in range(len(protein_sequences)-1):
        prot_seq = protein_sequences[i+1]
        prot_name = protein_names[i+1]
        prot_vec = seq2vec(prot_seq)
        print(f"{target_protein} distance to {prot_name}: ",np.linalg.norm(target_vec - prot_vec))

