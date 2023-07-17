import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

trained_gan_path = "./../data/model/trained_gan.ckpt.meta"

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

'''
load the trained WGAN model
'''
session = K.get_session()
gen_handler = tf.train.import_meta_graph(trained_gan_path, import_scope="generator")
gen_handler.restore(session, trained_gan_path[:-5])

latents = tf.get_collection('latents')[0]
gen_output = tf.get_collection('outputs')[0]
batch_size, latent_dim = session.run(tf.shape(latents))
latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]

def probs_to_chars(genseqs):
    argmax = np.argmax(genseqs,2)
    result = []
    for line in argmax:
        result.append("".join(rev_rna_vocab[d] for d in line))
    result = [a.replace("*","") for a in result]
    return result

def one_hot_encode(seq, SEQ_LEN=32):
    mapping = dict(zip("ACGU", range(4)))    
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.ones(4)/4] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(4)[seq2] , extra])
    return np.eye(4)[seq2]




noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=1e-5)


s = session.run(tf.shape(latents))

seqs = []
max_iters=300
for opt_iter in tqdm(range(max_iters)):
    noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=1e-5)
    start_noise = np.random.normal(size=[batch_size, latent_dim])
    session.run(tf.assign(latent_vars, start_noise))
    generated_sequences = session.run(gen_output)
    generated_sequences = probs_to_chars(generated_sequences)

    for seq in generated_sequences:
        if seq not in seqs:
            seqs.append(seq)


with open('generated.txt', 'w') as f:
    for item in seqs:
        f.write("%s\n" % item)
