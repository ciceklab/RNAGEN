import numpy as np
import pandas as pd
import pdb
import tensorflow.compat.v1 as tf
import sys
import lib
import socket
import datetime
import os
import argparse
'''
This script is a part of project RNAGEN (piRNA's). The script is responsible of training of the generator 
which learns to generate realistic piRNA's for homo-sapiens. This script is an essential part of the project since
the underlying data distribution is captured through WGAN with gradient penalty architecture. On top of the 
generated piRNA sequences, that looks real, an optimization via activation maximization will be applied jointly 
(i.e., many classifiers on piRNAs, potentially cancer related research) to have realistic piRNA sequences with desired
properties. 

This architecture is based on WGAN with Gradient Penalty method. The dataset used to train the GAN here has 50397 samples in total
with minimum rna-seq length 26 and maximum 32. The mean sequence length is 28.63 with an std of 1.742. The data used is obtained from
DASHR project (hg38). 
CicekLab 2023, Furkan Ozden
'''

'''
Data loading and pre-processing.
'''
tf.disable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str, required=False, default='./data/DASHR2_GEO_hg38_sequenceTable_export.csv')
parser.add_argument('-n', type=int, required=False, default=200000)
parser.add_argument('-lr', type=int, required=False ,default=4)
parser.add_argument('-gpu', type=str, required=False ,default='-1')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", "gan_test", stamp)
    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

""" Read the input data """
data_path = args.i
data = pd.read_csv(data_path)
piRNAdf = data.loc[data['rnaClass'] == 'piRNA']
piRNAarr = piRNAdf.values

rnaids = piRNAarr[:,1]
chrs = piRNAarr[:,2]
chrstart = piRNAarr[:,3]
chrend = piRNAarr[:,4]
strand = piRNAarr[:,5]
lens = piRNAarr[:,6]
sequences = piRNAarr[:,7]

sequences = [x.upper() for x in sequences]

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def one_hot_encode(seq, SEQ_LEN=32):
    mapping = dict(zip("ACGU*", range(5)))    
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(5)[seq2] , extra])
    return np.eye(5)[seq2]

ohe_sequences = np.asarray([one_hot_encode(x) for x in sequences])

BATCH_SIZE = 64 # Batch size
ITERS = args.n # How many iterations to train for
SEQ_LEN = 32 # Sequence length in characters
DIM = 25 # Model dimensionality.
CRITIC_ITERS = 5 # How many critic iterations per generator iteration. 
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load.
LR = np.exp(-args.lr) # The learning rate
'''
Set seed for reproducibility
'''
seed = 35
np.random.seed(seed)
tf.set_random_seed(seed)

logdir, checkpoint_baseline = log(samples_dir=True)

'''
Build GAN
'''
model_type = "resnet"
data_enc_dim = 5
data_size = SEQ_LEN * data_enc_dim
generator_layers = 5
disc_layers = 10
lmbda = 10. #lipschitz penalty hyperparameter.

latent_vars = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, DIM], seed=seed), name='latent_vars')

with tf.variable_scope("Generator", reuse=None) as scope:
  if model_type=="mlp":
    gen_data = lib.models.mlp_generator(latent_vars, dim=DIM, input_size=DIM, output_size=data_size, num_layers=generator_layers)
  elif model_type=="resnet":
    gen_data = lib.models.resnet_generator(latent_vars, DIM, SEQ_LEN, data_enc_dim, False)
  gen_vars = lib.get_vars(scope)

if model_type=="mlp":
  real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN])
  eps = tf.random_uniform([BATCH_SIZE, 1])
elif model_type=="resnet":
  real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN, data_enc_dim])
  eps = tf.random_uniform([BATCH_SIZE, 1, 1])
interp = eps * real_data + (1 - eps) * gen_data

with tf.variable_scope("Discriminator", reuse=None) as scope:
  if model_type=="mlp":
    gen_score = lib.models.discriminator(gen_data, dim=DIM, input_size=data_size, num_layers=disc_layers)
  elif model_type=="resnet":
    gen_score = lib.models.resnet_discriminator(gen_data, DIM, SEQ_LEN, data_enc_dim, res_layers=disc_layers)
  disc_vars = lib.get_vars(scope)
with tf.variable_scope("Discriminator", reuse=True) as scope:
  if model_type=="mlp":
    real_score = lib.models.mlp_discriminator(real_data, dim=DIM, input_size=data_size, num_layers=disc_layers)
    interp_score = lib.models.mlp_discriminator(interp, dim=DIM, input_size=data_size, num_layers=disc_layers)
  elif model_type=="resnet":
    real_score = lib.models.resnet_discriminator(real_data, DIM, SEQ_LEN, data_enc_dim, res_layers=disc_layers)
    interp_score = lib.models.resnet_discriminator(interp, DIM, SEQ_LEN, data_enc_dim, res_layers=disc_layers)


'''
Cost function with gradient penalty. 
'''

gen_cost = - tf.reduce_mean(gen_score)
disc_diff = tf.reduce_mean(gen_score) - tf.reduce_mean(real_score)
# gradient penalty
grads = tf.gradients(interp_score, interp)[0]
grad_norms = tf.norm(grads, axis=[1,2]) # might need extra term for numerical stability of SGD
grad_penalty = lmbda * tf.reduce_mean((grad_norms - 1.) ** 2)
disc_cost = disc_diff + grad_penalty

gen_optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5, beta2=0.9, name='gen_optimizer') #Note: Adam optimizer requires fixed shape
disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5, beta2=0.9, name='disc_optimizer')
gen_train_op = gen_optimizer.minimize(gen_cost, var_list=gen_vars)
disc_train_op = disc_optimizer.minimize(disc_cost, var_list=disc_vars)
tf.add_to_collection('latents', latent_vars)
tf.add_to_collection('outputs', gen_data)

session = tf.Session()
session.run(tf.global_variables_initializer())


'''
load data
'''

validate = True

data = ohe_sequences
if validate:
    split = len(data) // 8
    train_data = data[:split]
    valid_data = data[split:]
    if len(train_data) == 1: train_data = train_data[0]
    if len(valid_data) == 1: valid_data = valid_data[0]
else:
    train_data = data


def feed(data, batch_size=BATCH_SIZE, reuse=True):
    num_batches = len(data) // batch_size
    if model_type=="mlp":
        reshaped_data = np.reshape(data, [data.shape[0], -1])
    elif model_type=="resnet":
        reshaped_data = data
    while True:
        for ctr in range(num_batches):
            yield reshaped_data[ctr * batch_size : (ctr + 1) * batch_size]
        if not reuse and ctr == num_batches - 1:
            yield None

train_seqs = feed(train_data)
valid_seqs = feed(valid_data, reuse=False)

saver = tf.train.Saver()

print("Training GAN")
print("================================================")
train_iters = ITERS
disc_iters = 5
validation_iters = 10
checkpoint_iters = 1000
checkpoint = None
fixed_latents = np.random.normal(size=[BATCH_SIZE, DIM])
train_cost = []
train_counts = []
valid_cost = []
valid_dcost = []
valid_counts = []
for idx in range(train_iters):
  true_count = idx + 1 + checkpoint_baseline
  train_counts.append(true_count)
  # train generator
  if idx > 0:
    noise = np.random.normal(size=[BATCH_SIZE, DIM])
    _ = session.run(gen_train_op, {latent_vars: noise})
  # train discriminator "to optimality"
  for d in range(disc_iters):
    data = next(train_seqs)
    noise = np.random.normal(size=[BATCH_SIZE, DIM])
    cost, _ = session.run([disc_cost, disc_train_op], {latent_vars: noise, real_data: data})
  train_cost.append(-cost)

  if true_count % validation_iters == 0:
    #validation
    cost_vals = []
    data = next(valid_seqs)
    while data is not None:
      noise = np.random.normal(size=[BATCH_SIZE, DIM])
      score_diff = session.run(disc_diff, {latent_vars: noise, real_data: data})
      cost_vals.append(score_diff)
      data = next(valid_seqs)
    valid_cost.append(-np.mean(cost_vals))
    valid_dcost.append(-cost)
    valid_counts.append(true_count)
    name = "valid_disc_cost"
    if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
    lib.validplot(valid_counts, valid_cost, valid_dcost, logdir, name, xlabel="Iteration", ylabel="Discriminator cost")

    # log results
    print("Iteration {}: train_disc_cost={:.5f}, valid_disc_cost={:.5f}".format(true_count, cost, score_diff))
    samples = session.run(gen_data, {latent_vars: fixed_latents}).reshape([-1, SEQ_LEN, data_enc_dim])
    lib.save_samples(logdir, samples, true_count, rev_rna_vocab, annotated=False)
    name = "train_disc_cost"
    if checkpoint_baseline > 0: name += "_{}".format(checkpoint_baseline)
    lib.plot(train_counts, train_cost, logdir,name, xlabel="Iteration", ylabel="Cost")
    
  # save checkpoint
  if checkpoint_iters and true_count % checkpoint_iters == 0:
    ckpt_dir = os.path.join(logdir, "checkpoints", "checkpoint_{}".format(true_count))
    os.makedirs(ckpt_dir, exist_ok=True)
    saver.save(session, os.path.join(ckpt_dir, "trained_gan.ckpt"))

