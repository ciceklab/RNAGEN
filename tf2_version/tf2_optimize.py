import numpy as np
import tensorflow as tf
import sys
import os
import pdb
from tensorflow import keras
from tensorflow.keras import backend as K
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Concatenate, LeakyReLU, SpatialDropout1D
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.special import softmax
import pandas as pd
import argparse

# TF2 - Enable eager execution by default (no need to disable)
# tf.config.run_functions_eagerly(True) 

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=False, default='./input/opt_input.csv')
parser.add_argument('-n', type=int, required=False, default=10000)
parser.add_argument('-t', type=str, required=False, default='./logs_tf2/gan_test/*/checkpoints/best_model/')
parser.add_argument('-lr', type=float, required=False, default=0.1)
parser.add_argument('-gpu', type=str, required=False, default='0')

args = parser.parse_args()

# Set GPU and disable XLA to avoid libdevice issues
if args.gpu != '-1':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
            # Disable XLA compilation to avoid libdevice issues
            tf.config.optimizer.set_jit(False)
        except (RuntimeError, ValueError, IndexError):
            print("Invalid GPU configuration, using CPU")
            
# Disable XLA globally to prevent libdevice errors
tf.config.optimizer.set_jit(False)

def convert_model(file_name, input_len, window_size=0):
    """Convert DeepBind model format to Keras model - same as original"""
    data = [x.rstrip() for x in open(file_name).read().split("\n")]
    
    reverse_complement = int(data[1].split(" = ")[1])
    num_detectors = int(data[2].split(" = ")[1])
    detector_len = int(data[3].split(" = ")[1])
    has_avg_pooling = int(data[4].split(" = ")[1])
    num_hidden = int(data[5].split(" = ")[1])
    
    if (window_size < 1):
        window_size = (int)(detector_len*1.5) #copying deepbind code
    if (window_size > input_len):
        window_size = input_len

    detectors = (np.array(
        [float(x) for x in data[6].split(" = ")[1].split(",")])
        .reshape(detector_len, 4, num_detectors))
    biases = np.array([float(x) for x in data[7].split(" = ")[1].split(",")])
    weights1 = np.array([float(x) for x in data[8].split(" = ")[1].split(",")]).reshape(
                        num_detectors*(2 if has_avg_pooling else 1),
                        (1 if num_hidden==0 else num_hidden))
    if (has_avg_pooling > 0):
        weights1 = weights1.reshape((num_detectors,2,-1))
        new_weights1 = np.zeros((2*num_detectors, weights1.shape[-1]))
        new_weights1[:num_detectors, :] = weights1[:,0,:]
        new_weights1[num_detectors:, :] = weights1[:,1,:]
        weights1 = new_weights1
    biases1 = np.array([float(x) for x in data[9].split(" = ")[1].split(",")]).reshape(
                        (1 if num_hidden==0 else num_hidden))
    if (num_hidden > 0):
        weights2 = np.array([float(x) for x in data[10].split(" = ")[1].split(",")]).reshape(
                        num_hidden,1)
        biases2 = np.array([float(x) for x in data[11].split(" = ")[1].split(",")]).reshape(
                        1)
    
    def seq_padding(x):
        return tf.pad(x,
                [[0, 0],
                 [detector_len-1, detector_len-1],
                 [0, 0]],
                mode='CONSTANT',
                name=None,
                constant_values=0.25)

    input_tensor = keras.layers.Input(shape=(input_len,4))
    padding_out_fwd = keras.layers.Lambda(seq_padding)(input_tensor)
    conv_layer = keras.layers.Conv1D(filters=num_detectors,
                                  kernel_size=detector_len,
                                  activation="relu")
    conv_out_fwd = conv_layer(padding_out_fwd)
    pool_out_fwd = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_fwd)
    if (has_avg_pooling > 0):
        gap_out_fwd = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_fwd)
        pool_out_fwd = keras.layers.Concatenate(axis=-1)([pool_out_fwd, gap_out_fwd])        
    dense1_layer = keras.layers.Dense((1 if num_hidden==0 else num_hidden))
    dense1_out_fwd = keras.layers.TimeDistributed(dense1_layer)(pool_out_fwd)
    if (num_hidden > 0):
        dense1_out_fwd = keras.layers.Activation("relu")(dense1_out_fwd)
        dense2_layer = keras.layers.Dense(1)
        dense2_out_fwd = keras.layers.TimeDistributed(dense2_layer)(dense1_out_fwd)
    
    if (reverse_complement > 0):
        padding_out_rev = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(padding_out_fwd)
        conv_out_rev = conv_layer(padding_out_rev)
        pool_out_rev = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_rev)
        if (has_avg_pooling > 0):
            gap_out_rev = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_rev)
            pool_out_rev = keras.layers.Concatenate(axis=-1)([pool_out_rev, gap_out_rev])
        dense1_out_rev = keras.layers.TimeDistributed(dense1_layer)(pool_out_rev)
        if (num_hidden > 0):
            dense1_out_rev = keras.layers.Activation("relu")(dense1_out_rev)
            dense2_out_rev = keras.layers.TimeDistributed(dense2_layer)(dense1_out_rev)
    
    cross_seq_max = keras.layers.Lambda(lambda x: tf.reduce_max(x,axis=1)[:,0],
                                        output_shape=lambda x: (None,1))
    
    if (reverse_complement > 0):
        if (num_hidden > 0):
            max_fwd = cross_seq_max(dense2_out_fwd)
            max_rev = cross_seq_max(dense2_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
        else:
            max_fwd = cross_seq_max(dense1_out_fwd)
            max_rev = cross_seq_max(dense1_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
    else:
        if (num_hidden > 0):
            output = cross_seq_max(dense2_out_fwd)
        else:
            output = cross_seq_max(dense1_out_fwd)
        
    model = keras.models.Model(inputs = [input_tensor], outputs = [output])
    model.compile(loss="mse", optimizer="adam")
    conv_layer.set_weights([detectors, biases])
    dense1_layer.set_weights([weights1, biases1])
    if (num_hidden > 0):
        dense2_layer.set_weights([weights2, biases2])
        
    return model

def onehot_encode_sequences(sequences):
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            arr[i, mapping[letter]] = 1.0
        onehot.append(arr)
    return onehot

def get_kmer(seq):
    ntarr = ("A","C","G","T")
    kmerArray = []
    kmerre = []
    rst = []
    fst = 0
    total = 0.0
    pp = 0.0
    item = 0.0

    for n in range(4):
        kmerArray.append(ntarr[n])

    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            kmerArray.append(str2)
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                kmerArray.append(str3)
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                for y in range(4):
                    str4 = str3 + ntarr[y]
                    kmerArray.append(str4)
    for i in ntarr:
        kmerre.append(i)
        for m in kmerArray:
            st = i + m
            kmerre.append(st)
    for n in range(len(kmerre)):
        item = countoverlap(seq,kmerre[n])
        total = total + item
        rst.append(item)

    sub_seq = []
    if seq.startswith("T"):
        sub_seq.append(seq[0:1])
        sub_seq.append(seq[0:2])
        sub_seq.append(seq[0:3])
        sub_seq.append(seq[0:4])
        sub_seq.append(seq[0:5])

    if seq[9:10] == "A":
        sub_seq.append(seq[9:10])
        sub_seq.append(seq[8:10])
        sub_seq.append(seq[7:10])
        sub_seq.append(seq[6:10])
        sub_seq.append(seq[5:10])
        sub_seq.append(seq[9:11])
        sub_seq.append(seq[9:12])
        sub_seq.append(seq[9:13])
        sub_seq.append(seq[9:14])

    for i in sub_seq:
        if "N" not in i:
            inx = kmerre.index(i)
            rst[inx] += 1

    for n in range(len(rst)):
        rst[n] = rst[n]/total

    return rst

def countoverlap(seq,kmer):
    return len([1 for i in range(len(seq)) if seq.startswith(kmer,i)])

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

def select_best(scores, target_scores, seqs, USE_TARGET=True):
    if USE_TARGET:
        scores = target_scores
    selected_scores = []
    selected_seqs = []
    for i in range(len(scores[0])):
        best = scores[0][i]
        target = target_scores[0][i]
        best_seq = seqs[0][i]
        for j in range(len(scores)-1):
            if scores[j+1][i] > best:
                best = scores[j+1][i]
                target = target_scores[j+1][i]
                best_seq = seqs[j+1][i]
        selected_scores.append(target)
        selected_seqs.append(best_seq)

    return selected_seqs, selected_scores

# Import the generator and discriminator classes from the training script
class ResNetGenerator(tf.keras.Model):
    def __init__(self, dim, seq_len, data_enc_dim):
        super().__init__()
        self.num_ups = 2
        self.initial_steps = seq_len // (2 ** self.num_ups)
        self.dense1 = tf.keras.layers.Dense(self.initial_steps * 4 * dim)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.reshape = tf.keras.layers.Reshape((self.initial_steps, 4 * dim))
        self.upsample = tf.keras.layers.UpSampling1D(size=2)
        self.conv1 = tf.keras.layers.Conv1DTranspose(2 * dim, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1DTranspose(dim, kernel_size=3, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv1DTranspose(data_enc_dim, kernel_size=3, padding='same', dtype='float32')

    def call(self, z, training=True):
        x = self.dense1(z)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.reshape(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        if x.shape[1] > self.initial_steps * (2 ** self.num_ups):
            x = x[:, :self.initial_steps * (2 ** self.num_ups), :]
        elif x.shape[1] < self.initial_steps * (2 ** self.num_ups):
            padding = (self.initial_steps * (2 ** self.num_ups)) - x.shape[1]
            x = tf.pad(x, [[0,0],[0,padding],[0,0]])
        return x

class ResNetDiscriminator(tf.keras.Model):
    def __init__(self, dim, seq_len, data_enc_dim, res_layers=10):
        super().__init__()
        self.dim = dim
        self.conv1 = tf.keras.layers.Conv1D(dim, 5, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(2 * dim, 5, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv1D(4 * dim, 5, strides=2, padding='same')
        self.conv4 = tf.keras.layers.Conv1D(8 * dim, 5, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, dtype='float32')

    def call(self, x, training=True):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = self.flatten(x)
        x = self.dense(x)
        return x

'''
This script is a major component of the project RNAGEN by CicekLab.
This script needs: i) Trained RNA sequences generator (WGAN), ii) DeepBind model weights for the desired protein. 
The script basically optimizes generated RNA sequences to have maximum binding scores to desired proteins. 
Updated for TF2 compatibility.
'''

if __name__ == '__main__':
    
    input_file = args.i
    df = pd.read_csv(input_file)

    protein_names = df['protein_name'].to_numpy()
    paths = df['model_id'].to_numpy()
    distances = df['dist'].to_numpy()

    protein_name = protein_names[0]
    trained_gan_path = args.t
    
    # Handle wildcard paths
    if '*' in trained_gan_path:
        import glob
        matching_paths = glob.glob(trained_gan_path)
        if not matching_paths:
            raise FileNotFoundError(f"No directories found matching pattern: {trained_gan_path}")
        
        trained_gan_path = max(matching_paths, key=os.path.getmtime)
        print(f"Resolved wildcard path to: {trained_gan_path}")
    base_deepbind_path = './deepbind_db/params/'

    # SOX Family Optimization
    deepbind_model_path = base_deepbind_path + paths[1] + '.txt'
    deepbind_model_path_2 = base_deepbind_path + paths[2] + '.txt'
    deepbind_model_path_3 = base_deepbind_path + paths[3] + '.txt'
    deepbind_model_target = base_deepbind_path + paths[0] + '.txt'
    deepbind_model_eval = base_deepbind_path + paths[4] + '.txt'
    MEASURE_TEST = False

    dist_protein_1 = float(distances[1])
    dist_protein_2 = float(distances[2])
    dist_protein_3 = float(distances[3])
    dists = [dist_protein_1, dist_protein_2, dist_protein_3]
    weights = softmax(dists)

    rna_vocab = {"A":0, "C":1, "G":2, "U":3, "*":4}
    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    # Model parameters
    BATCH_SIZE = 128
    DIM = 25
    SEQ_LEN = 32
    
    def find_best_checkpoint(base_path):
        """Find the best model checkpoint based on the training script's saving strategy."""
        
        # First, try to find the 'best_model' directory (saved during training)
        if os.path.isdir(base_path):
            best_model_path = os.path.join(base_path, "best_model")
            if os.path.exists(best_model_path):
                print(f"Found best_model directory: {best_model_path}")
                # Look for checkpoint files in best_model directory
                checkpoint_path = tf.train.latest_checkpoint(best_model_path)
                if checkpoint_path:
                    return checkpoint_path
                # If no checkpoint index, look for trained_gan files
                checkpoint_files = [f for f in os.listdir(best_model_path) 
                                  if 'trained_gan' in f and not f.endswith('.tmp')]
                if checkpoint_files:
                    # Remove file extension to get checkpoint prefix
                    checkpoint_prefix = checkpoint_files[0].split('.')[0]
                    return os.path.join(best_model_path, checkpoint_prefix)
            
            # If no best_model directory, look for checkpoints directory
            checkpoints_dir = os.path.join(base_path, "checkpoints")
            if os.path.exists(checkpoints_dir):
                print(f"Found checkpoints directory: {checkpoints_dir}")
                
                # First try to find best_model subdirectory
                best_model_subdir = os.path.join(checkpoints_dir, "best_model")
                if os.path.exists(best_model_subdir):
                    print(f"Found best_model subdirectory: {best_model_subdir}")
                    checkpoint_path = tf.train.latest_checkpoint(best_model_subdir)
                    if checkpoint_path:
                        return checkpoint_path
                    # Fallback to looking for trained_gan files
                    checkpoint_files = [f for f in os.listdir(best_model_subdir) 
                                      if 'trained_gan' in f and not f.endswith('.tmp')]
                    if checkpoint_files:
                        checkpoint_prefix = checkpoint_files[0].split('.')[0]
                        return os.path.join(best_model_subdir, checkpoint_prefix)
                
                # If no best_model, find the latest checkpoint in checkpoints directory
                print("No best_model found, looking for latest checkpoint...")
                subdirs = [d for d in os.listdir(checkpoints_dir) 
                          if os.path.isdir(os.path.join(checkpoints_dir, d)) and 'checkpoint_' in d]
                if subdirs:
                    # Sort by checkpoint number to get the latest
                    subdirs.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                    latest_checkpoint_dir = os.path.join(checkpoints_dir, subdirs[-1])
                    checkpoint_path = tf.train.latest_checkpoint(latest_checkpoint_dir)
                    if checkpoint_path:
                        return checkpoint_path
                    # Fallback to looking for trained_gan files
                    checkpoint_files = [f for f in os.listdir(latest_checkpoint_dir) 
                                      if 'trained_gan' in f and not f.endswith('.tmp')]
                    if checkpoint_files:
                        checkpoint_prefix = checkpoint_files[0].split('.')[0]
                        return os.path.join(latest_checkpoint_dir, checkpoint_prefix)
            
            checkpoint_path = tf.train.latest_checkpoint(base_path)
            if checkpoint_path:
                return checkpoint_path
            
            checkpoint_files = [f for f in os.listdir(base_path) 
                              if 'trained_gan' in f and not f.endswith('.tmp')]
            if checkpoint_files:
                checkpoint_prefix = checkpoint_files[0].split('.')[0]
                return os.path.join(base_path, checkpoint_prefix)
        
        else:
            return base_path
        
        return None
    
    # Load the trained WGAN model
    generator = ResNetGenerator(DIM, SEQ_LEN, 5)
    discriminator = ResNetDiscriminator(DIM, SEQ_LEN, 5)
    
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator
    )
    
    # Load the best model using the improved function
    def find_best_checkpoint(base_path):
        """Find the best model checkpoint based on the training script's saving strategy."""
        
        if os.path.isdir(base_path):
            best_model_path = os.path.join(base_path, "best_model")
            if os.path.exists(best_model_path):
                print(f"Found best_model directory: {best_model_path}")
                checkpoint_path = tf.train.latest_checkpoint(best_model_path)
                if checkpoint_path:
                    return checkpoint_path
                checkpoint_files = [f for f in os.listdir(best_model_path) 
                                  if 'trained_gan' in f and not f.endswith('.tmp')]
                if checkpoint_files:
                    checkpoint_prefix = checkpoint_files[0].split('.')[0]
                    return os.path.join(best_model_path, checkpoint_prefix)
            
            checkpoints_dir = os.path.join(base_path, "checkpoints")
            if os.path.exists(checkpoints_dir):
                print(f"Found checkpoints directory: {checkpoints_dir}")
                
                best_model_subdir = os.path.join(checkpoints_dir, "best_model")
                if os.path.exists(best_model_subdir):
                    print(f"Found best_model subdirectory: {best_model_subdir}")
                    checkpoint_path = tf.train.latest_checkpoint(best_model_subdir)
                    if checkpoint_path:
                        return checkpoint_path
                    checkpoint_files = [f for f in os.listdir(best_model_subdir) 
                                      if 'trained_gan' in f and not f.endswith('.tmp')]
                    if checkpoint_files:
                        checkpoint_prefix = checkpoint_files[0].split('.')[0]
                        return os.path.join(best_model_subdir, checkpoint_prefix)
                
                # If no best_model, find the latest checkpoint in checkpoints directory
                print("No best_model found, looking for latest checkpoint...")
                subdirs = [d for d in os.listdir(checkpoints_dir) 
                          if os.path.isdir(os.path.join(checkpoints_dir, d)) and 'checkpoint_' in d]
                if subdirs:
                    subdirs.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                    latest_checkpoint_dir = os.path.join(checkpoints_dir, subdirs[-1])
                    checkpoint_path = tf.train.latest_checkpoint(latest_checkpoint_dir)
                    if checkpoint_path:
                        return checkpoint_path
                    checkpoint_files = [f for f in os.listdir(latest_checkpoint_dir) 
                                      if 'trained_gan' in f and not f.endswith('.tmp')]
                    if checkpoint_files:
                        checkpoint_prefix = checkpoint_files[0].split('.')[0]
                        return os.path.join(latest_checkpoint_dir, checkpoint_prefix)
            
            # Last resort: look for any checkpoint in the base directory
            checkpoint_path = tf.train.latest_checkpoint(base_path)
            if checkpoint_path:
                return checkpoint_path
            
            # Final fallback: look for trained_gan files in base directory
            checkpoint_files = [f for f in os.listdir(base_path) 
                              if 'trained_gan' in f and not f.endswith('.tmp')]
            if checkpoint_files:
                checkpoint_prefix = checkpoint_files[0].split('.')[0]
                return os.path.join(base_path, checkpoint_prefix)
        
        else:
            # It's a file path
            return base_path
        
        return None

    checkpoint_path = find_best_checkpoint(trained_gan_path)
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"Could not find any valid checkpoint in {trained_gan_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint.restore(checkpoint_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Available files in directory:")
        if os.path.isdir(trained_gan_path):
            for root, dirs, files in os.walk(trained_gan_path):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
        raise

    # Initialize the generator with dummy input to build the model
    dummy_noise = tf.random.normal([BATCH_SIZE, DIM], dtype=tf.float32)
    _ = generator(dummy_noise, training=False)

    # Get DeepBind Models Ready
    model = convert_model(file_name=deepbind_model_path, input_len=32)
    model2 = convert_model(file_name=deepbind_model_path_2, input_len=32)
    model3 = convert_model(file_name=deepbind_model_path_3, input_len=32)
    model_target = convert_model(file_name=deepbind_model_target, input_len=32)
    model_test = convert_model(file_name=deepbind_model_eval, input_len=32)

    # WEIGHT TUNER NETWORK definition
    wtn_inps = keras.layers.Input(shape=(3,))
    batchs = K.shape(wtn_inps)[0]
    constants = np.asarray(dists).reshape((1,-1))
    k_consts = K.variable(constants)
    k_consts = K.tile(k_consts, (batchs,1))
    fixed_inp = keras.layers.Input(tensor=k_consts)
    wtn_concated_inps = keras.layers.concatenate([fixed_inp, wtn_inps])
    wtn_hidden1 = keras.layers.Dense(6, activation='relu')(wtn_concated_inps)
    wtn_hidden2 = keras.layers.Dense(12, activation='relu')(wtn_hidden1)
    wtn_out = keras.layers.Dense(3, activation='softmax')(wtn_hidden2)
    wtn_model = keras.models.Model(inputs=[wtn_inps, fixed_inp], outputs=wtn_out)

    # Initialize latent variables that we'll optimize
    latent_vars = tf.Variable(
        tf.random.normal([BATCH_SIZE, DIM], dtype=tf.float32),
        trainable=True,
        name='latent_vars'
    )


    @tf.function
    def compute_gradients_and_scores(latent_z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(latent_z)
            
            # Generate sequences
            generated_sequences = generator(latent_z, training=False)
            
            # Convert to sequences that can be fed to DeepBind models
            # Remove the last dimension (the * token) for DeepBind models
            sequences_for_deepbind = generated_sequences[:, :, :4]
            
            scores1 = model(sequences_for_deepbind)
            scores2 = model2(sequences_for_deepbind) 
            scores3 = model3(sequences_for_deepbind)
            scores_target = model_target(sequences_for_deepbind)
            
            loss1 = -tf.reduce_mean(scores1)
            loss2 = -tf.reduce_mean(scores2)
            loss3 = -tf.reduce_mean(scores3)
            loss_target = -tf.reduce_mean(scores_target)
        
        # Compute gradients
        grad1 = tape.gradient(loss1, latent_z)
        grad2 = tape.gradient(loss2, latent_z)
        grad3 = tape.gradient(loss3, latent_z)
        grad_target = tape.gradient(loss_target, latent_z)
        
        del tape
        
        return (grad1, grad2, grad3, grad_target, 
                scores1, scores2, scores3, scores_target, 
                generated_sequences)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)
    
    lambda_lat_rep = 0.02    
    cos_margin      = 0.2      
    lambda_seq_ent  = 0.02     
    softmax_tau     = 1.5      
    noise_std0      = 1e-4     
    reinit_every    = 200      
    reinit_frac     = 0.10    
    
    def hard_one_hot_from_logits(logits_5):
        idx = tf.argmax(logits_5, axis=-1)                # [B, L] int indices in 0..4
        idx_clipped = tf.clip_by_value(idx, 0, 3)         # map any '*' to {0..3}
        ohe4 = tf.one_hot(idx_clipped, depth=4, dtype=tf.float32)  # [B, L, 4]
        return ohe4, idx
    
    def decode_indices_to_strings(idx_2d):
        idx_np = idx_2d.numpy()
        return ["".join(rev_rna_vocab[int(k)] for k in row) for row in idx_np]
    
    def latent_repulsion_loss(z, margin=0.2):
        # z: [B, D]
        z_norm = tf.math.l2_normalize(z, axis=1)                         # [B, D]
        sims = tf.matmul(z_norm, z_norm, transpose_b=True)               # [B, B]
        b = tf.shape(z)[0]
        mask = tf.ones_like(sims) - tf.eye(b)                            # zero diagonal
        sims_off = sims * mask
        penalties = tf.nn.relu(sims_off - margin)
        denom = tf.cast(b*(b-1), tf.float32) + 1e-8
        return tf.reduce_sum(penalties) / denom
    
    def sequence_entropy_loss(gen_out_5, tau=1.5):
        logits4 = gen_out_5[:, :, :4]                       # drop '*' channel for entropy
        probs = tf.nn.softmax(logits4 / tau, axis=-1)       # [B, L, 4]
        ent = -tf.reduce_sum(probs * tf.math.log(tf.clip_by_value(probs, 1e-7, 1.0)), axis=-1)  # [B, L]
        return -tf.reduce_mean(ent)
    
    bind_scores_list, bind_scores_list2, bind_scores_list_target = [], [], []
    bind_scores_means, bind_scores_means2, bind_scores_means_target = [], [], []
    bind_scores_means_total = []
    scores_history, target_scores_history, sequences_history = [], [], []
    sequences_list = []
    max_iters = args.n
    
    print("Starting diversity-augmented optimization …")
    for opt_iter in tqdm(range(max_iters)):
        with tf.GradientTape() as tape_gen:
            tape_gen.watch(latent_vars)
            gen_out = generator(latent_vars, training=False)  # [B, L, 5]
    
        ohe4, idx_discrete = hard_one_hot_from_logits(gen_out)
    
        # Compute DeepBind losses on hard inputs (non-target only for update)
        with tf.GradientTape() as t1:
            t1.watch(ohe4)
            loss1 = -tf.reduce_mean(model(ohe4))
        g_in1 = t1.gradient(loss1, ohe4)
    
        with tf.GradientTape() as t2:
            t2.watch(ohe4)
            loss2 = -tf.reduce_mean(model2(ohe4))
        g_in2 = t2.gradient(loss2, ohe4)
    
        with tf.GradientTape() as t3:
            t3.watch(ohe4)
            loss3 = -tf.reduce_mean(model3(ohe4))
        g_in3 = t3.gradient(loss3, ohe4)
    
        target_scores_batch = model_target(ohe4)
    
        g_in = weights[0] * g_in1 + weights[1] * g_in2 + weights[2] * g_in3  # [B, L, 4]
    
        with tf.GradientTape() as t_div:
            t_div.watch(gen_out)
            loss_seq_ent = sequence_entropy_loss(gen_out, tau=softmax_tau)
        g_out_div = t_div.gradient(loss_seq_ent, gen_out)  # [B, L, 5]
    
        g_in_padded = tf.pad(g_in, paddings=[[0, 0], [0, 0], [0, 1]], mode="CONSTANT")
    
        g_out_total = g_in_padded + lambda_seq_ent * g_out_div
    
        # Backprop the combined output gradients to the latents
        g_latent_from_db = tape_gen.gradient(gen_out, latent_vars, output_gradients=g_out_total)
    
        with tf.GradientTape() as t_lat:
            t_lat.watch(latent_vars)
            loss_rep = latent_repulsion_loss(latent_vars, margin=cos_margin)
        g_lat_rep = t_lat.gradient(loss_rep, latent_vars)
    
        frac = 1.0 - (opt_iter / float(max_iters))
        noise_std = noise_std0 * (0.5 + 0.5 * frac)  # linearly decay to 0.5×
        g_noise = tf.random.normal(tf.shape(latent_vars), stddev=noise_std)
    
        g_latent = g_latent_from_db
        optimizer.apply_gradients([(g_latent, latent_vars)])
    
        # ---- Bookkeeping (unchanged outputs) ----
        s1 = model(ohe4).numpy()
        s2 = model2(ohe4).numpy()
        s3 = model3(ohe4).numpy()
        s_target = target_scores_batch.numpy()
    
        bind_scores_list.append(s1)
        bind_scores_list2.append(s2)
        bind_scores_list_target.append(s_target)
    
        bind_scores_means.append(float(np.mean(s1)))
        bind_scores_means2.append(float(np.mean(s2)))
        bind_scores_means_target.append(float(np.mean(s_target)))
    
        scores_history.append(np.mean(np.stack([s1, s2, s3], axis=0), axis=0))  # [B, 1]
        target_scores_history.append(s_target)
    
        seq_chars = decode_indices_to_strings(idx_discrete)
        sequences_list.append(seq_chars)
        sequences_history.append(seq_chars)
        bind_scores_means_total.append(float(np.mean(s1 + s2)))
    
        if (opt_iter + 1) % reinit_every == 0:
            s_flat = s_target.reshape(-1)
            k = max(1, int(reinit_frac * s_flat.shape[0]))
            worst_idx = np.argsort(s_flat)[:k]
            z_std = tf.math.reduce_std(latent_vars)
            new_z = tf.random.normal([k, int(latent_vars.shape[1])], stddev=tf.cast(z_std + 1e-6, tf.float32))
            latent_vars_numpy = latent_vars.numpy()
            latent_vars_numpy[worst_idx, :] = new_z.numpy()
            latent_vars.assign(latent_vars_numpy)
    
    print("Optimization completed (TF1-style).")


    best_sequences, best_scores = select_best(scores_history, target_scores_history, sequences_history)

    if MEASURE_TEST:
        final_generated = generator(latent_vars, training=False)
        final_sequences_for_deepbind = final_generated[:, :, :4]
        bind_scores_test = model_test(final_sequences_for_deepbind)
    else:
        bind_scores_test = None

    dirname = './output_tf2/' + protein_name + "_inv_distance_softmax_method" + "_maxiters_" + str(max_iters) + "/"
    os.makedirs(dirname, exist_ok=True)
    
    with open(dirname + protein_name + '_best_binding_sequences.txt', 'w') as f:
        for item in best_sequences:
            f.write(f"{item}\n")
    
    with open(dirname + protein_name + '_initial_sequences.txt', 'w') as f:
        for item in sequences_list[0]:
            f.write(f"{item}\n")
    
    with open(dirname + protein_name + '_best_binding_scores.txt', 'w') as f:
        for item in best_scores:
            f.write(f"{item}\n")
    
    with open(dirname + protein_name + '_initial_binding_scores.txt', 'w') as f:
        for item in bind_scores_list_target[0]:
            f.write(f"{item}\n")

    if MEASURE_TEST and bind_scores_test is not None:
        np.savetxt(dirname + f"{protein_name}_test_binding_scores" + ".txt", bind_scores_test.numpy())
    
    prebind = bind_scores_list_target[0]
    postbind = bind_scores_list_target[np.argmax(bind_scores_means_target)]
    
    sortinds = np.flip(postbind.argsort())
    prebind = prebind[sortinds]
    postbind = postbind[sortinds]
    
    print("Optimization completed!")
    print(f"Results saved to: {dirname}")
    print(f"Initial mean binding score: {np.mean(prebind):.4f}")
    print(f"Final mean binding score: {np.mean(best_scores):.4f}")
    print(f"Improvement: {np.mean(best_scores) - np.mean(prebind):.4f}")
    print(f"Initial max binding score: {np.max(prebind):.4f}")
    print(f"Final max binding score: {np.max(best_scores):.4f}")
    print(f"Improvement: {np.max(best_scores) - np.max(prebind):.4f}")