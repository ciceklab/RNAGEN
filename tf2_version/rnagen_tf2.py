import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import socket
import datetime
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.mixed_precision import set_global_policy
from tqdm import tqdm, trange
from tensorflow.keras.mixed_precision import LossScaleOptimizer

'''
This script is a part of project RNAGEN (piRNA's). The script is responsible for training of the generator 
which learns to generate realistic piRNA's for homo-sapiens. This script is an essential part of the project since
the underlying data distribution is captured through WGAN with gradient penalty architecture. On top of the 
generated piRNA sequences, that looks real, an optimization via activation maximization will be applied jointly 
(i.e., many classifiers on piRNAs, potentially cancer related research) to have realistic piRNA sequences with desired
properties. 

This architecture is based on WGAN with Gradient Penalty method. The dataset used to train the GAN here has 50397 samples in total
with minimum rna-seq length 26 and maximum 32. The mean sequence length is 28.63 with an std of 1.742. The data used is obtained from
DASHR project (hg38). 
CicekLab 2023, Furkan Ozden - Converted to TF2, Optimized 2025
'''

# Enable mixed precision
set_global_policy('mixed_float16')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=False, default='./data/DASHR2_GEO_hg38_sequenceTable_export.csv')
parser.add_argument('-n', type=int, required=False, default=200000)
parser.add_argument('-lr', type=int, required=False, default=4)
parser.add_argument('-gpu', type=str, required=False, default='0')
args = parser.parse_args()

# Set GPU
if args.gpu != '-1':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu)], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[int(args.gpu)], True)
        except (RuntimeError, ValueError, IndexError):
            print("Invalid GPU configuration, using CPU")

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs_tf2/", "gan_test", stamp)
    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: 
        os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

# Data loading and preprocessing
data_path = args.i
precomputed_path = 'precomputed_ohe_sequences.npy'

rna_vocab = {"A": 0, "C": 1, "G": 2, "U": 3, "*": 4}
rev_rna_vocab = {v: k for k, v in rna_vocab.items()}

def one_hot_encode(seq, SEQ_LEN=32):
    mapping = dict(zip("ACGU*", range(5)))    
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(5)[seq2], extra]).astype(np.float16)
    return np.eye(5)[seq2].astype(np.float16)

# Precompute one-hot encodings if not already done
if not os.path.exists(precomputed_path):
    data = pd.read_csv(data_path)
    piRNAdf = data.loc[data['rnaClass'] == 'piRNA']
    piRNAarr = piRNAdf.values
    sequences = [x.upper() for x in piRNAarr[:, 7]]
    ohe_sequences = np.asarray([one_hot_encode(x) for x in sequences])
    np.save(precomputed_path, ohe_sequences)
else:
    ohe_sequences = np.load(precomputed_path)

# Hyperparameters
BATCH_SIZE = 128
ITERS = args.n
SEQ_LEN = 32
DIM = 25
CRITIC_ITERS = 5
LAMBDA = 10
LR = 0.0001

# Set seed for reproducibility
seed = 35
np.random.seed(seed)
tf.random.set_seed(seed)

logdir, checkpoint_baseline = log(samples_dir=True)

# Model definitions
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

# Initialize models
generator = ResNetGenerator(DIM, SEQ_LEN, 5)
discriminator = ResNetDiscriminator(DIM, SEQ_LEN, 5)

# Optimizers
base_gen_opt = tf.keras.optimizers.legacy.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
base_disc_opt = tf.keras.optimizers.legacy.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)

gen_optimizer  = LossScaleOptimizer(base_gen_opt)
disc_optimizer = LossScaleOptimizer(base_disc_opt)

@tf.function
def gradient_penalty(real_samples, fake_samples, discriminator):
    # both real_samples and fake_samples come in as float16 or float32
    real_f32 = tf.cast(real_samples, tf.float32)
    fake_f32 = tf.cast(fake_samples, tf.float32)
    
    batch_size = tf.shape(real_f32)[0]
    alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interp = real_f32 + alpha * (fake_f32 - real_f32)
    
    with tf.GradientTape() as tape:
        tape.watch(interp)
        pred = discriminator(interp, training=True)
    grads = tape.gradient(pred, interp)
    
    # back to float32 for norm
    grads_f32 = tf.cast(grads, tf.float32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads_f32), axis=[1,2]) + 1e-12)
    gp = tf.reduce_mean((norm - 1.0)**2)
    return gp

@tf.function
def train_discriminator(real_data):
    # real_data comes in as float16
    with tf.GradientTape() as tape:
        fake = generator(tf.random.normal([tf.shape(real_data)[0], DIM], dtype=tf.float16),
                         training=True)
        real_out = discriminator(real_data,    training=True)
        fake_out = discriminator(fake,         training=True)
        gp      = gradient_penalty(real_data, fake, discriminator)
        # compute WGAN-GP loss in float32
        d_loss  = tf.reduce_mean(tf.cast(fake_out, tf.float32)) \
                  - tf.reduce_mean(tf.cast(real_out, tf.float32)) \
                  + LAMBDA * gp

        # scale it
        scaled_d_loss = disc_optimizer.get_scaled_loss(d_loss)

    # gradients of the scaled loss
    grads = tape.gradient(scaled_d_loss, discriminator.trainable_variables)
    # un-scale them
    grads = disc_optimizer.get_unscaled_gradients(grads)
    # apply
    disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss

@tf.function
def train_generator(batch_size):
    with tf.GradientTape() as tape:
        fake = generator(tf.random.normal([batch_size, DIM], dtype=tf.float16),
                         training=True)
        fake_out = discriminator(fake, training=True)
        g_loss = -tf.reduce_mean(tf.cast(fake_out, tf.float32))
        scaled_g_loss = gen_optimizer.get_scaled_loss(g_loss)

    grads = tape.gradient(scaled_g_loss, generator.trainable_variables)
    grads = gen_optimizer.get_unscaled_gradients(grads)
    gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss

@tf.function
def compute_validation_metric(real_data):
    """Compute validation metric: negative Wasserstein distance (lower is better)."""
    fake = generator(tf.random.normal([tf.shape(real_data)[0], DIM], dtype=tf.float16),
                     training=False)
    real_out = discriminator(real_data, training=False)
    fake_out = discriminator(fake, training=False)
    
    # Wasserstein distance approximation (without gradient penalty for cleaner metric)
    # We return the negative Wasserstein distance: -(E[D(fake)] - E[D(real)])
    # Higher values indicate better generator (smaller Wasserstein distance)
    wasserstein_distance = tf.reduce_mean(tf.cast(fake_out, tf.float32)) - tf.reduce_mean(tf.cast(real_out, tf.float32))
    return -wasserstein_distance  # Negative so higher is better

@tf.function
def compute_validation_discriminator_loss(real_data):
    """Compute full validation discriminator loss with gradient penalty for monitoring."""
    fake = generator(tf.random.normal([tf.shape(real_data)[0], DIM], dtype=tf.float16),
                     training=False)
    real_out = discriminator(real_data, training=False)
    fake_out = discriminator(fake, training=False)
    gp = gradient_penalty(real_data, fake, discriminator)
    
    # WGAN-GP discriminator loss
    d_loss = tf.reduce_mean(tf.cast(fake_out, tf.float32)) \
             - tf.reduce_mean(tf.cast(real_out, tf.float32)) \
             + LAMBDA * gp
    return d_loss

# Data preparation
data = ohe_sequences.astype(np.float16)
validate = True

if validate:
    split = len(data) // 10
    train_data = data[split:]
    valid_data = data[:split]
else:
    train_data = data

def create_dataset(data, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_dataset = create_dataset(train_data, BATCH_SIZE)
if validate:
    valid_dataset = create_dataset(valid_data, BATCH_SIZE, shuffle=False)

# Utility functions
def save_samples(logdir, samples, iteration, rev_vocab, annotated=False):
    samples_dir = os.path.join(logdir, "samples")
    sequences = []
    for sample in samples:
        seq = ""
        probs = tf.nn.softmax(sample, axis=-1).numpy()
        for pos in probs:
            idx = np.argmax(pos)
            if idx < 4:
                seq += rev_vocab[idx]
        sequences.append(seq)
    
    filename = f"samples_{iteration}.txt"
    filepath = os.path.join(samples_dir, filename)
    with open(filepath, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f"Sample_{i}: {seq}\n")

def plot_losses(train_counts, train_cost, logdir, name, xlabel="Iteration", ylabel="Cost"):
    if len(train_counts) == 0 or len(train_cost) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_counts, train_cost)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(logdir, f"{name}.png"))
    plt.close()

def validplot(valid_counts, valid_cost, valid_dcost, smoothed_cost, logdir, name, xlabel="Iteration", ylabel="Loss/Metric"):
    if len(valid_counts) == 0:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot validation generator loss if available
    if len(valid_cost) == len(valid_counts) and len(valid_cost) > 0:
        plt.plot(valid_counts, valid_cost, label='Validation Generator Loss', color='blue', alpha=0.7)
    
    # Plot validation discriminator loss if available
    if len(valid_dcost) == len(valid_counts) and len(valid_dcost) > 0:
        plt.plot(valid_counts, valid_dcost, label='Validation Discriminator Loss', color='red', alpha=0.7)
        
    # Plot smoothed primary metric if available (note: this is now negative Wasserstein distance)
    if len(smoothed_cost) == len(valid_counts) and len(smoothed_cost) > 0:
        plt.plot(valid_counts, smoothed_cost, label='Smoothed Val Metric (↑ better)', color='darkgreen', linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logdir, f"{name}.png"))
    plt.close()

# Dynamic critic iterations
def get_critic_iters(step, max_iters=5, min_iters=1, switch_step=50000):
    return max(min_iters, max_iters - (step // switch_step))

# Training loop
print("Training GAN")
print("================================================")

train_iters = ITERS
validation_iters = 50
checkpoint_iters = 1000

fixed_noise = tf.random.normal([BATCH_SIZE, DIM], dtype=tf.float16)
train_cost = []
train_counts = []
valid_cost = []
valid_dcost = []
valid_counts = []

checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer
)

train_dataset_iter = iter(train_dataset)
disc_loss_history = []
gen_loss_history = []
steps = []

# Enhanced early stopping parameters
# For WGAN-GP discriminator loss, we want to minimize the absolute value or use Wasserstein distance
# Better approach: use negative Wasserstein distance (higher is better)
best_val_metric = float('inf')  # Will be set on first validation
metric_delta = 1e-4  # Reduced delta - was potentially too large
patience = 0
max_patience = 10000  # Give it 5 validation rounds without improvement
val_history = []  # Store raw validation metrics for smoothing
smoothed_history = []  # Store smoothed validation metrics
N_smooth = 3  # Number of points for rolling average

# Training loop with tqdm
pbar = trange(train_iters, desc="Training GAN", unit="iter")
for idx in pbar:
    true_count = idx + 1 + checkpoint_baseline

    # Discriminator training
    critic_iters = get_critic_iters(idx)
    disc_losses = []
    for _ in range(critic_iters):
        try:
            batch = next(train_dataset_iter)
        except StopIteration:
            train_dataset_iter = iter(train_dataset)
            batch = next(train_dataset_iter)
        d_loss = train_discriminator(batch)
        disc_losses.append(d_loss)
    avg_d_loss = tf.reduce_mean(disc_losses)

    # Generator training
    g_loss = train_generator(BATCH_SIZE)
    
    # Store losses for plotting
    disc_loss_history.append(float(avg_d_loss))
    gen_loss_history.append(float(g_loss))
    steps.append(true_count)

    # Validation and enhanced early stopping
    if validate and (true_count % validation_iters == 0):
        # Compute validation losses
        val_d_losses = []
        val_g_losses = []
        val_metrics = []  # Primary metric: negative Wasserstein distance
        
        for valid_batch in valid_dataset:
            # Compute primary validation metric (negative Wasserstein distance)
            val_metric = compute_validation_metric(valid_batch)
            val_metrics.append(float(val_metric))
            
            # Compute validation discriminator loss (with GP) for monitoring
            val_d_loss = compute_validation_discriminator_loss(valid_batch)
            val_d_losses.append(float(val_d_loss))
            
            # Also compute generator validation loss for monitoring
            noise = tf.random.normal([BATCH_SIZE, DIM], dtype=tf.float32)
            fake_data = generator(noise, training=False)
            fake_out = discriminator(fake_data, training=False)
            val_g_loss = -tf.reduce_mean(tf.cast(fake_out, tf.float32))
            val_g_losses.append(float(val_g_loss))
        
        # Average validation losses and metric
        current_val_metric = np.mean(val_metrics)      # Primary metric (higher is better)
        current_val_d_loss = np.mean(val_d_losses)     # Discriminator loss (for monitoring)
        current_val_g_loss = np.mean(val_g_losses)     # Generator loss (for monitoring)
        
        # Store raw validation metric
        val_history.append(current_val_metric)
        
        # Compute smoothed metric over the last N points
        N = min(len(val_history), N_smooth)
        smoothed_metric = sum(val_history[-N:]) / N
        smoothed_history.append(smoothed_metric)
        
        # Store validation metrics for plotting
        valid_cost.append(current_val_g_loss)      # Generator validation loss
        valid_dcost.append(current_val_d_loss)     # Discriminator validation loss  
        valid_counts.append(true_count)

        # Initialize best_val_metric on first validation
        if len(val_history) == 1:
            best_val_metric = smoothed_metric
            patience = 0
            # Save best model checkpoint
            best_ckpt_dir = os.path.join(logdir, "checkpoints", "best_model")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            checkpoint.save(os.path.join(best_ckpt_dir, "trained_gan"))
            pbar.write(f"    → Initial best model saved! Smoothed metric: {smoothed_metric:.4f}")
        # Check for improvement using smoothed metric (HIGHER IS BETTER)
        elif smoothed_metric < best_val_metric - metric_delta:
            best_val_metric = smoothed_metric
            patience = 0
            # Save best model checkpoint
            best_ckpt_dir = os.path.join(logdir, "checkpoints", "best_model")
            os.makedirs(best_ckpt_dir, exist_ok=True)
            checkpoint.save(os.path.join(best_ckpt_dir, "trained_gan"))
            pbar.write(f"    → New best model saved! Smoothed metric: {smoothed_metric:.4f}")
        else:
            patience += 1

        # Update progress bar with enhanced metrics
        pbar.set_postfix({
            "D_loss": f"{avg_d_loss:.4f}",
            "G_loss": f"{g_loss:.4f}",
            "Val_D": f"{current_val_d_loss:.4f}",
            "Val_Metric": f"{current_val_metric:.4f}",
            "Smooth": f"{smoothed_metric:.4f}",
            "Patience": f"{patience}/{max_patience}",
            "Best": f"{best_val_metric:.4f}"
        })

        # Enhanced logging
        pbar.write(
            f"[VAL {true_count:6d}] Val_Metric: {current_val_metric:.4f} | "
            f"Val_D_loss: {current_val_d_loss:.4f} | "
            f"Val_G_loss: {current_val_g_loss:.4f} | "
            f"Smoothed_Metric: {smoothed_metric:.4f} | "
            f"Best_Smooth: {best_val_metric:.4f} | "
            f"Patience: {patience}/{max_patience}"
        )

        # Check for early stopping
        if patience >= max_patience:
            pbar.write(f"Early stopping: No improvement in {max_patience} validations (smoothed={smoothed_metric:.4f})")
            break

        # Plot training curves
        if len(steps) > 0:
            plot_losses(steps, gen_loss_history, logdir, "train_gen_loss",
                       xlabel="Iteration", ylabel="Generator loss")
            plot_losses(steps, disc_loss_history, logdir, "train_disc_cost",
                       xlabel="Iteration", ylabel="Discriminator cost")
        
        # Plot validation curves with smoothed line
        if len(valid_counts) > 1:
            validplot(valid_counts, valid_cost, valid_dcost, smoothed_history, logdir,
                     name="valid_losses", xlabel="Iteration", ylabel="Loss")

        # Save routine checkpoint
        ckpt_dir = os.path.join(logdir, "checkpoints", f"checkpoint_{true_count}")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint.save(os.path.join(ckpt_dir, "trained_gan"))

        # Generate and save samples
        samples = generator(fixed_noise, training=False)
        save_samples(logdir, samples, true_count, rev_rna_vocab)

# Final rollback to best model
if validate and os.path.exists(os.path.join(logdir, "checkpoints", "best_model")):
    pbar.write("Loading best model for final evaluation...")
    best_checkpoint_path = os.path.join(logdir, "checkpoints", "best_model", "trained_gan")
    checkpoint.restore(best_checkpoint_path)
    pbar.write(f"Best model loaded (validation metric: {best_val_metric:.4f})")
    
    # Generate final samples with best model
    final_samples = generator(fixed_noise, training=False)
    save_samples(logdir, final_samples, "final_best", rev_rna_vocab)

pbar.write("Training completed!")
print("Training loop exited.")