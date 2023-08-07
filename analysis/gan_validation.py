from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from scipy.stats import norm,ks_2samp,kstest,ttest_ind,wilcoxon,bootstrap, mannwhitneyu, pearsonr
from polyleven import levenshtein
import seaborn as sns
random.seed(1337)
import os
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None 
import RNA
from cliffs_delta import cliffs_delta

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

colors = ['#2e4045', '#83adb5', '#c7bbc9', '#5e3c58', '#bfb5b2']
colors = ['#48595e', '#83adb5', '#c7bbc9', '#826b7e', '#bfb5b2']
colors = ['#455357', '#83adb5', '#c7bbc9', '#826c7e', '#bfb5b2']

# colors = ['#8dd3c7', '#ffffb3', '#bebada']
palette={'Generated':colors[0],'Natural':colors[3],'Random':colors[4]}

MAX_LEN = 32
BATCH_SIZE = 2048
gen_path = './generated.txt'
real_path = './piRNAs.fa'

params = {'legend.fontsize': 90,
        'figure.figsize': (64, 24),
        'axes.labelsize': 90,
        'axes.titlesize':90,
        'xtick.labelsize':90,
        'ytick.labelsize':90}

plt.rcParams.update(params)

def recover_seq(samples, rev_charmap):
    """Convert samples to strings and save to log directory."""
    
    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)
    return seqs        

def random_data(length):
    rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3}

    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    mapping = dict(zip([0,1,2,3],"ACGT"))
    rsample = ''
    for i in range(length):
        p = random.random()
        if p < 0.3:
            rsample += 'C'
        elif p < 0.6:
            rsample += 'G'
        elif p < 0.8:
            rsample += 'A'
        else:
            rsample += 'T'

    return rsample

def hamming_dist(src, target, verbose=False):
    
    
    dists = []
    for i in range(len(src)):
        smallest = np.inf
        for j in range(len(target)):
            dist = levenshtein(src[i],target[j])
            if dist > 0 and dist < smallest:
                smallest = dist

        if i%5000 and verbose:
            print(f"{i} number of iterations completed ...")

        dists.append(dist)

    return np.array(dists)
   
def gc_percentage(seq):
    count = 0.0
    for char in seq:
        if char == 'C' or char == 'G':
            count +=1

    return float(count/len(seq))

def get_gc_content(data):
    gc_content = []
    for seq in data:
        seq.replace('\n','')
        seq.replace('*','')
        gc = gc_percentage(seq)
        gc_content.append(gc)

def get_gc_content_many(data):
    collection = []

    gc_content = []
    for seq in data:
        seq.replace('\n','')
        seq.replace('*','')
        gc = gc_percentage(seq)
        gc_content.append(gc)


    return gc_content


rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def recover_seq(samples, rev_charmap):
    """Convert samples to strings and save to log directory."""
    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)

    return seqs
    
def file_to_list(file_name):
    data = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        
        for seq in lines:
            seq_ = seq.replace('\n','')
            seq_ = seq_.replace('U','T')
            data.append(seq_)

    return data


min_len = None
nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
                'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
                'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}


    
def encode_seq(seq, max_len=MAX_LEN):
    # print(seq)
    length = len(seq)
    if max_len > 0 and min_len is None:
        padding_needed = max_len - length
        seq = "N"*padding_needed + seq
    if min_len is not None:
        if len(seq) < min_len:
            seq = "N"*(min_len - len(seq)) + seq

        if len(seq) > min_len:
            seq = seq[(len(seq) - min_len):]
    seq = seq.lower()
    one_hot = np.array([nuc_dict[x] for x in seq]) # get stacked on top of each other
    
    # print(np.shape(one_hot))
    return one_hot

def read_gen_seqs(path):
    readseqs = []
    with open(path, 'r') as f:
        readseqs = f.readlines()

    for i in range(len(readseqs)):
        readseqs[i] = readseqs[i].replace('\n','')

    
    return readseqs

def read_reals(path):
    readseqs = []
    with open(path, 'r') as f:
        readseqs = f.readlines()

    for i in range(len(readseqs)):
        readseqs[i] = readseqs[i].replace('\n','')

    
    return readseqs


def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def encode(seqs,en_type="one_hot",reshape=True):

    ohe_sequences = np.asarray([one_hot_encode(x) for x in seqs])
    return ohe_sequences



rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}


if __name__ == "__main__":

    np.random.seed(35)

    seqs_gen_init = read_gen_seqs(gen_path)

    gens = []
    for i in range(len(seqs_gen_init)):
        if seqs_gen_init[i] not in gens:
            gens.append(seqs_gen_init[i])
            if len(gens) == BATCH_SIZE:
                break


    genpreds = []
    gen_gc = get_gc_content_many(gens)

    for i in range(len(gens)):
        (ss, mfe) = RNA.fold(seqs_gen_init[i])
        genpreds.append(mfe)


    randoms = []

    for i in range(BATCH_SIZE):
        length = random.randint(26,32)
        seqs = random_data(length)
        randoms.append(seqs)

    randpreds = []
    rand_gc = get_gc_content_many(randoms)

    for i in range(len(randoms)):
        (ss, mfe) = RNA.fold(randoms[i])
        randpreds.append(mfe)



    reals = read_reals(real_path)

    indices = []

    for i in range(len(reals)):

        indices.append(i)

    samples = np.random.choice(len(indices),BATCH_SIZE,replace=False)

    chosen = []

    for i in range(len(samples)):
        chosen.append(reals[samples[i]])

    realpreds = []
    real_gc = get_gc_content_many(chosen)

    for i in range(len(chosen)):
        (ss, mfe) = RNA.fold(chosen[i])
        realpreds.append(mfe)

    realpreds_stat = []

    for i in range(len(reals)):
        (ss, mfe) = RNA.fold(reals[i])
        realpreds_stat.append(mfe)


    """ Distances """
    real = reals
    rand = randoms
    gen = gens

    print("Calculating Levenshtein distances. This may take a while ...")

    if os.path.exists('./files/rand_ham_dist.npy'):
        dist_rand = np.load('./files/rand_ham_dist.npy', allow_pickle=True)
    else:
        dist_rand = hamming_dist(rand,real)
        with open('./files/rand_ham_dist.npy', 'wb') as f:
            np.save(f,dist_rand)

    if os.path.exists('./files/real_ham_dist.npy'):
        dist_real = np.load('./files/real_ham_dist.npy', allow_pickle=True)
    else:
        dist_real = hamming_dist(real,real)
        with open('./files/real_ham_dist.npy', 'wb') as f:
            np.save(f,dist_real)        

    if os.path.exists('./files/gen_ham_dist.npy'):
        dist_gen = np.load('./files/gen_ham_dist.npy', allow_pickle=True)
    else:
        dist_gen = hamming_dist(gens, real)
        with open('./files/gen_ham_dist.npy', 'wb') as f:
            np.save(f,dist_gen)

    fig, axs = plt.subplots(1,2)

    real_x = ['Natural' for i in range(len(dist_real))]
    gen_x = ['Generated' for i in range(len(dist_gen))]
    rand_x = ['Random' for i in range(len(dist_rand))]


    x = np.concatenate((gen_x,real_x,rand_x))
    y = np.concatenate((dist_gen,dist_real,dist_rand))

    dist_t_rand = mannwhitneyu(dist_rand, dist_real)
    dist_t_gen = mannwhitneyu(dist_gen, dist_real)


    n1 = len(dist_rand)
    n2 = len(dist_real)

    u1 = np.mean(dist_rand)
    u2 = np.mean(dist_real)
    
    s1 = np.std(dist_rand)
    s2 = np.std(dist_real)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    dist_rand_effect_size = (u1 - u2)/s
    dist_rand_effect_size = cliffs_delta(dist_rand, dist_real)
    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = dist_rand
    ds2 = dist_real
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    dist_rand_CI = (diffs[k], diffs[len(diffs)-k])

    n1 = len(dist_gen)
    n2 = len(dist_real)

    u1 = np.mean(dist_gen)
    u2 = np.mean(dist_real)
    
    s1 = np.std(dist_gen)
    s2 = np.std(dist_real)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    dist_gen_effect_size = (u1 - u2)/s
    dist_gen_effect_size = cliffs_delta(dist_gen, dist_real)
    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = dist_gen
    ds2 = dist_real
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    dist_gen_CI = (diffs[k], diffs[len(diffs)-k])

    df = pd.DataFrame({'x':x,'y':y})

    sns.boxplot(x=df['x'],y=df['y'],ax=axs[0],palette=palette)

    axs[0].set_ylabel("Min. Levenshtein Distance")
    axs[0].set_xlabel("")

    """ Finished """

    bins = np.linspace(2.5, 9, 30)

    o_patch = mpatches.Patch(color='orange', label='Generated')
    g_patch = mpatches.Patch(color='green', label='Random')
    b_patch = mpatches.Patch(color='blue', label='Natural')

    real_x = ['Natural' for i in range(len(real_gc))]
    gen_x = ['Generated' for i in range(len(gen_gc))]

    x = np.concatenate((gen_x,real_x))
    y = np.concatenate((gen_gc,real_gc))
    gc_t = mannwhitneyu(gen_gc, real_gc)

    n1 = len(gen_gc)
    n2 = len(real_gc)

    u1 = np.mean(gen_gc)
    u2 = np.mean(real_gc)
    
    s1 = np.std(gen_gc)
    s2 = np.std(real_gc)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    gc_effect_size = (u1 - u2)/s
    gc_effect_size = cliffs_delta(gen_gc, real_gc)



    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = gen_gc
    ds2 = real_gc
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    gc_CI = (diffs[k], diffs[len(diffs)-k])

    df = pd.DataFrame({'x':x,'y':y})

    sns.violinplot(x=df['x'],y=df['y'],ax=axs[1],palette=palette)

    axs[1].set_ylabel("G/C Content")
    axs[1].set_xlabel("")

    axs[0].set_title('A',weight='bold',fontsize=100,loc='left')
    axs[1].set_title('B',weight='bold',fontsize=100,loc='left')

    fig.tight_layout(pad=2)

    plt.savefig('./../figures/dist_gc.png')
    plt.clf()
    fig, axs = plt.subplots(1,2,gridspec_kw={'width_ratios': [2, 1]})


    

    seqs_gen_init = read_gen_seqs(gen_path)

    gens = []
    for i in range(len(seqs_gen_init)):
        if seqs_gen_init[i] not in gens:
            gens.append(seqs_gen_init[i])
            if len(gens) == BATCH_SIZE:
                break


    genpreds = []

    for i in range(len(gens)):
        (ss, mfe) = RNA.fold(seqs_gen_init[i])
        genpreds.append(mfe)

    randoms = []

    for i in range(BATCH_SIZE):
        length = random.randint(26,32)
        seqs = random_data(length)
        randoms.append(seqs)

    randpreds = []

    for i in range(len(randoms)):
        (ss, mfe) = RNA.fold(randoms[i])
        randpreds.append(mfe)


    reals = read_reals(real_path)

    indices = []

    for i in range(len(reals)):

        indices.append(i)

    samples = np.random.choice(len(indices),BATCH_SIZE,replace=False)

    chosen = []

    for i in range(len(samples)):
        chosen.append(reals[samples[i]])

    realpreds = []

    for i in range(len(chosen)):
        (ss, mfe) = RNA.fold(chosen[i])
        realpreds.append(mfe)

    realpreds_stat = []

    for i in range(len(reals)):
        (ss, mfe) = RNA.fold(reals[i])
        realpreds_stat.append(mfe)


    o_patch = mpatches.Patch(color='orange', label='Generated')
    g_patch = mpatches.Patch(color='green', label='Random')
    b_patch = mpatches.Patch(color='blue', label='Natural')

    data = pd.DataFrame({'Natural': realpreds, 'Generated': genpreds, 'Random':randpreds})

    real_x = ['Natural' for i in range(len(realpreds))]
    gen_x = ['Generated' for i in range(len(genpreds))]
    rand_x = ['Random' for i in range(len(randpreds))]

    x = np.concatenate((gen_x,real_x,rand_x))
    y = np.concatenate((genpreds,realpreds,randpreds))
    mfe_t_rand = ttest_ind(randpreds, realpreds)
    mfe_t_gen = ttest_ind(genpreds, realpreds)
    
    n1 = len(randpreds)
    n2 = len(realpreds)

    u1 = np.mean(randpreds)
    u2 = np.mean(realpreds)
    
    s1 = np.std(randpreds)
    s2 = np.std(realpreds)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    mfe_rand_effect_size = (u1 - u2)/s
    mfe_rand_effect_size = cliffs_delta(randpreds, realpreds)
    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = randpreds
    ds2 = realpreds
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    mfe_rand_CI = (diffs[k], diffs[len(diffs)-k])

    df = pd.DataFrame({'x':x,'y':y})

    sns.violinplot(x=df['x'],y=df['y'],ax=axs[0],palette=palette)


    n1 = len(genpreds)
    n2 = len(realpreds)

    u1 = np.mean(genpreds)
    u2 = np.mean(realpreds)
    
    s1 = np.std(genpreds)
    s2 = np.std(realpreds)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    mfe_gen_effect_size = (u1 - u2)/s
    mfe_gen_effect_size = cliffs_delta(genpreds, realpreds)
    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = genpreds
    ds2 = realpreds
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    mfe_gen_CI = (diffs[k], diffs[len(diffs)-k])

    df = pd.DataFrame({'x':x,'y':y})

    sns.violinplot(x=df['x'],y=df['y'],ax=axs[0],palette=palette)

    axs[0].set_ylabel("Minimum Free Energy")
    axs[0].set_xlabel("")

    reals = read_reals(real_path)
    gens = read_gen_seqs(gen_path)

    gen_lens = [len(seq) for seq in gens]
    real_lens = [len(seq) for seq in reals]


    selected = random.choices([i for i in range(len(real_lens))],k=len(gen_lens))

    real_lens_ = [real_lens[i] for i in selected]

    y = np.concatenate((gen_lens,real_lens_))
    gen_x = ['Generated' for i in range(len(gen_lens))]
    real_x = ['Natural' for i in range(len(real_lens_))]
    x = np.concatenate((gen_x,real_x))

    df = pd.DataFrame({'x':x,'y':y})

    sns.boxplot(x=df['x'],y=df['y'],ax=axs[1],palette=palette)
    
    #############################################################
    
    axs[1].set_ylabel("Sequence Length")
    axs[1].set_xlabel("")
   
    axs[0].set_title('A',weight='bold',fontsize=100,loc='left')
    axs[1].set_title('B',weight='bold',fontsize=100,loc='left')

    fig.tight_layout()

    plt.savefig('./../figures/mfe_length.png')

    print("#############################################")
    print(f'GC content p-value: {gc_t}')
    print(f'MFE rand content p-value: {mfe_t_rand}')
    print(f'MFE gen content p-value: {mfe_t_gen}')
    print(f'Dist rand content p-value: {dist_t_rand}')
    print(f'Dist gen content p-value: {dist_t_gen}')
    print("#############################################")
    print(f'GC content ES: {gc_effect_size}')
    print(f'MFE rand content ES: {mfe_rand_effect_size}')
    print(f'MFE gen content ES: {mfe_gen_effect_size}')
    print(f'Dist rand content ES: {dist_rand_effect_size}')
    print(f'Dist gen content ES: {dist_gen_effect_size}')
    print("#############################################")
    print(f'GC content CI: {gc_CI}')
    print(f'MFE rand content CI: {mfe_rand_CI}')
    print(f'MFE gen content CI: {mfe_gen_CI}')
    print(f'Dist rand content CI: {dist_rand_CI}')
    print(f'Dist gen content CI: {dist_gen_CI}')
    print("#############################################")