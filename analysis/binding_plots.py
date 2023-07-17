import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from polyleven import levenshtein
import seaborn as sns
import random
random.seed(1337)
import os
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None 
import argparse
import RNA

parser = argparse.ArgumentParser()

parser.add_argument('-p1', type=str, required=False, default='SOX4')
parser.add_argument('-p2', type=str, required=False, default='SOX2')
parser.add_argument('-n1', type=int, required=False, default=3000)
parser.add_argument('-n2', type=int, required=False, default=3000)

args = parser.parse_args()

# Dont use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

params = {'legend.fontsize': 68,
        'figure.figsize': (64, 42),
        'axes.labelsize': 68,
        'axes.titlesize':68,
        'xtick.labelsize':68,
        'ytick.labelsize':68}

plt.rcParams.update(params)


def read_score(path):
    readseqs = []

    with open(path, 'r') as f:
        readseqs = f.readlines()

    seqs = readseqs

    for i in range(len(seqs)):
        seqs[i] = seqs[i].replace('\n','')

    for i in range(len(seqs)):
        seqs[i] = float(seqs[i])
        
    return seqs


rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}


if __name__ == "__main__":



    BATCH_SIZE = 2048
    np.random.seed(35)

    init_path = f'./../output/{args.p1}_inv_distance_softmax_method_maxiters_{args.n1}/{args.p1}_initial_binding_scores.txt'
    best_path = f'./../output/{args.p1}_inv_distance_softmax_method_maxiters_{args.n1}/{args.p1}_best_binding_scores.txt'

    seqs_init = read_score(init_path)
    seqs_final = read_score(best_path)

    init_score = seqs_init
    best_score = seqs_final

    init_x = ['Initial' for i in range(len(init_score))]
    best_x = ['Optimized' for i in range(len(best_score))]

    init_x_in = [i for i in range(len(init_score))]
    best_x_in = [i for i in range(len(best_score))]
    # rand_x = ['random' for i in range(len(randpreds))]

    x = np.concatenate((init_x,best_x))
    y = np.concatenate((init_score,best_score))


    print(best_score)
    print(init_score)

    df = pd.DataFrame({'x':x,'y':y})

    fig, axs = plt.subplots(2,2)

    sns.distplot(a=init_score,ax=axs[0,0])
    sns.distplot(a=best_score,ax=axs[0,0])

    axs[0,0].set_ylabel("SOX4 Binding Score")

    axs[0,0].set_xlabel("")
    

    sns.boxplot(x=x,y=y,ax=axs[0,1])

    axs[0,1].set_ylabel("SOX4 Binding Score")

    axs[0,1].set_xlabel("")   

    init_path = f'./../output/{args.p2}_inv_distance_softmax_method_maxiters_{args.n2}/{args.p2}_initial_binding_scores.txt'
    best_path = f'./../output/{args.p2}_inv_distance_softmax_method_maxiters_{args.n2}/{args.p2}_best_binding_scores.txt'
    seqs_init = read_score(init_path)
    seqs_final = read_score(best_path)


    init_score = seqs_init
    best_score = seqs_final

    init_x = ['Initial' for i in range(len(init_score))]
    best_x = ['Optimized' for i in range(len(best_score))]

    init_x_in = [i for i in range(len(init_score))]
    best_x_in = [i for i in range(len(best_score))]
    # rand_x = ['random' for i in range(len(randpreds))]

    x = np.concatenate((init_x,best_x))
    y = np.concatenate((init_score,best_score))

    print(best_score)
    print(init_score)

    df = pd.DataFrame({'x':x,'y':y})

    

    sns.distplot(a=init_score,ax=axs[1,0])
    sns.distplot(a=best_score,ax=axs[1,0])

    axs[1,0].set_ylabel("SOX2 Binding Score")
    axs[1,0].set_xlabel("")
    

    sns.boxplot(x=x,y=y,ax=axs[1,1])

    axs[1,1].set_ylabel("SOX2 Binding Score")
    axs[1,1].set_xlabel("")

    axs[1,0].set_title('C',weight='bold',fontsize=90)
    axs[1,1].set_title('D',weight='bold',fontsize=90)
    axs[0,0].set_title('A',weight='bold',fontsize=90)
    axs[0,1].set_title('B',weight='bold',fontsize=90)

    axs[0,0].legend(labels=['Initial Binding Score','Optimized Binding Score'])

    fig.tight_layout()  

    plt.savefig('./../figures/bind_init_best_boxplot.png')


    