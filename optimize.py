import numpy as np
import tensorflow.compat.v1 as tf
import sys
import os
import pdb
import tensorflow.keras as keras
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

tf.disable_eager_execution()

K.clear_session()

parser = argparse.ArgumentParser()

parser.add_argument('-i', type=str, required=False, default='./input/opt_input.csv')
parser.add_argument('-n', type=int, required=False, default=3000)
parser.add_argument('-t', type=str, required=False, default='./data/model/trained_gan.ckpt.meta')
parser.add_argument('-lr', type=int, required=False ,default=1)
parser.add_argument('-gpu', type=str, required=False ,default='-1')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def convert_model(file_name, input_len, window_size=0):
    
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
        #print("Model has average pooling")
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
        #print("Model has reverse complementation")
        padding_out_rev = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(padding_out_fwd)
        conv_out_rev = conv_layer(padding_out_rev)
        pool_out_rev = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_rev)
        if (has_avg_pooling > 0):
            #print("Model has average pooling")
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
        
        
    model = keras.models.Model(inputs = [input_tensor],
                               outputs = [output])
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

'''
This script is a major component of the project RNAGEN by CicekLab.
This script needs: i) Trained RNA sequences generator (WGAN), ii) DeepBind model weights for the desired protein. 
The script basically optimizes generated RNA sequences to have maximum binding scores to desired proteins. 
'''


if __name__ == '__main__':
    
    input_file = args.i

    df = pd.read_csv(input_file)

    protein_names = df['protein_name'].to_numpy()
    paths = df['model_id'].to_numpy()
    distances = df['dist'].to_numpy()

    protein_name = protein_names[0]

    trained_gan_path = args.t

    base_deepbind_path = './deepbind_models/params/'

    ''' SOX Family Optimization'''
    deepbind_model_path = base_deepbind_path + paths[1] + '.txt'
    deepbind_model_path_2 = base_deepbind_path + paths[2] + '.txt'
    deepbind_model_path_3 = base_deepbind_path + paths[3] + '.txt'

    deepbind_model_target = base_deepbind_path + paths[0] + '.txt'

    # ADDITIONAL TEST, OPT FOR P1, TEST FOR P2 (Optional)
    deepbind_model_eval = base_deepbind_path + paths[4] + '.txt'
    MEASURE_TEST = False

    dist_protein_1 = float(distances[1])
    dist_protein_2 = float(distances[2])
    dist_protein_3 = float(distances[3])

    dists = [dist_protein_1,dist_protein_2,dist_protein_3]

    weights = softmax(dists)

    rna_vocab = {"A":0,
                "C":1,
                "G":2,
                "U":3,
                "*":4}

    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    '''
    load the trained WGAN model
    '''
    session = tf.keras.backend.get_session()
    gen_handler = tf.train.import_meta_graph(trained_gan_path, import_scope="generator")
    gen_handler.restore(session, trained_gan_path[:-5])

    latents = tf.get_collection('latents')[0]
    gen_output = tf.get_collection('outputs')[0]
    batch_size, latent_dim = session.run(tf.shape(latents))
    latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]


    '''
    Get DeepBind Model Ready !!!
    '''
    model = convert_model(file_name = deepbind_model_path, input_len=32) # sox15
    model2 = convert_model(file_name = deepbind_model_path_2, input_len=32) # sox14
    model3 = convert_model(file_name = deepbind_model_path_3, input_len=32) # sox7
    model_target = convert_model(file_name = deepbind_model_target, input_len=32) # sox2 - target protein
    model_test = convert_model(file_name = deepbind_model_eval, input_len=32)

    '''
    Random noise for initial latent space of generator. Drawn from standard normal distribution.
    ''' 
    start_noise = np.random.normal(size=[batch_size, latent_dim])
    session.run(tf.assign(latent_vars, start_noise))






    

    '''
    WEIGHT TUNER NETWORK definiton. 
    '''

    wtn_inps = keras.layers.Input(shape=(3,))
    batchs = K.shape(wtn_inps)[0]

    constants = np.asarray(dists).reshape((1,-1))
    k_consts = K.variable(constants)
    k_consts = K.tile(k_consts, (batchs,1))
    fixed_inp = keras.layers.Input(tensor = k_consts)

    wtn_concated_inps = keras.layers.concatenate([fixed_inp, wtn_inps])
    wtn_hidden1 = keras.layers.Dense(6, activation='relu')(wtn_concated_inps)
    wtn_hidden2 = keras.layers.Dense(12, activation='relu')(wtn_hidden1)
    wtn_out = keras.layers.Dense(3, activation='softmax')(wtn_hidden2)
    wtn_model = keras.models.Model(inputs = [wtn_inps, fixed_inp], outputs = wtn_out)

    '''
    ################# Variables Defined:
    '''

    outputTensor = model.output
    outputTensor2 = model2.output
    outputTensor3 = model3.output
    outputTensor_target = model_target.output
    outputTensor_test = model_test.output

    cost = tf.reduce_mean(-outputTensor)
    cost2 = tf.reduce_mean(-outputTensor2)
    cost3 = tf.reduce_mean(-outputTensor3)
    cost_target = tf.reduce_mean(-outputTensor_target)
    cost_test = tf.reduce_mean(-outputTensor_test)

    listOfVariableTensors = model.inputs[0]
    listOfVariableTensors2 = model2.inputs[0]
    listOfVariableTensors3 = model3.inputs[0]
    listOfVariableTensors_target = model_target.inputs[0]
    listOfVariableTensors_test = model_test.inputs[0]


    gradients_cost_seq = K.gradients(cost, listOfVariableTensors)[0]
    gradients_cost_seq2 = K.gradients(cost2, listOfVariableTensors2)[0]
    gradients_cost_seq3 = K.gradients(cost3, listOfVariableTensors3)[0]
    gradients_cost_seq_target = K.gradients(cost_target, listOfVariableTensors_target)[0]

    gradients_cost_seq_expanded = tf.pad(gradients_cost_seq, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
    gradients_cost_seq_expanded2 = tf.pad(gradients_cost_seq2, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
    gradients_cost_seq_expanded3 = tf.pad(gradients_cost_seq3, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
    gradients_cost_seq_expanded_target = tf.pad(gradients_cost_seq_target, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)


    grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded)[0]
    grad_cost_latent2 = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded2)[0]
    grad_cost_latent3 = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded3)[0]
    grad_cost_latent_target = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded_target)[0]

    noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=1e-5)
    global_step = tf.Variable(args.lr * 1e-1, trainable=False)
    session.run(global_step.initializer)
    tf.add_to_collection('global_step', global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=global_step)

    design_op = optimizer.apply_gradients([(grad_cost_latent + noise + grad_cost_latent2, latent_vars)])
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
    session.run(adam_initializers)
    tf.add_to_collection('design_op', design_op)
    s = session.run(tf.shape(latents))

    '''
    Optimization takes place here.
    '''
    bind_scores_list = []
    bind_scores_list2 = []
    bind_scores_list_target = []
    bind_scores_means = []
    bind_scores_means2 = []
    bind_scores_means_total = []
    bind_scores_means_target = []
    sequences_list = []
    max_iters=args.n

    for opt_iter in tqdm(range(max_iters)):
        generated_sequences = session.run(gen_output)
        generated_sequences = probs_to_chars(generated_sequences)
        ohe_genseqs = np.array([one_hot_encode(x) for x in generated_sequences])
        bind_scores = model.predict(ohe_genseqs)
        bind_scores2 = model2.predict(ohe_genseqs)
        bind_scores3 = model3.predict(ohe_genseqs)

        bind_scores_target = model_target.predict(ohe_genseqs)

        wtn_inp_scores = np.stack((bind_scores, bind_scores2, bind_scores3), 1) 
        wtn_out_weights = wtn_model.predict(wtn_inp_scores)

        sequences_list.append(generated_sequences)

        evaluated_gradients_cost_seq = session.run(gradients_cost_seq,feed_dict={model.input:ohe_genseqs})
        evaluated_gradients_cost_seq2 = session.run(gradients_cost_seq2,feed_dict={model2.input:ohe_genseqs})
        evaluated_gradients_cost_seq3 = session.run(gradients_cost_seq3,feed_dict={model3.input:ohe_genseqs})
        evaluated_gradients_cost_seq_target = session.run(gradients_cost_seq_target,feed_dict={model_target.input:ohe_genseqs})

        evaluated_gradients_cost_seq_expanded = session.run(gradients_cost_seq_expanded, feed_dict={model.input:ohe_genseqs})
        evaluated_gradients_cost_seq_expanded2 = session.run(gradients_cost_seq_expanded2, feed_dict={model2.input:ohe_genseqs})
        evaluated_gradients_cost_seq_expanded3 = session.run(gradients_cost_seq_expanded3, feed_dict={model3.input:ohe_genseqs})
        evaluated_gradients_cost_seq_expanded_target = session.run(gradients_cost_seq_expanded_target, feed_dict={model_target.input:ohe_genseqs})

        evaluated_gradients_cost_latent = session.run(grad_cost_latent, feed_dict={model.input:ohe_genseqs}) # sox 15
        evaluated_gradients_cost_latent2 = session.run(grad_cost_latent2, feed_dict={model2.input:ohe_genseqs}) #  sox 14
        evaluated_gradients_cost_latent3 = session.run(grad_cost_latent3, feed_dict={model3.input:ohe_genseqs}) # sox 7
        evaluated_gradients_cost_latent_target = session.run(grad_cost_latent_target, feed_dict={model_target.input:ohe_genseqs}) # sox2
        
        desop = (evaluated_gradients_cost_latent * weights[0])
        desop1 = (evaluated_gradients_cost_latent2 * weights[1])
        desop2 = (evaluated_gradients_cost_latent3 * weights[2])

        design_op = optimizer.apply_gradients([(desop + desop1 + desop2 + noise, latent_vars)])
        session.run(design_op)

        bind_scores_list.append(bind_scores)
        bind_scores_list2.append(bind_scores2)
        bind_scores_list_target.append(bind_scores_target)
        
        bind_scores_means.append(np.mean(bind_scores))
        bind_scores_means2.append(np.mean(bind_scores2))
        bind_scores_means_target.append(np.mean(bind_scores_target))
        bind_scores_means_total.append(np.mean(bind_scores + bind_scores2))


    ohe_genseqs = np.array([one_hot_encode(x) for x in generated_sequences])
    bind_scores_test = model_test.predict(ohe_genseqs) # sox15 - validation protein - closest

    dirname='./output/'+protein_name+"_inv_distance_softmax_method"+"_maxiters_"+str(max_iters)+"/"
    os.mkdir(dirname)
    with open(dirname+protein_name+'_best_binding_sequences.txt', 'w') as f:
        for item in sequences_list[np.argmax(bind_scores_means_total)]:
            f.write(f"{item}\n")
    with open(dirname+protein_name+'_initial_sequences.txt', 'w') as f:
        for item in sequences_list[0]:
            f.write(f"{item}\n")


    np.savetxt(dirname+f"{protein_name}_initial_binding_scores"+".txt", bind_scores_list_target[0])
    np.savetxt(dirname+f"{protein_name}_best_binding_scores"+".txt", bind_scores_list_target[np.argmax(bind_scores_means_target)])

    if MEASURE_TEST:
        np.savetxt(dirname+f"{protein_name}_test_binding_scores"+".txt", bind_scores_test)
    
    prebind = bind_scores_list_target[0]
    postbind = bind_scores_list_target[np.argmax(bind_scores_means_target)]

    sortinds = np.flip(postbind.argsort())
    prebind = prebind[sortinds]
    postbind = postbind[sortinds]


