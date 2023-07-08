import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import pdb
import keras
from keras import backend as K
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras.layers import Concatenate, LeakyReLU, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.regularizers import l2
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.special import softmax
# from rnasamba.core.model import get_rnasamba_model
# from rnasamba.core.inputs import RNAsambaInput

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
        #in the orignal deepbind model, these weights are interleaved.
        #what a nightmare.
        weights1 = weights1.reshape((num_detectors,2,-1))
        new_weights1 = np.zeros((2*num_detectors, weights1.shape[-1]))
        new_weights1[:num_detectors, :] = weights1[:,0,:]
        new_weights1[num_detectors:, :] = weights1[:,1,:]
        weights1 = new_weights1
    biases1 = np.array([float(x) for x in data[9].split(" = ")[1].split(",")]).reshape(
                        (1 if num_hidden==0 else num_hidden))
    if (num_hidden > 0):
        #print("Model has a hidden layer")
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

    
    input_tensor = keras.layers.Input(shape=(None,4))
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

'''
This script is a major component of the project RNAGEN by CicekLab.
This script needs: i) Trained RNA sequences generator (WGAN), ii) DeepBind model weights for the desired protein. 
The script basically optimizes generated RNA sequences to have maximum binding scores to desired proteins. 
'''

trained_gan_path = "/home/sina/ml/RNAGEN/scripts/logs/gan_test/2020.08.17-14h04m49s_neo/checkpoints/checkpoint_199950/trained_gan.ckpt.meta"

''' SOX Family Optimization'''
deepbind_model_path ="/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00640.003.txt" #sox15 - d = 40.81 
deepbind_model_path_2 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00639.003.txt" #sox14 - d = 44.86
deepbind_model_path_3 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00647.002.txt" #sox7 - d = 54.14
deepbind_model_path_4 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00644.003.txt" #sox21 - d = 55.85
deepbind_model_path_5 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00645.006.txt" # SOX2 target protein 

''' SRS Family Optimization'''
deepbind_model_path ="/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00153.001.txt" #SRSF2 - d = 43.72 
deepbind_model_path_2 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00154.001.txt" #SRSF7 - d = 48.67
deepbind_model_path_3 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00169.001.txt" #SRSF1 - d = 52.88
deepbind_model_path_4 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00148.002.txt" #SRSF9 - d = 71.86
deepbind_model_path_5 = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00162.002.txt" # SRSF10 target protein 

deepbind_model_eval = "/home/sina/ml/RNAGEN/deepbind_models/deepbind/db/params/D00645.006.txt" # SOX2 target protein 

dist15 = 43.72
dist14 = 48.67
dist7 = 52.88
dist21 = 71.86

weights = softmax([dist15,dist14,dist7])

protein_name = "SOX4-SOX2"

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

'''
load the trained WGAN model
'''

#session = tf.Session()
session = K.get_session()
gen_handler = tf.train.import_meta_graph(trained_gan_path, import_scope="generator")
gen_handler.restore(session, trained_gan_path[:-5])

latents = tf.get_collection('latents')[0]
gen_output = tf.get_collection('outputs')[0]
batch_size, latent_dim = session.run(tf.shape(latents))
latent_vars = [c for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'generator/latent_vars' in c.name][0]


'''
TO DO: Get DeepBind Model Ready !!!
'''
model = convert_model(file_name = deepbind_model_path, input_len=32) # sox15
model2 = convert_model(file_name = deepbind_model_path_2, input_len=32) # sox14
model3 = convert_model(file_name = deepbind_model_path_3, input_len=32) # sox7
model4 = convert_model(file_name = deepbind_model_path_4, input_len=32) # sox21
model_target = convert_model(file_name = deepbind_model_path_5, input_len=32) # sox2 - target protein
model_test = convert_model(file_name = deepbind_model_eval, input_len=32)

onehot_sequences = onehot_encode_sequences(
        ['AGGUAAUAAUUUGCAUGAAAUAACUUGGAGAGGAUAGC',
         'AGACAGAGCUUCCAUCAGCGCUAGCAGCAGAGACCAUU',
         'GAGGTTACGCGGCAAGATAA',
         'TACCACTAGGGGGCGCCACC'])
#res = model.predict(np.array(onehot_sequences[0:2])[:,:,:])



'''
Random noise for initial latent space of generator. Drawn from standard normal distribution.
''' 
start_noise = np.random.normal(size=[batch_size, latent_dim])
session.run(tf.assign(latent_vars, start_noise))

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
WEIGHT TUNER NETWORK definiton. 
'''

wtn_inps = keras.layers.Input(shape=(3,))
batchs = K.shape(wtn_inps)[0]

constants = np.asarray([44.86, 54.14, 55.85]).reshape((1,-1))
k_consts = K.variable(constants)
k_consts = K.tile(k_consts, (batchs,1))
fixed_inp = keras.layers.Input(tensor = k_consts)

wtn_concated_inps = keras.layers.concatenate([fixed_inp, wtn_inps])
wtn_hidden1 = keras.layers.Dense(6, activation='relu')(wtn_concated_inps)
wtn_hidden2 = keras.layers.Dense(12, activation='relu')(wtn_hidden1)
wtn_out = keras.layers.Dense(3, activation='softmax')(wtn_hidden2)
wtn_model = keras.models.Model(inputs = [wtn_inps, fixed_inp], outputs = wtn_out)

'''
#################
'''
# pred_input = model.input
# bind_scores = model.predict(ohe_genseqs)
# cost = tf.reduce_mean(-bind_scores)

outputTensor = model.output # sox15
outputTensor2 = model2.output # sox14
outputTensor3 = model3.output  # sox7
outputTensor4 = model4.output  # sox21
outputTensor_target = model_target.output  # sox2 - target protein
outputTensor_test = model_test.output

cost = tf.reduce_mean(-outputTensor) # validation protein, cost for WTN weights
cost2 = tf.reduce_mean(-outputTensor2)
cost3 = tf.reduce_mean(-outputTensor3)
cost4 = tf.reduce_mean(-outputTensor4)
cost_target = tf.reduce_mean(-outputTensor_target)
cost_test = tf.reduce_mean(-outputTensor_test)

listOfVariableTensors = model.inputs[0]
listOfVariableTensors2 = model2.inputs[0]
listOfVariableTensors3 = model3.inputs[0]
listOfVariableTensors4 = model4.inputs[0]
listOfVariableTensors_target = model_target.inputs[0]
listOfVariableTensors_test = model_test.inputs[0]

listOfVariableTensors_WTN = wtn_model.trainable_weights[0]
gradients_WTN = K.gradients(cost, listOfVariableTensors_WTN)

gradients_cost_seq = K.gradients(cost, listOfVariableTensors)[0]
gradients_cost_seq2 = K.gradients(cost2, listOfVariableTensors2)[0]
gradients_cost_seq3 = K.gradients(cost3, listOfVariableTensors3)[0]
gradients_cost_seq4 = K.gradients(cost4, listOfVariableTensors4)[0]
gradients_cost_seq_target = K.gradients(cost_target, listOfVariableTensors_target)[0]

gradients_cost_seq_expanded = tf.pad(gradients_cost_seq, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
gradients_cost_seq_expanded2 = tf.pad(gradients_cost_seq2, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
gradients_cost_seq_expanded3 = tf.pad(gradients_cost_seq3, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
gradients_cost_seq_expanded4 = tf.pad(gradients_cost_seq4, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)
gradients_cost_seq_expanded_target = tf.pad(gradients_cost_seq_target, [[0,0], [0,0], [0,1]], 'CONSTANT', constant_values=0)

grad_seq_latent = tf.gradients(ys=gen_output, xs=latents)[0]
grad_seq_latent2 = tf.gradients(ys=gen_output, xs=latents)[0]
grad_seq_latent3 = tf.gradients(ys=gen_output, xs=latents)[0]
grad_seq_latent4 = tf.gradients(ys=gen_output, xs=latents)[0]
grad_seq_latent_target = tf.gradients(ys=gen_output, xs=latents)[0]

grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded)[0]
grad_cost_latent2 = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded2)[0]
grad_cost_latent3 = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded3)[0]
grad_cost_latent4 = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded4)[0]
grad_cost_latent_target = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq_expanded_target)[0]

noise = tf.random_normal(shape=[batch_size, latent_dim], stddev=1e-5)
global_step = tf.Variable(1e-1, trainable=False)
session.run(global_step.initializer)
tf.add_to_collection('global_step', global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=global_step)

# cost2 = session.run(cost,feed_dict={model.input:ohe_genseqs})
# preds = session.run(outputTensor,feed_dict={model.input:ohe_genseqs})

design_op = optimizer.apply_gradients([(grad_cost_latent + noise + grad_cost_latent2, latent_vars)])
adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]
session.run(adam_initializers)
tf.add_to_collection('design_op', design_op)
s = session.run(tf.shape(latents))
#update_pred_input = tf.assign(model.input, )
#session.run()


#grad_cost_latent = tf.gradients(ys=gen_output, xs=latents, grad_ys=gradients_cost_seq)[0]

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
max_iters=3000
for opt_iter in tqdm(range(max_iters)):
    generated_sequences = session.run(gen_output)
    generated_sequences = probs_to_chars(generated_sequences)
    ohe_genseqs = np.array([one_hot_encode(x) for x in generated_sequences])
    bind_scores = model.predict(ohe_genseqs) # sox15 - validation protein - closest
    bind_scores2 = model2.predict(ohe_genseqs) # sox14
    bind_scores3 = model3.predict(ohe_genseqs) # sox7
    bind_scores4 = model4.predict(ohe_genseqs) # sox21
    bind_scores_target = model_target.predict(ohe_genseqs) # sox2 - target protein

    wtn_inp_scores = np.stack((bind_scores2, bind_scores3, bind_scores4), 1) 
    wtn_out_weights = wtn_model.predict(wtn_inp_scores)

    sequences_list.append(generated_sequences)

    evaluated_gradients_cost_seq = session.run(gradients_cost_seq,feed_dict={model.input:ohe_genseqs})
    evaluated_gradients_cost_seq2 = session.run(gradients_cost_seq2,feed_dict={model2.input:ohe_genseqs})
    evaluated_gradients_cost_seq3 = session.run(gradients_cost_seq3,feed_dict={model3.input:ohe_genseqs})
    evaluated_gradients_cost_seq4 = session.run(gradients_cost_seq4,feed_dict={model4.input:ohe_genseqs})
    evaluated_gradients_cost_seq_target = session.run(gradients_cost_seq_target,feed_dict={model_target.input:ohe_genseqs})

    evaluated_gradients_cost_seq_expanded = session.run(gradients_cost_seq_expanded, feed_dict={model.input:ohe_genseqs})
    evaluated_gradients_cost_seq_expanded2 = session.run(gradients_cost_seq_expanded2, feed_dict={model2.input:ohe_genseqs})
    evaluated_gradients_cost_seq_expanded3 = session.run(gradients_cost_seq_expanded3, feed_dict={model3.input:ohe_genseqs})
    evaluated_gradients_cost_seq_expanded4 = session.run(gradients_cost_seq_expanded4, feed_dict={model4.input:ohe_genseqs})
    evaluated_gradients_cost_seq_expanded_target = session.run(gradients_cost_seq_expanded_target, feed_dict={model_target.input:ohe_genseqs})

    evaluated_gradients_cost_latent = session.run(grad_cost_latent, feed_dict={model.input:ohe_genseqs}) # sox 15
    evaluated_gradients_cost_latent2 = session.run(grad_cost_latent2, feed_dict={model2.input:ohe_genseqs}) #  sox 14
    evaluated_gradients_cost_latent3 = session.run(grad_cost_latent3, feed_dict={model3.input:ohe_genseqs}) # sox 7
    evaluated_gradients_cost_latent4 = session.run(grad_cost_latent4, feed_dict={model4.input:ohe_genseqs}) # sox 21
    evaluated_gradients_cost_latent_target = session.run(grad_cost_latent_target, feed_dict={model_target.input:ohe_genseqs}) # sox2
    
    desop = (evaluated_gradients_cost_latent * weights[0]) # sox15
    desop1 = (evaluated_gradients_cost_latent2 * weights[1]) # sox14
    desop2 = (evaluated_gradients_cost_latent3 * weights[2]) # sox7
    #desop3 = (evaluated_gradients_cost_latent4) # sox21

    #design_op = optimizer.apply_gradients([(evaluated_gradients_cost_latent + noise + evaluated_gradients_cost_latent2, latent_vars)])
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

dirname="SRSF10_inv_distance_softmax_method"+"_maxiters_"+str(max_iters)+"/"
os.mkdir(dirname)
with open(dirname+protein_name+'_best_binding_sequences.txt', 'w') as f:
    for item in sequences_list[np.argmax(bind_scores_means_total)]:
        f.write("%s/n" % item)
with open(dirname+protein_name+'_initial_sequences.txt', 'w') as f:
    for item in sequences_list[0]:
        f.write("%s/n" % item)


np.savetxt(dirname+"targetprotein_initial_binding_scores"+".txt", bind_scores_list_target[0])
np.savetxt(dirname+"targetprotein_test_binding_scores"+".txt", bind_scores_test)
np.savetxt(dirname+"targetprotein_best_binding_scores"+".txt", bind_scores_list_target[np.argmax(bind_scores_means_target)])
pdb.set_trace()


