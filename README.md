# RNAGEN


# A generative adversarial network-based model to generate synthetic RNA sequences to target proteins



> RNAGEN is a is a deep learning based model for novel piRNA generation and optimziation. We use the WGAN-GP architecture for the generative model, and the DeepBind models for optimizing binding of the generated piRNA sequences to the target protein. To find the closest relatives of a target protein to be used in optimization, we use the Prot2Vec model.

> <a href="https://en.wikipedia.org/wiki/Deep_learning" target="_blank">**Deep Learning**</a>,<a href="https://arxiv.org/pdf/1704.00028v3.pdf" target="_blank">**WGAN-GP**</a>, <a href="https://www.nature.com/articles/nbt.3300." target="_blank">**DeepBind**</a>, <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN" target="_blank">**Prot2Vec**</a>

> Diagram of the generative model and the optimization procedure

<p align="center">
<img src="./figures/diagram.png"   class="center"><br>

<!-- > Diagram of the optimization procedure for ribosome load and gene expression

<p align="center">
<img src="./figures/optimization.png"   class="center"><br> -->

---

## Authors

Furkan Ozden, Sina Barazandeh, Dogus Akboga, Urartu Ozgur Safak Seker, A. Ercument Cicek

---

## Questions & comments 

[secondauthorname].[secondauthorsurname]@bilkent.edu.tr

[firstcorrespondingauthorfirstname]@bilkent.edu.tr

[secondcorrespondingauthorsurname]@cs.bilkent.edu.tr


---


## Table of Contents 

> Warning: Please note that the RNAGEN model is completely free for academic usage. However it is licenced for commercial usage. Please first refer to the [License](#license) section for more info.

- [Installation](#installation)
- [Features](#features)
- [Instructions Manual](#instructions-manual)
- [Usage Examples](#usage-examples)
- [Citations](#citations)
- [License](#license)


---

## Installation

- RNAGEN is easy to use and does not require installation. The scripts can be used if the requirements are installed.

### Requirements

For easy requirement handling, you can use RNAGEN.yml files to initialize conda environment with requirements installed:

```shell
$ conda env create --name rnagen -f RNAGEN.yml
$ conda activate rnagen
```

Note that the provided environment yml file is for Linux systems. For MacOS users, the corresponding versions of the packages might need to be changed.
---

## Features

- RNAGEN components are trained using GPUs and GPUs are used for the project. However, depending on the type of Tensorflow <a href="https://www.tensorflow.org/" target="_blank">**Tensorflow**</a> the model can run on both GPU and CPU. The run time on CPU is considerably longer compared to GPU.


## Instructions Manual
Important notice: Please call the wgan.py script from the ./src/gan directory. The optimization scripts for gene expression and MRL are in the ./src/gene_optimization and ./src/mrl_optimization directories, respectively. To analyze the generated seqeunces use the ./src/analysis/analyze.py script.

### Train teh GAN modes:

```shell
$ python ./train_rnagen.py
```


#### Optional Arguments

##### -mil, --min_length
- Minimum length of the piRNAs used for training the model. Default: 26.

##### -mxl, --max_length
- Maximum length of the piRNAs used for training the model. Default: 32.

##### -d, --dataset
- The path to file including the piRNA samples. The default path used is './data/piRNAs.fa'.

##### -lr, --learning_rate
- The learning rate of the Adam optimizer used to optimize the model parameters. The default value is 1e-5. If 4 is provided, the learning rate will be 1e-4. 

##### -bs, --batch_size
- The batch size used to train the model, the default value is 64 

##### -dim, --dimension
- The dimension of the input noise, with the default value of 40.

##### -g, --gpu
- The ID of the GPU that will be used for training. The default setting is used otherwise.
- The same optional argument applies to the optimization and analysis scripts.

### Optimize for a target Protein using 3 Relative Proteins:

```shell
$ python ./optimize.py
```

#### Required Arguments

##### -dp, --deepbind_path
- The path to a text file including paths of the deepbind models of the relative proteins

##### -pn, --protein_name
- The name of the protine

##### -lr, --learning_rate
- The learning rate of the Adam optimizer used to optimize the model parameters. The default value is 3e-5. 

##### -n, --n_iterations
- The number of iterations the optimization is performed. The default value is 3,000 iterations.

##### -rc, --rna_count
- The number of piRNA sequences generated and optimized. Default: 64.

## Usage Examples

> Usage of RNAGEN is very simple. You need to install conda to install the specific environment and run the scripts.

### Step-0: Install conda package management

- This project uses conda package management software to create virtual environment and facilitate reproducability.

- For Linux users:
 - Please take a look at the <a href="https://repo.anaconda.com/archive/" target="_blank">**Anaconda repo archive page**</a>, and select an appropriate version that you'd like to install.
 - Replace this `Anaconda3-version.num-Linux-x86_64.sh` with your choice

```shell
$ wget -c https://repo.continuum.io/archive/Anaconda3-vers.num-Linux-x86_64.sh
$ bash Anaconda3-version.num-Linux-x86_64.sh
```


### Step-1: Set Up your environment.

- It is important to set up the conda environment which includes the necessary dependencies.
- Please run the following lines to create and activate the environment:

```shell
$ conda env create --name rnagen -f RNAGEN.yml
$ conda activate rnagen
```

## Citations

---

## License


- **[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)**
- Copyright 2023 © RNAGEN.
- For commercial usage, please contact.