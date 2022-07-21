#!/usr/bin/env pythonf
"""methods.py: script that contains supporting functions"""

import os
import pickle
import time
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from mpl_toolkits.mplot3d import Axes3D
import regex as re
import allel

def init_gpu(SEED, CUDA, GPU):
    """Initialize GPU on pytorch framework
    Parameters
    ----------
    SEED : Seed random state, (int)
    GPU : True/False
    CUDA : Availability of GPU (True/False)
    Returns
    -------
    device : cuda or cpu
    kwargs : 
    """
    torch.manual_seed(SEED)
    
    # Allow use of cpu memory
    if GPU is True:
        torch.cuda.manual_seed(SEED)
        device = torch.device("cuda" if CUDA else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True}
    
    # Allow use of cpu memory
    else :
        device = torch.device("cpu")
        kwargs = {}

    return device, kwargs

def ingest_data(infile):
    """Ingest data, if vcf converts it to torch tensors
    Parameters
    ----------
    infile : path to the vcf file
    Returns
    -------
    infile : path to the tensor directory
    """
    infile = os.path.normpath(infile)
    if re.search(r"\S+.vcf", infile):
        print("-- Reading vcf file and creating torch tensors --")
        # Create dir where to _zdims torch tensors
        pytorch_tensors = os.path.dirname(infile)+"/pytorch_tensors_"+os.path.basename(infile)[:-4]
        
        try:
            os.mkdir(pytorch_tensors)
        
        except Exception as e:
            if os.path.exists(pytorch_tensors) is True:
                return pytorch_tensors+"/"
                
            else:
                sys.exit("Error! Creation of the results directory failed")

        vcf_to_pt(infile, pytorch_tensors)
        print(f"    Done    ")
        
        return pytorch_tensors+"/"

    # The input is not a vcf file, return input path
    else:
        return infile+"/"

def vcf_to_pt(inpath, outpath):
    """Takes the path to a vcf file and creates a pytorch tensor for each observation
    Parameters
    ----------
    inpath : path to input file
    Returns
    -------
    outpath : None
    """
    # Read file
    start_time = time.time()

    # Read vcf
    vcf=allel.read_vcf(inpath,log=sys.stderr)
    samples = vcf["samples"]
    features = vcf["variants/ID"]

    # Save snp list
    with open(outpath+"/"+"features", "wb") as fp:
            pickle.dump(features, fp, protocol=3)

    # Save ID list
    for i in range(len(samples)):
        # ID and FAMID and separated by _, we just need the ID
        samples[i] = samples[i].split("_")[0]

    with open(outpath+"/"+"sample_ids", "wb") as fp:
            pickle.dump(samples, fp, protocol=3)

    # Get id of samples
    gen = allel.GenotypeArray(vcf["calldata/GT"])
    gen_array = gen.values

    # Split numpy array into chunks per observation
    split_array = np.hsplit(gen_array, gen_array.shape[1])

    # Save each observation as tensor
    for i in range(len(split_array)):
        # replace -1 to np.nan and convert to float
        split_array[i] = np.where(split_array[i] == -1, np.nan, split_array[i])
        split_array[i] = split_array[i].reshape((split_array[i].shape[0], split_array[i].shape[2]))
        split_array[i] = split_array[i].astype(np.float)
        
        # Save as tensor
        gen_tensor = torch.from_numpy(split_array[i])
        torch.save(gen_tensor, outpath+"/"+str(i))

    print("--- %s seconds --- for converting vcf to tensors" % (time.time() - start_time))
    return None

def get_inputs(infile, metadata, plot_flag):
    """Takes path to input files and metadata and outputs a series of objects 
    (input data, feature names, targets, metadata, ancestry labels)
    Parameters
    ----------
    infile : path to input files
    metadat : path to metadata
    Returns
    -------
    X : torch tensor containing the genotype array data
    enc_targets :
    targets : 
    features : ids of each variant
    metadata : dataframe containing the metadata
    ancestries : list of labels explaining ancestry
    """
    # Create list of files in infile
    files = os.listdir(infile)
    files.sort()

    X = files[:-2]
    enc_targets = np.array(X).astype(int)

    targets = files[-2]

    # open targets file (ids of each file)
    file = open(infile+"/"+targets, 'rb')
    targets = pickle.load(file)
    file.close()
    
    #open names of each variant
    features = files[-1]
    file = open(infile+"/"+features, 'rb')
    targets = pickle.load(file)

    if plot_flag is False:
        return X, enc_targets, targets, features, None, None

    # read metadata dataframe
    metadata = pd.read_csv(metadata, sep="\t") 
    # create list of ancestry labels for each input
    ancestries = metadata[metadata.id.isin(targets)].ancestry
    return X, enc_targets, targets, features, metadata, ancestries

def binary_encoding(list_to_encode):
    """Binary encoding of a list of integers
    Parameters
    ----------
    list_to_encode : list containing int values
    Returns
    -------
    one_hot_encoded : one-hot encoded numpy array
    """
    # Initialize encoded array
    array = np.array(list_to_encode)
    encoded_array = np.zeros((len(array),2))

    # Get index of every case
    index_0 = np.where(array == 0)
    index_1 = np.where(array == 1)
    index_2 = np.where(array == 2)
    #index_nan = np.where(array == np.nan)

    # Replace values
    encoded_array[index_1,0] = np.ones_like(encoded_array[index_1,0])
    encoded_array[index_2,] = np.ones_like(encoded_array[index_2,])
    #encoded_array[index_nan,] = np.zeros_like(encoded_array[index_nan,])
    return encoded_array


def split_train_test(X, targets, prop):
    """Splits X and targets np.arrays into train and test according to prop
    Parameters
    ----------
    X : vector of inputs
    targets : vector of targets
    prop : floating point corresponding to the training test partition
    
    Returns
    -------
    X_train : X * prop
    X_test X * (1-prop)
    targets_train : targets * prop
    targets_test : targets * (1-prop)

    X_train = X[:int(len(X)*prop)]
    X_test = X[int(len(X)*prop+1):]
    
    y_train = targets[:int(len(X)*prop)]
    y_test = targets[int(len(X)*prop+1):]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, targets, train_size= prop, test_size=1-prop, random_state=42)
    return X_train, X_test, y_train, y_test


def get_enc_dict(original_targets, targets):
    """Takes a list of encoded targets and original strings and makes a dictionary of the encoding
    Parameters
    ----------
    original_targets : str targets
    targets : encoded targets
    
    Returns
    -------
    dictionary with encoding mapping
    """
    targets = tuple(map(tuple, targets))

    # Initialize df
    df = pd.DataFrame({"targets":list(targets), "original_targets":original_targets})
    df = df.drop_duplicates(subset=["original_targets"])

    # Create dict 
    dict_encoding = defaultdict()
    dict_encoding = pd.Series(df.original_targets.values, index=df.targets.values).to_dict()
    return dict_encoding


def loss_ignore_nans(device, loss, x):
    #Flatten both tensors
    shape = loss.shape
    loss = loss.flatten()
    x = x.flatten()

    # Get indices of missing values
    if device != "cpu":
        idx_nan = (x.cpu()!=x.cpu()).nonzero().cuda()
    else :
        idx_nan = x!=x.nonzero()

    # Multiply the loss of those indices by 0
    ignore_tensor = torch.ones(loss.shape)
    ignore_tensor[idx_nan] = 0.0
    if device != "cpu":
        ignore_tensor = ignore_tensor.to(device)
    loss = loss.mul(ignore_tensor)
    loss = torch.reshape(loss, shape)
    return loss


def de_encoding(enc_targets, target_ids):
    """Uses a dictionary to de-encode the target values
    Parameters
    ----------
    enc_targets     : np.array of encoded target values
    dict_encoding   : dictionary containing encoding pairing
    Returns
    -------
    df              : list of non-encoded target values
    """
    # Index target list with encoded integers
    targets = [target_ids[i] for i in enc_targets]
    return targets

def mk_results_dir(out_path, args, train_flag):
    """Take the output path and creates a dir structure to save results
    Parameters
    ----------
    out_path : path to send results
    results_version : id of specific set of parameters
    
    Returns
    -------
    None
    """

    results_id = f"{args.name}_projection"
    
    if train_flag is True:
        results_id = f"{args.name}_batch{args.batch_size}_zdims{args.latent_dim}_train{args.train_prop}_hidden{args.width}x{args.depth}"


    # Create results dir
    try:
        os.mkdir(out_path+"/results")
    except: 
        if os.path.exists(out_path+"/results") is True:
            pass
        else:
            sys.exit("Error! Creation of the results directory failed")
    
    # Create results dir for set of hyperparamenters
    try:
        os.mkdir(out_path+"/results/"+results_id)
    except Exception as e:
        if os.path.exists(out_path+"/results/"+results_id) is True:
            pass
        else:
            sys.exit("Error! Creation of the results directory failed")

    # Create subdirs for data and plots
    try:
        os.mkdir(out_path+"/results"+"/"+results_id+"/data")
        os.mkdir(out_path+"/results"+"/"+results_id+"/plots")

    except Exception as e:
        if os.path.exists(out_path+"/results"+"/"+results_id+"/data") is True or os.path.exists(out_path+"/results"+"/"+results_id+"/plots") is True:
            pass
        else:
            sys.exit("Error! Creation of the results directory failed")
        
    return results_id

def impute_data(tensor, batch_size):
    """Replaces missing values in tensor by mean frequency
    Parameters
    ----------
    X : tensor 
    batch_size : integer with amount of observations per batch
    
    Returns
    -------
    tensor with replaced values
    """
    index_nan = (tensor!=tensor).nonzero()


    tensor = tensor.cpu()
    shape = tensor.shape

    # Get index X == nan and frequencies for those
    #index_nan = (tensor!=tensor).nonzero()
    tensor =  np.array(tensor)
    index_nan = np.where(np.isnan(tensor))
    tensor[index_nan] = np.zeros_like(tensor[index_nan])
    tensor = torch.FloatTensor(tensor)

    assert len((tensor!=tensor).nonzero()) == 0
    return tensor


def save_diagnostics(out=None, results_id=None, train_loss_values=None, train_bce=None, train_kld=None, lr_train_values=None, test_loss_values=None,test_bce=None,test_kld=None,cosine_sim_list=None, df_latent_train=None,df_latent_test=None, epoch=None):
    """Saves loss for training and test(BCE, KLD and their sum), learning rate and
    cosine similarity between consecutive iterations of the latent space 
    Parameters
    ----------
    out : path to output results (str)
    results_id : name of the analysis (str)
    epoch : current epoch

    train_loss_values : list of train loss values until epoch n
    train_bce : list of train bce loss values until epoch n
    train_kld : list of train kld loss values until epoch n
    lr_train_values : list of learning rate values until epoch n
    df_latent_train : df containing latent spaces values for train set
    
    test_loss_values : list of test loss values until epoch n
    test_bce : list of test bce loss values until epoch n
    test_kld : list of test kld loss values until epoch n
    df_latent_test: df containing latent spaces values for test set
    cosine_sim_list : list cosine similarity between consecutive iterations of the latent space 

    Returns
    -------
    None 
    """
    if train_loss_values is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/train_loss.json", "wb") as fp:
            pickle.dump(train_loss_values, fp, protocol=3)
    
    if lr_train_values is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/lr_extra.json", "wb") as fp:
            pickle.dump(lr_train_values, fp, protocol=3)

    if test_loss_values is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/test_loss.json", "wb") as fp:
            pickle.dump(test_loss_values, fp, protocol=3)

    
    if train_bce is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/train_bce.json", "wb") as fp:
            pickle.dump(train_bce, fp, protocol=3)

    if train_kld is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/train_kld.json", "wb") as fp:
            pickle.dump(train_kld, fp, protocol=3)
    
    if test_bce is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/test_bce.json", "wb") as fp:
            pickle.dump(test_bce, fp, protocol=3)

    if test_kld is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/test_kld.json", "wb") as fp:
            pickle.dump(test_kld, fp, protocol=3)

    if cosine_sim_list is not None:
        with open(str(out)+"/results/"+str(results_id)+"/data/cosine_sim.json", "wb") as fp:
            pickle.dump(cosine_sim_list, fp, protocol=3)

    if df_latent_test is not None:
        df_latent_test.to_csv(str(out)+"/results/"+str(results_id)+"/data/test_latent_space_epoch"+str(epoch)+".csv")

    if df_latent_train is not None:
        df_latent_train.to_csv(str(out)+"/results/"+str(results_id)+"/data/train_latent_space_epoch"+str(epoch)+".csv")

    return None


def training_report(train_loss, bce_train, kld_train, lr_train, cos_sim, out):
    """Plots a training report
    Parameters
    ----------
    out : path to output results (str)

    train_loss : list of train loss values until epoch n
    bce_train : list of train bce loss values until epoch n
    kld_train : list of train kld loss values until epoch n
    lr_train : list of learning rate values until epoch n
    cosine_sim : list cosine similarity between consecutive iterations of the latent space 

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(14, 14)
    # Train loss
    ax[0, 0].plot(train_loss, 'b')
    ax[0, 0].set_title("Training loss")

    # BCE loss
    ax[0, 1].plot(bce_train, "g")
    ax[0, 1].set_title("Training reconstruction error")

    # For Tangent Function
    ax[1, 0].plot(kld_train, "r")
    ax[1, 0].set_title("Training KLD")

    # For Tangent Function
    ax[1, 1].plot(lr_train, "y")
    ax[1, 1].set_title("Learning rate over training")

    plt.tight_layout()
    plt.savefig(out+".png")
    plt.clf()

    plt.plot(cos_sim, 'b')
    plt.title('Cosine similarity per epoch')
    plt.xlabel('epochs')
    plt.ylabel('Cosine Similarity')
    plt.savefig(out+"_cosine.png")
    plt.clf()

    return None

def test_report(test_loss, test_bce, test_kld, out):
    """Plots a test report
    Parameters
    ----------
    out : path to output results (str)

    test_loss : list of test loss values until epoch n
    bce_test : list of test bce loss values until epoch n
    kld_train : list of test kld loss values until epoch n

    Returns
    -------
    None 
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(14, 14)
    # Train loss
    ax[0, 0].plot(test_loss, 'b')
    ax[0, 0].set_title("Test loss")

    # BCE loss
    ax[0, 1].plot(test_bce, "g")
    ax[0, 1].set_title("Test reconstruction error")

    # KLD Loss
    ax[1, 0].plot(test_kld, "r")
    ax[1, 0].set_title("Test KLD")

    plt.tight_layout()
    plt.savefig(out)
    plt.clf()

    return None

def plot_latent_space(df_latent, metadata, ZDIMS ,out):
    """Plots either 2d or 3d latent space
    Parameters
    ----------
    out : path to output results (str)
    df_latent : df containing latent spaces values
    ZDIMS : integer with number of dimensions

    Returns
    -------
    None 
    """
    fig = plt.gcf()
    
    if ZDIMS == 2 :
        # Merge metadata with latent space dataframe
        df_latent = df_latent.set_axis(['z1', 'z2', 'id', 'reconstruction_error'], axis=1, inplace=False)
        metadata = pd.read_csv(metadata, sep="\t")
        metadata = metadata[["id", "ancestry"]]
        df_latent = pd.merge(df_latent, metadata, on="id")

        # Plot latent space
        sns.scatterplot(data=df_latent, x="z1", y="z2", hue="ancestry", alpha=0.75, palette="tab20", linewidth=0)
        fig.set_size_inches(16, 16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(out)
        plt.clf()
        return None
    
    else :
        fig.set_size_inches(14, 14)
        # Merge metadata with latent space dataframe
        df_latent = df_latent.set_axis(['z1', 'z2', 'z3', 'id', 'reconstruction_error'], axis=1, inplace=False)
        metadata = pd.read_csv(metadata, sep="\t")
        metadata = metadata[["id", "ancestry"]]
        df_latent = pd.merge(df_latent, metadata, on="id")

        fig = plt.figure()
        ax = Axes3D(fig)

        palette = sns.color_palette("tab20")
        ancestries = df_latent.ancestry.unique()
        color_dict = dict()
        for i in range(len(ancestries)) :
            color_dict[ancestries[i]] = palette[i]
            
        color_array = list()
        for i in df_latent.ancestry:
            color_array.append(color_dict[i])

        sc = ax.scatter(df_latent.z1, df_latent.z2, df_latent.z3, alpha=0.75, c=color_array)
        ax.set_xlabel('z1')
        ax.set_ylabel('z2')
        ax.set_zlabel('z3')
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=df_latent.ancestry)
        plt.tight_layout()
        plt.savefig(out)
        plt.clf()
        return None




