#!/usr/bin/env python
"""pop_embedding.py: Implementation of VAEs for the CVD cohort"""
__author__      = "Felix Pacheco"

# Basic libraries
import os
import sys
import regex as re
import numpy as np
import time
import pickle
import pandas as pd
from sklearn.metrics import pairwise
import regex as re
import allel

# Custom methods
from dataloader import SNPLoading
from vae_module import VAE, loss_function, train, test, get_rec_error
from methods import split_train_test, get_enc_dict, de_encoding, mk_results_dir, save_diagnostics,  training_report,test_report,plot_latent_space, ingest_data, init_gpu, get_inputs
from client.client import argument_parser

import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader


#######################
### Argument parser ###
#######################

args = argument_parser(argv_list=sys.argv[1:])

##################################################
### Initialize hyper parameters, CUDA and seed ###
##################################################

# Hyperparams
CUDA = torch.cuda.is_available()
SEED = args.seed
BATCH_SIZE = args.batch_size
EPOCHS = args.max_epochs
ZDIMS = args.latent_dim
TRAIN = args.train_prop
HIDDEN_UNITS = args.width
HIDDEN_LAYERS = args.depth
infile = args.infile
METADATA = args.metadata
OUT = args.out
DATASET = args.name
EXTRA_ANNOT = args.extra_annot
PLOT = args.plot
SAVE_MODEL = args.save_model
PROJECT_DATA = args.project_data
MODEL_PATH = args.model_path
CPU = args.cpu

# Set seed and gpu
device, kwargs = init_gpu(SEED, CUDA, CPU)

###########################################################
### Interpret input file format (VCF or tensors) ###
###########################################################

infile = ingest_data(infile)

#####################################
### Map metadata to observations  ###
#####################################

results_id = mk_results_dir(OUT, args)

X, enc_targets, targets, features, metadata, ancestries = get_inputs(infile, args.metadata)
#X, enc_targets, targets, features, metadata, ancestries = X[0:1000], enc_targets[0:1000], targets[0:1000], features[0:1000], metadata[0:1000], ancestries[0:1000]

if PROJECT_DATA is False:
    ####################################
    ### Partition train and test set ###
    ####################################

    X_train, X_test, y_train, y_test = split_train_test(X, enc_targets, 0.8, ancestries)
    
    # Define train and test set
    train_set = SNPLoading(data_path=infile, data_files=X_train, targets=y_train)
    test_set = SNPLoading(data_path=infile, data_files=X_test, targets=y_test)

    # Define train and test dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Get input features and encoding len of targets
    input_features = train_set.__getitem__(1)[0].shape[0]
    target_enc_len = len(targets)

    # Call model to device, define optimizer and scheduler
    model = VAE(input_features=input_features, input_batch=BATCH_SIZE, zdims=ZDIMS, hidden_units=HIDDEN_UNITS, hidden_layers=HIDDEN_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.1, 
        patience=2, 
        verbose=True)

    train_loss_values = []
    train_bce = []
    train_kld = []

    test_loss_values = []
    test_bce = []
    test_kld = []

    lr_train_values = []
    cosine_sim_list = []

    epochs_save = [1,2,3,4,5,6,7,8,9,10,15,20,35,50]

    for epoch in range(1, EPOCHS + 1):
        train_loss_, train_bce_, train_kld_, lr_values, mu_train, targets_train, rec_train  = train(epoch, model, train_loader, CUDA, device, optimizer, scheduler, input_features, BATCH_SIZE,target_enc_len, ZDIMS)
        test_loss_, test_bce_, test_kld_, mu_test, targets_test, rec_test = test(epoch, model, test_loader, CUDA, device, input_features, BATCH_SIZE, target_enc_len, ZDIMS)
        
        train_loss_values = train_loss_values + train_loss_
        train_bce = train_bce + train_bce_
        train_kld = train_kld + train_kld_
        lr_train_values = lr_train_values + lr_values

        test_loss_values = test_loss_values + test_loss_
        test_bce = test_bce + test_bce_
        test_kld = test_kld + test_kld_

        save_diagnostics(
        out=OUT,
        results_id=results_id,
        train_loss_values=train_loss_values,
        train_bce=train_bce,
        train_kld=train_kld,
        lr_train_values=lr_train_values,
        test_loss_values=test_loss_values,
        test_bce=test_bce,
        test_kld=test_kld,
        cosine_sim_list=None,
        df_latent_train=None,
        df_latent_test=None,
        epoch=None
        )

        if epoch == 1:
                previous_mu = mu_test

        else :

            # Compute similarity between current latent space and previous epoch latent space
            cosine_matrix = pairwise.cosine_similarity(previous_mu, mu_test)
            cosine_between_epochs = np.diagonal(cosine_matrix)
            mean_cosine_sim = np.mean(cosine_between_epochs)
            cosine_sim_list.append(mean_cosine_sim)

            save_diagnostics(
                out=OUT,
                results_id=results_id,
                train_loss_values=None,
                train_bce=None,
                train_kld=None,
                lr_train_values=None,
                test_loss_values=None,
                test_bce=None,
                test_kld=None,
                cosine_sim_list=cosine_sim_list,
                df_latent_train=None,
                df_latent_test=None,
                epoch=None
                )

            # Update the scheduler depending on the similarity of latent spaces
            scheduler.step(mean_cosine_sim)

            # Assign the current epoch as the previous one
            previous_mu = mu_test

        # Save latent dimension every x epochs epochs
        if epoch in epochs_save:

            # Save train df
            targets_train = de_encoding(targets_train, targets) # The vector is shuffled so y_train does not work
            df_train = pd.DataFrame(mu_train)
            rec_train = np.asarray(rec_train)
            df_train["id"] = targets_train
            df_train["reconstruction_error"] = rec_train

            # Save test df
            targets_test = de_encoding(y_test, targets)
            df_test = pd.DataFrame(mu_test)
            rec_test = np.asarray(rec_test)
            df_test["id"] = targets_test
            df_test["reconstruction_error"] = rec_test
            
            if ZDIMS == 2:
                df_test = df_test.set_axis(['z1', 'z2', 'id', 'reconstruction_error'], axis=1, inplace=False)
                df_train = df_train.set_axis(['z1', 'z2', 'id', 'reconstruction_error'], axis=1, inplace=False)

            elif ZDIMS == 3:
                df_test = df_test.set_axis(['z1', 'z2', 'z3' ,'id', 'reconstruction_error'], axis=1, inplace=False)
                df_train = df_train.set_axis(['z1', 'z2', 'z3' ,'id', 'reconstruction_error'], axis=1, inplace=False)

            save_diagnostics(
                out=OUT,
                results_id=results_id,
                train_loss_values=None,
                train_bce=None,
                train_kld=None,
                lr_train_values=None,
                test_loss_values=None,
                test_bce=None,
                test_kld=None,
                cosine_sim_list=None,
                df_latent_train=df_train,
                df_latent_test=df_test,
                epoch=epoch
                )
            
            if PLOT is True:

                if ZDIMS <= 3 : 
                    plot_latent_space(df_test, METADATA, ZDIMS ,OUT+"/results/"+results_id+"/plots"+"/test_latent_space"+str(epoch)+".png")
                    plot_latent_space(df_train, METADATA, ZDIMS ,OUT+"/results/"+results_id+"/plots"+"/train_latent_space"+str(epoch)+".png")

                training_report(
                    train_loss_values,
                    train_bce,
                    train_kld, 
                    lr_train_values, 
                    cosine_sim_list, 
                    OUT+"/results/"+results_id+"/plots"+"/training_report")


                test_report(
                    test_loss_values,
                    test_bce,
                    test_kld, 
                    OUT+"/results/"+results_id+"/plots"+"/test_report.png")

            print(f"--> Epoch {epoch}: Saved latent values")

            if  optimizer.param_groups[0]['lr'] < 1.0e-10 or epoch==EPOCHS:

                print("### Training process completed ###")
                print("### Saving last state of training ###")

                if SAVE_MODEL is True:
                    torch.save(model.state_dict(), OUT+"/results/"+results_id+"/data/model.pt")
                            
                save_diagnostics(
                out=OUT,
                results_id=results_id,
                train_loss_values=train_loss_values,
                train_bce=train_bce,
                train_kld=train_kld,
                lr_train_values=lr_train_values,
                test_loss_values=test_loss_values,
                test_bce=test_bce,
                test_kld=test_kld,
                cosine_sim_list=cosine_sim_list,
                df_latent_train=df_train,
                df_latent_test=df_test,
                epoch=epoch
                )
                
                if ZDIMS <= 3 : 
                    plot_latent_space(df_test, METADATA, ZDIMS ,OUT+"/results/"+results_id+"/plots"+"/test_latent_space"+str(epoch)+".png")
                    plot_latent_space(df_train, METADATA, ZDIMS ,OUT+"/results/"+results_id+"/plots"+"/train_latent_space"+str(epoch)+".png")

                training_report(
                    train_loss_values,
                    train_bce,
                    train_kld, 
                    lr_train_values, 
                    cosine_sim_list, 
                    OUT+"/results/"+results_id+"/plots"+"/training_report")


                test_report(
                    test_loss_values,
                    test_bce,
                    test_kld, 
                    OUT+"/results/"+results_id+"/plots"+"/test_report.png")

                sys.exit(0)



else:

    BATCH_SIZE = int(re.search(r'batch(\d+)', MODEL_PATH).group(1))
    ZDIMS = int(re.search(r'zdims(\d+)', MODEL_PATH).group(1))
    HIDDEN_UNITS = int(re.search(r'hidden(\d+)', MODEL_PATH).group(1))
    HIDDEN_LAYERS = int(re.search(r'x(\d+)', MODEL_PATH).group(1))

    # Load data to project 
    project_set = SNPLoading(data_path=infile, data_files=X, targets=enc_targets.astype(float))
    project_loader = DataLoader(project_set, batch_size=BATCH_SIZE)

    input_features = project_set.__getitem__(1)[0].shape[0]
    target_enc_len = len(targets)

    # Load pretrained model
    model = VAE(input_features=input_features, input_batch=BATCH_SIZE, zdims=ZDIMS, hidden_units=HIDDEN_UNITS, hidden_layers=HIDDEN_LAYERS).to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    test_loss_, test_bce_, test_kld_, mu_test, targets_test, rec_test = test(1, model, project_loader, CUDA, device, input_features, BATCH_SIZE, target_enc_len, ZDIMS)

    # Save test df
    print(targets)
    print(targets_test.astype(int))
    targets_test = de_encoding(targets_test.astype(int), targets)
    df_test = pd.DataFrame(mu_test)
    rec_test = np.asarray(rec_test)
    df_test["label"] = targets_test
    df_test["reconstruction_error"] = rec_test
    
    if ZDIMS == 2:
        df_test = df_test.set_axis(['z1', 'z2', 'id', 'reconstruction_error'], axis=1, inplace=False)
        plot_latent_space(df_test, METADATA, ZDIMS ,OUT+"/results/"+results_id+"/plots"+"/test_latent_space"+".png")

    save_diagnostics(
        out=OUT,
        results_id=results_id,
        train_loss_values=None,
        train_bce=None,
        train_kld=None,
        lr_train_values=None,
        test_loss_values=None,
        test_bce=None,
        test_kld=None,
        cosine_sim_list=None,         
        df_latent_train=None,
        df_latent_test=df_test,
        epoch=1
        )

    test_report(
            test_loss_,
            test_bce_,
            test_kld_, 
            OUT+"/results/"+results_id+"/plots"+"/project_report.png")

    print("projection made")
    sys.exit(0)