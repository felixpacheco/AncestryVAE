#!/usr/bin/env python
"""client.py: argument parser for the ancestry_vae package"""

import argparse 

def argument_parser(argv_list):
    """Takes a list of arguments and outputs parameters or settings for the VAE
    Parameters
    ----------
    argv_list   : list of command line arguments

    Returns
    -------
    args         : list of arguments from the user
    """    
    parser=argparse.ArgumentParser()

    parser.add_argument("--name",default=None,
        help="name of dataset or analysis")

    parser.add_argument("--infile",
                        help="path to input (BED or pytorch tensors)")

    parser.add_argument("--out",default="vae",
                        help="Path for saving output")

    parser.add_argument("--patience",default=50,type=int,
                        help="training patience, default=1000")

    parser.add_argument("--max_epochs",default=500,type=int,
                        help="max training epochs, default=50")

    parser.add_argument("--batch_size",default=20,type=int,
                        help="Batch size, default=20")

    parser.add_argument("--save_model",default=False,action="store_true",
                        help="Save model as model.pt")

    parser.add_argument("--cpu",default=False,action="store_true",
                        help="Use GPU for computation")

    parser.add_argument("--seed",default=1,type=int,help="random seed, default: 1")

    parser.add_argument("--train_prop",default=0.8,type=float,
                        help="proportion of samples to use for training, default: 0.8")

    parser.add_argument("--grid_search",default=False,action="store_true",
                        help='run grid search over network sizes and use the network with \
                              minimum test loss. default: False. ')

    parser.add_argument("--depth",default=3,type=int,
                        help='number of hidden layers, default=3')

    parser.add_argument("--width",default=1000,type=int,
                        help='nodes per hidden layer. default=1000')

    parser.add_argument("--latent_dim",default=2,type=int,
                        help="N latent dimensions to fit, default: 2")

    parser.add_argument("--plot",default=False,action="store_true",
                        help="generate an interactive scatterplot of the latent space. requires --metadata. Run python scripts/plotvae.py --h for customizations")

    parser.add_argument("--metadata",default=None,
                        help="path to tab-delimited metadata file with column 'sampleID'.")

    parser.add_argument("--extra_annot",default="",
                        help="extra annotation at the end of the out file")

    parser.add_argument("--project_data",default=False,action="store_true",
                        help="Project data on pre-trained model")
    
    parser.add_argument("--model_path",
                        help="path to pre-trained model")

    return parser.parse_args(argv_list)


#python vae.py --name human_origins --infile /users/data/pr_00006/pytorch_input/human_origins_QC --out /users/home/felpac/ancestry_vae --max_epochs 50 --batch_size 20 --seed 112 --plot --metadata /users/home/felpac/pop_cielab_2/data/metadata/v44_metadata_clear.tsv

#python vae.py --name 14_02_balanced --infile /users/data/pr_00006/CHB_DBDS/pytorch_input/COVID_balanced_subset/ --out /users/home/felpac/dim_ancestry/ancestry_vae --max_epochs 50 --batch_size 20 --seed 234 --plot --metadata /users/data/pr_00006/CHB_DBDS/metadata/metadata_clear_COVID.tsv --latent_dim 2 --extra_annot 2.0

#python vae.py --name 08_03_balanced --infile /users/data/pr_00006/CHB_DBDS/pytorch_input/COVID_balanced_subset/ --out /users/home/felpac/dim_ancestry/ancestry_vae --max_epochs 1 --batch_size 20 --seed 234 --plot --metadata /users/data/pr_00006/CHB_DBDS/metadata/metadata_clear_COVID.tsv --latent_dim 2 --extra_annot test