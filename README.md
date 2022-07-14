# AncestryVAE

AncestryVAE is a git repository to compute VAEs on genotype array data to visualize genetic data in low dimensions.

## Installation

```bash
git clone https://github.com/felixpacheco/AncestryVAE
pip install -r AncestryVAE/requirements.txt
```

## Usage

The package can be used to compute latent spaces or to project new observations on a pre-computed VAE (encoder function)

python -m AncestryVAE/src/vae.py 

### Options :

  #### Mandatory :
  
  ``--name``  Name of output files

  ``--infile`` path to input vcf file or tensors dir

  ``--out`` path to save output
  
  #### Optional :
  
  ``--patience`` training patience (default=1000)

  ``--max_epochs`` maximum training epochs (default=50)

  ``--batch_size`` Batch size (default=20)

  ``--save_model`` Save model together with analysis of training and plots
  
  ``--cpu`` Use CPU instead of GPU (default=False)
  
  ``--seed`` Seed (deault=1)
  
  ``--train_prop`` Proportion of samples to use for training, (default=0.8)
  
  ``--grid_search`` Run grid search over network sizes and use the network with minimum test loss (default=False). 
  
  ``--depth`` Number of hidden layers (default=3).
  
  ``--width`` Nodes per hidden layer (default=1000).
  
  ``--plots`` generate an interactive scatterplot of the latent space (default=False, requires -metadata).

  ``--metadata`` Path to the tab-delimited metadata file with column "id" and "label".
  
  ``--extra_annot`` extra annotation at the end of the file
  
  ``--project_data`` Project data on pre-trained model (default=False, requires model_path)
  
  ``--model_path`` path to pre-trained model
  
Train VAE from scratch :
```
python -m  --name outputname —infile ... —-out .. -—max_epochs .. —batch_size ... —seed ... —plot
```

Project new observations on pre-computed VAE:
```
python -m  --name outputname —infile — out ... —max_epochs ...  —batch_size —seed ... —-project_data ... --model_path ...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
