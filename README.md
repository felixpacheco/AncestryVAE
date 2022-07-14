# AncestryVAE

AncestryVAE is a git repository to compute VAEs on genotype array data to visualize genetic data in low dimensions.

![alt text]([https://github.com/felixpacheco/AncestryVAE/docs/main/VAE_workflow.png])


## Installation

```bash
git clone https://github.com/felixpacheco/AncestryVAE
pip install -r AncestryVAE/requirements.txt
```
## Dependencies 
  requirements.txt (pip3 freeze)
  
## Usage

The package can be used to compute latent spaces or to project new observations on a pre-computed VAE (encoder function)

## Training from scratch

python -m AncestryVAE/src/vae.py train

### Options :

  #### Mandatory :
  
  ``--name``  Name of output files

  ``--infile`` path to input vcf file or tensors dir

  ``--out`` path to save output
  
  #### Optional :
  
  ``--patience`` training patience (default=2)

  ``--max_epochs`` maximum training epochs (default=50)

  ``--batch_size`` Batch size (default=20)
  
  ``--gpu`` Use GPU instead of CPU (default=False)
  
  ``--seed`` Seed (deault=1)
  
  ``--train_prop`` Proportion of samples to use for training, (default=0.8)
    
  ``--depth`` Number of hidden layers (default=3).
  
  ``--width`` Nodes per hidden layer (default=1000).
  
  ``--plot`` generate an interactive scatterplot of the latent space (default=False, requires --metadata).

  ``--metadata`` Path to the tab-delimited metadata file with column "id" and "ancestry".
  
Train VAE from scratch :
```
python vae.py train --name myVAE_results —infile dataset1 —-out /Desktop/analysis/VAE
```

## Projection of new data on pre-computed model

python AncestryVAE/src/vae.py project

### Options :

  #### Mandatory :
  
  ``--name``  Name of output files

  ``--infile`` path to input vcf file or tensors dir
   
  ``--model_path`` path to model

  ``--out`` path to save output
  
  
  #### Optional :
  
  ``--gpu`` Use GPU instead of CPU (default=False)
  
  ``--plot`` generate an interactive scatterplot of the latent space (default=False, requires --metadata).

  ``--metadata`` Path to the tab-delimited metadata file with column "id" and "ancestry".


Project new observations on pre-computed VAE:
```
python vae.py project   --name myVAE_projection —infile dataset2 —-out /Desktop/analysis/VAE --model_path /path/to/model.pt
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
