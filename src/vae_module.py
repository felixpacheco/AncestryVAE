#!/usr/bin/env python
"""vae_module.py: Module containing the VAE class"""

from methods import loss_ignore_nans, impute_data

# Torch dependencies
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader

import numpy as np

class VAE(nn.Module):
    def __init__(self, input_features, input_batch, zdims,hidden_units, hidden_layers):
        super(VAE, self).__init__()
        
        # Input data
        self.input_features = input_features
        self.input_batch = input_batch
        self.zdims = zdims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.relu = nn.ReLU()
        
        # ENCODER : From input dimension to bottleneck (zdims)
        # Input layer
        self.fc1 = nn.Linear(2, 1)
        self.bn1 = nn.BatchNorm1d(num_features=input_features)

        #self.cnn1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        self.fc2 = nn.Linear(self.input_features, self.hidden_units)
        self.bn2 = nn.BatchNorm1d(num_features=self.hidden_units)
        
        # Hidden layers
        self.encode_hidden = nn.ModuleList()
        self.encode_bn = nn.ModuleList()
        for k in range(0,hidden_layers):
            self.encode_hidden.append(nn.Linear(self.hidden_units, self.hidden_units))
            self.encode_bn.append(nn.BatchNorm1d(num_features=self.hidden_units))

        # Bottleneck
        self.fc21 = nn.Linear(self.hidden_units, self.zdims)  # mu layer
        self.bn21 = nn.BatchNorm1d(num_features=self.zdims)
        self.fc22 = nn.Linear(self.hidden_units, self.zdims)  # logvariance layer
        self.bn22 = nn.BatchNorm1d(num_features=self.zdims)

        # DECODER : From bottleneck to input dimension
        # Latent to first hidden
        self.fc3 = nn.Linear(self.zdims, self.hidden_units)
        self.bn3 = nn.BatchNorm1d(num_features=self.hidden_units)
        # Hidden Layers
        self.decode_hidden = nn.ModuleList()
        self.decode_bn = nn.ModuleList()
        for k in range(0,hidden_layers):
            self.decode_hidden.append(nn.Linear(self.hidden_units, self.hidden_units))
            self.decode_bn.append(nn.BatchNorm1d(num_features=self.hidden_units))

        self.fc4 = nn.Linear(self.hidden_units, self.input_features)
        self.bn4 = nn.BatchNorm1d(num_features=self.input_features)
        #self.cnn1t = nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=1, stride=1)
        
        self.fc5 = nn.Linear(1, 2)

    def encode(self, x, impute=True):
        """Input vector x -> fully connected layer 1 -> ReLU -> (fc21, fc22)
        Parameters
        ----------
        x : [input_batch, input_features] matrix

        Returns
        -------
        mu     : zdims mean units one for each latent dimension (fc21(h1))
        logvar :  zdims variance units one for each latent dimension (fc22(h1))
        """
        # One-hot-encoded -> input_features

        h1 = self.bn1(self.fc1(x))
        h1 = torch.reshape(h1, (h1.shape[0], self.input_features))

        # Input features -> hidden_units
        h1 = self.relu(self.bn2(self.fc2(h1)))

        # Hidden_units -> hidden units
        for i in range(0,self.hidden_layers):
            h1 = self.relu(self.encode_bn[i](self.encode_hidden[i](h1)))

        return self.bn21(self.fc21(h1)), self.bn22(self.fc22(h1))

    def reparameterize(self, mu, logvar, inference=False):
        """Reparametrize to have variables instead of distribution functions
        Parameters
        ----------
        mu     : [input_batch, zdims] mean matrix
        logvar : [input_batch, zdims] variance matrix

        Returns
        -------
        During training random sample from the learned zdims-dimensional
        normal distribution; during inference its mean.
        """
        # Standard deviation
        std = torch.exp(0.5*logvar)
        # Noise term epsilon
        eps = torch.rand_like(std)
        
        if inference is True:
            return mu

        return mu+(eps*std)

    def decode(self, z):
        """z sample (20) -> fc3 -> ReLU (400) -> fc4 -> sigmoid -> reconstructed input
        Parameters
        ----------
        z : z vector

        Returns
        -------
        Reconstructed x'
        """
        # zdims -> hidden
        h2 = self.relu(self.bn3((self.fc3(z))))

        # Hidden -> hidden
        for i in range(0,self.hidden_layers):
            h2 = self.relu(self.decode_bn[i](self.decode_hidden[i](h2)))
        
        # Hidden -> input features
        h2 = self.relu(self.bn4(self.fc4(h2)))

        # One hot encoded
        h2 = torch.reshape(h2, (h2.shape[0], self.input_features, 1))
        h2 = self.fc5(h2)
        return h2

    def forward(self, x):
        """Connects encoding and decoding by doing a forward pass"""
        # Get mu and logvar
        mu, logvar = self.encode(x)
        # Get latent samples
        z = self.reparameterize(mu, logvar)
        # Reconstruct input
        return self.decode(z), mu, logvar

    def get_z(self, x):
        """Returns latent space of input x"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, inference=True)
        return z


def loss_function(recon_x, x, imputed_data, mu, logvar, beta, input_features, input_batch, zdims, device):
    """Computes the ELBO Loss (cross entropy + KLD)"""
    # KLD is Kullback–Leibler divergence
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * 1.5
    KLD /= (zdims) 

    # Compute loss between imputed x and reconstructed imputed x
    loss = nn.BCEWithLogitsLoss(reduction="none")
    BCE = loss(recon_x, imputed_data)

    # Compute BCE and ignore values that come from a nan
    BCE = loss_ignore_nans(device, BCE, x)
    BCE = torch.sum(BCE, dim=0)/float(BCE.shape[0])
    BCE = torch.sum(BCE)

    print(f"BCE : {BCE}, KLD : {KLD}")
    #return KLD, BCE, KLD#+ beta*KLD, BCE, KLD
    return BCE + beta*KLD, BCE, KLD

def get_rec_error(recon_x, x, imputed_data, batch_size, device):
    """Computes the ELBO Loss (cross entropy + KLD)"""
    # Compute loss between imputed x and reconstructed imputed x
    loss = nn.BCEWithLogitsLoss(reduction="none")
    BCE = loss(recon_x, imputed_data)

    # Compute BCE and ignore values that come from a nan
    BCE = loss_ignore_nans(device, BCE, x)
    BCE = torch.sum(BCE, dim=1)/float(BCE.shape[1]) # sum over snps
    BCE = torch.sum(BCE, dim=1)/float(BCE.shape[1]) # Sum over encoding vector
    return BCE # One reconstruction value per sample


def train(epoch, model, train_loader, CUDA, device, optimizer, scheduler, input_features, input_batch, train_classes, zdims):
    # toggle model to train mode
    model.train()
    train_loss = 0
    
    # Init save training
    train_loss_values = []
    train_bce = []
    train_kld = []
    lr_values = []
    similarity_values = []

    # Save latent space
    mu_train = np.empty([0,zdims])
    rec_train = np.empty([0])
    targets_train = np.empty([0],dtype=int)
    dtype=int

    # init beta param
    beta = 0.0

    # Iterate over train loader in batches of 20
    for batch_idx, (data, _) in enumerate(train_loader):
        
        if CUDA:
            data = data.to(device)
        
        optimizer.zero_grad()
        imputed_data = impute_data(tensor=data.cpu(), batch_size=input_batch)

        if torch.cuda.is_available():
            data = data.to(device)
            imputed_data = imputed_data.to(device)


        # Push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(imputed_data)
        # calculate loss function
        
        if epoch == 1 :
            beta = 0.0

        elif epoch == 2 :
            beta = (batch_idx * len(data)) / (len(train_loader.dataset)*2)

        elif epoch == 3 :
            beta = 0.5 + ((batch_idx * len(data)) / (len(train_loader.dataset)*2))

        else :
            beta = 1.0
            
        print(beta)
        loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, beta, input_features, input_batch, zdims, device)
        reconstruction_error = get_rec_error(recon_batch, data, imputed_data, input_batch, device)
        
        # calculate the gradient of the loss w.r.t. the graph leaves
        loss.backward()
        train_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        lr_value = optimizer.param_groups[0]['lr']

        print(f"Learning rate = {optimizer.param_groups[0]['lr']}")

        # Append values to then save them
        train_loss_values.append(loss.item())
        train_bce.append(bce.item())
        train_kld.append(kld.item())
        lr_values.append(lr_value)

        # Save latent space

        mu_ = mu.cpu().detach().numpy()
        #target = _.cpu().detach().numpy()
        target =  np.array(_)
        print(target)
        rec_error = reconstruction_error.cpu().detach().numpy()

        mu_train = np.append(mu_train, mu_, axis=0)
        targets_train = np.append(targets_train,target, axis=0)
        print(targets_train)
        rec_train = np.append(rec_train,rec_error, axis=0)

        print(f"Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]")

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss_values, train_bce, train_kld, lr_values, mu_train, targets_train, rec_train
           


def test(epoch, model, test_loader, CUDA, device, input_features, input_batch, test_classes, zdims):
    # toggle model to test / inference mode
    test_loss = 0
    model.eval()

    # Save test loss
    test_loss_values = []
    test_bce = []
    test_kld = []
    beta = 1.0

    # Save latent space
    mu_test = np.empty([0,zdims])
    rec_test = np.empty([0])
    targets_test = np.empty([0],dtype=int)
    

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            #Impute data
            imputed_data = impute_data(data.cpu(), batch_size=input_batch)

            if CUDA:
                data = data.to(device)
                imputed_data = imputed_data.to(device)

            # Push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = model(imputed_data)
            
            # calculate loss function
            test_loss, bce, kld = loss_function(recon_batch, data, imputed_data, mu, logvar, beta, input_features, input_batch, zdims, device)
            reconstruction_error = get_rec_error(recon_batch, data, imputed_data, input_batch, device)

            test_loss += test_loss.item()
            print(f"Test epoch: {epoch} [{i * len(data)}/{len(test_loader.dataset)}]")
            
            # Save test error 
            test_loss_values.append(test_loss.item())
            test_bce.append(bce.item())
            test_kld.append(kld.item())

            # Save latent space

            mu_ = mu.cpu().detach().numpy()
            #target = _.cpu().detach().numpy()
            target =  np.array(_)
            rec_error = reconstruction_error.cpu().detach().numpy()

            mu_test = np.append(mu_test, mu_, axis=0)
            targets_test = np.append(targets_test,target, axis=0)
            rec_test = np.append(rec_test,rec_error, axis=0)


        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return test_loss_values, test_bce, test_kld, mu_test, targets_test, rec_test