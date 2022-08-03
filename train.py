import numpy as np
from numpy import linspace
from Loader.data_loader import DataLoader
import os
import click
from sklearn.model_selection import train_test_split
from util import ModelConfiguration
from models import SigmaNet,RobustModel,UnconstrainedModel
import sys
import torch
from tqdm import tqdm
from plotter import plotter
import pickle
from collections import OrderedDict
import json
from util import save_model_json

from IPython import embed

@click.group()
def cli():
    pass



@cli.command()
@click.option('-i','--input',required=True, help="Input Json file for training and validation.")
@click.option('-uc','--unconstraint',is_flag = True,default=False,show_default =True, help= "declare flag if Unconstraint model is desired")
@click.option('-t','--training-directory',required=True, help="Training directory")
@click.option('-m','--model-config',required=True, help="Path to model config .yml file")
@click.option('-e','--num-epochs',required=False,default = 10000, help="Number of epochs for training")
@click.option('-lr','--learning-rate',required=False,default = 1e-3, help="learning rate for training")
def setup( input:str,model_config:str,num_epochs:int,learning_rate:int,training_directory:str,unconstraint:bool)-> None:
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)

    #load features from .yml file
    mconfig = ModelConfiguration(model_config)
    features = mconfig.features

    #load data with features desired
    dataset,labels = DataLoader(input,model_config, features)

    #split data according to desired fraction
    x_train, x_val, y_train, y_val = train_test_split(dataset, labels,train_size = mconfig.train_size ,shuffle=mconfig.shuffle)
    EPOCHS =  num_epochs

    # reshape data
    xtrain = torch.tensor(x_train.values).float()
    ytrain = torch.tensor(y_train.values).reshape(-1,1).float()
    
    #setup model
    loss_func = torch.nn.L1Loss()
    if unconstraint:
        model = UnconstrainedModel()
    else:
        
        model = SigmaNet(RobustModel(), sigma=1,monotone_constraints=mconfig.monotone_constraints)

    optim_robust = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #run training
    loss = []
    pbar = tqdm(range(EPOCHS))
    for i in pbar:
        y_robust = model(xtrain)
        loss_robust = loss_func(y_robust, ytrain)
        loss_robust.backward()
        optim_robust.step()
        optim_robust.zero_grad()
        loss.append(loss_robust.item())
        pbar.set_description(
            f"epoch: {i} loss: {loss_robust.item():.4f}")

    #save model and model dict  for later use
    with open(os.path.join(training_directory,'objs.pkl'), 'wb') as f: 
        pickle.dump([EPOCHS, loss, y_train,y_robust,x_val,y_val], f)
    model_dict = model.state_dict()
    model_dict = OrderedDict({k: model_dict[k].detach().cpu().tolist() for k in model_dict})
    save_model_json(os.path.join(training_directory,"model.json"),model_dict)
    with open(os.path.join(training_directory,"model.json"), "w") as outfile:
        json.dump(model_dict, outfile)
    

    torch.save(model.state_dict(),os.path.join(training_directory,"model.pt"))






if __name__ == '__main__':
    cli()

