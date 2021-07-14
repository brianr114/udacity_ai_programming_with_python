import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# Save the checkpoint 
def save_model(file_path, model, hidden_layers, class_to_idx, optimizer, epochs):
    """
    Saves the model for future retrieval
    Inputs: Filepath: Path and name to save file
            Model: Model to save
            Hidden Layers: List of size of each hidden layer
            Class_to_Index: The mapping of classes to index value.
            Optimizer: Current optimizer being used
            Epochs: number of epochs to run
            
    Outputs: Saves a model checkpoint file at the location specified
            
    """
    checkpoint = {'arch': 'densenet121',
                  'classifier': model.classifier,
                  'hidden_layers': hidden_layers,
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'epochs': epochs}

    torch.save(checkpoint, file_path)


    # TODO: Write a function that loads a checkpoint and rebuilds the model
def freeze_model_features(model):
    """
    Freeze the features of input model
    Inputs: Model
    Outputs: None
    """
    # Freeze model features
    for feature in model.features:
        feature.requires_grad = False


def create_network(hidden_layers):
    """
    Creates the classifier feed-forward network based on the parameters of hidden_layers
    Inputs: Array of numbers, each representing the size of each hidden layer.
            Expects the number of input features to be at the first element of the list
            and number of output categories to be the last element of the list
    Outputs:A list of network parameters, including size, dropout, and activation functions
    """
    network = []
    for i in range(len(hidden_layers) - 1):
        # Last iteration/output layer
        if i == (len(hidden_layers) - 2):
            network.extend([('fc_{}'.format(i), nn.Linear(hidden_layers[i],hidden_layers[i+1])),
                            ('out', nn.LogSoftmax(dim=1))])

        else:
            network.extend([('dropout_{}'.format(i), nn.Dropout(.2)), 
                            ('fc_{}'.format(i), nn.Linear(hidden_layers[i],hidden_layers[i+1])),
                            ('relu_{}'.format(i), nn.ReLU())])
    return network
       

def create_model(arch='densenet121', hidden_layers=None, lr=.003, class_to_idx=None):
    """
    Creates a model with specified architecture, number of hidden layers, learning rate,
    and class-to-index.
    Inputs: Model architecture: Currently supports 'densenet121'(default) and 'vgg16'
            List of hidden layers: Each element in the list represents the size of one
            hidden layer.
            Learning Rate: Learning rate of the optimizer.  Default is .003
            Class_to_Index: The mapping of classes to index value.
    Outputs: The fully configured model, optimizer, and criterion
    
    """
    # Load pre-trained model and replace classifier with custom classifier
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        # In features: 25088, Output 102 classes
        if hidden_layers == None:
            hidden_layers = [25088, 4096, 102] # Default configuration
        else:
            hidden_layers.insert(0, 25088)
            hidden_layers.append(102)
            
        model.classifier = nn.Sequential(OrderedDict(create_network(hidden_layers)))
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        
        # In features: 1024, Output 102 classes
        if hidden_layers == None:
            hidden_layers = [1024, 512, 102]  # Default configuration
        else:
            hidden_layers.insert(0, 1024)
            hidden_layers.append(102)
            
        model.classifier = nn.Sequential(OrderedDict(create_network(hidden_layers)))
    
    # Freeze model features
    freeze_model_features(model)
    
    # Create optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    #Set model class to idx values
    model.class_to_idx = class_to_idx
    
    # Define criterion
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion
    

def load_model(filepath, lr=.003):
    """
    Loads a previously saved model.
    Inputs: Filepath to the model parameters file
            Learning rate: default = .003
    Outputs: The fully configured model, optimizer, and criterion
    """
    # Load model checkpoint file
    checkpoint = torch.load(filepath)
    
    # Load model
    model, optimizer, criterion = create_model(checkpoint['arch'], checkpoint['hidden_layers'], lr)
    
    # Load model parameters from checkpoint file
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Load optimizer parameters
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    epochs = checkpoint['epochs']
    
    return model, optimizer, criterion