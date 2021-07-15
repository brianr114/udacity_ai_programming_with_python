import torch
import time
import numpy as np

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict



# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406],
                                                            [.229, .224, .225])])

validation_test_transforms = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([.485, .456, .406],
                                                                        [.229, .224, .225])])

def load_train_data(data_dir, batch_size):
    # Load the datasets with ImageFolder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validation_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    data = {}
    data['train'] = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    data['validate'] = torch.utils.data.DataLoader(validate_data, batch_size=batch_size)
    return data, train_data.class_to_idx

def load_test_data(data_dir, batch_size):
    # Load the datasets with ImageFolder
    test_dir = data_dir + '/test'

    test_data = datasets.ImageFolder(test_dir, transform=validation_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    return torch.utils.data.DataLoader(test_data, batch_size=batch_size)

def train(model, optimizer, criterion, epochs, device, data):
    # Put model in training mode
    model.train()

    # Send model to defined device
    model.to(device)
    for e in range(epochs):
        start_time = time.time()
        train_loss = 0
            
        for images, labels in data['train']:
        
            # Send training to device
            images, labels = images.to(device), labels.to(device)
        
            # Initialize optimizer
            optimizer.zero_grad()
        
            # Send training data to model, compute loss, and calcuate gradients
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Perform validation every n steps and print results
        else:
            validation_loss = 0
            accuracy = 0

            # Put model in eval mode and disable gradient calculation to run faster
            model.eval()
            with torch.no_grad():
                for images, labels in data['validate']:

                    # Send data to device
                    images, labels = images.to(device), labels.to(device)

                    # Run model and calculate loss
                    log_ps = model.forward(images)
                    validation_loss += criterion(log_ps, labels).item()

                    # Calculate highest probability class and determine if there is a match
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {e+1:0>2d}/{epochs:0>2d} |",
                  f"Training Loss: {train_loss/len(data['train']):.3f} |",
                  f"Validation Loss: {validation_loss/len(data['validate']):.3f} |",
                  f"Accuracy: {accuracy / len(data['validate']):.3f} |",
                  f"Cycle Time: {int(round((time.time() - start_time) / 60)):0>3d} minutes")

            # Put model back into training mode
            model.train()

            # If accuracy exceeds threshold, save model and exit training
            if (accuracy / len(data['validate'])) > .8:
                break;

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