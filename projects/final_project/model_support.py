import torch
import time
import numpy as np

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from copy import copy


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
    """
        Loads test images and returns a dataset of images for training and validation
    Inputs:
        data_dir: Root directory of images folder
        batch_size: Number of images in batch returned by data loader
    Outputs:
        Dictionary with training data and validatoin data
    """
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
    """
    Loads test images and returns a dataset of images for testing
    Inputs:
        data_dir: Root directory of images folder
        batch_size: Number of images in batch returned by data loader
    Outputs:
        Test data loader
    """
    # Load the datasets with ImageFolder
    test_dir = data_dir + '/test'

    test_data = datasets.ImageFolder(test_dir, transform=validation_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    return torch.utils.data.DataLoader(test_data, batch_size=batch_size)


def train(model, optimizer, criterion, epochs, device, data):
    """
    Trains the model and tests against a validation set of data
    Inputs: 
        Model: A torchvision model
        Optimizer: Definition of optimizer function
        Criterion: Algorithm used to calculate loss
        Epochs: Number of training loops to execute
        Device: Device ['cuda', 'cpu'] on which to implement model
        Data: Dictionary of training images and validation images
    Outputs:
        None
    """
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

            print(f"Epoch: {e+1:0>2d}/{epochs:0>2d} |",
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
def save_model(args, model, optimizer):
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
    checkpoint = {'arch': args.arch,
                  'classifier': model.classifier,
                  'hidden_layers': args.hidden_layers,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'epochs': args.epochs}

    torch.save(checkpoint, args.model_checkpoint)


def freeze_model_features(model):
    """
    Freeze the features of input model
    Inputs: Model
    Outputs: None
    """
    # Freeze model features
    for param in model.features.parameters():
        param.requires_grad = False


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
    # Create by-value copy of hidden layers, since hidden_layers is passed by reference
    hl = copy(hidden_layers)
    
    # Load pre-trained model and replace classifier with custom classifier
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        # In features: 25088, Output 102 classes
        if hl == None:
            hl = [25088, 4096, 102] # Default configuration
        else:
            hl.insert(0, 25088)
            hl.append(102)
            
        model.classifier = nn.Sequential(OrderedDict(create_network(hl)))
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        
        # In features: 1024, Output 102 classes
        if hl == None:
            hl = [1024, 512, 102]  # Default configuration
        else:
            hl.insert(0, 1024)
            hl.append(102)
            
        model.classifier = nn.Sequential(OrderedDict(create_network(hl)))
    
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
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Pytorch tensor
    '''
    # Calculate aspect ratio of image
    width, height = image.size
    ar = height / float(width)
    
    # Calculate dimensions of scaled image
    if (height > width):
        resize_dims = (256, int(round(256 * ar)))
    else:
        resize_dims = (int(round(256 / ar)), 256)
    
    # Center of the "crop box"
    cc = 112
    
    # Calculate crop box (224 x 224)
    crop_box = (int(resize_dims[0] / 2 - cc), int(resize_dims[1] / 2 - cc),\
                int(resize_dims[0] / 2 + cc), int(resize_dims[1] / 2 + cc))

    # Resize to dimensions
    image.resize(resize_dims)
    
    # Crop to size of crop box and normalize to 0-1
    np_image = np.array(image.crop(crop_box)) / 255
    
    # Normalize color channels
    mean = np.array([.485, .456, .406])
    sd = np.array([.229, .224, .225])
    np_image = (np_image - mean) / sd
    
    # Transpose
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to FloatTensor type before returning
    return torch.from_numpy(np_image).type(torch.FloatTensor)


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Load image and process
    image = process_image(Image.open(image_path))
    
    # Send model to target device and put into eval mode
    model = model.to(device)
    model.eval()
    
    # Send image to target device modify dimensions
    # Ref: https://knowledge.udacity.com/questions/186897
    image.unsqueeze_(0)
    image = image.to(device)

    # Disable gradient calculation prior to running model
    with torch.no_grad(): 
        # Calculate class probabilities and determine "matches"
        class_p = torch.exp(model.forward(image))
   
    # Return model to training mode
    model.train()
    
    # Compute top n classes, return them to the cpu, convert to numpy array, and finally flatten
    probs, classes = class_p.topk(topk, dim=1) 
    probs, classes = probs.to('cpu'), classes.to('cpu')
    probs, classes = probs.numpy().flatten(), classes.numpy().flatten()
    
    # Ref: https://knowledge.udacity.com/questions/259422
    idx_to_class = {k:i for i, k in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in classes]

    return probs, classes
