

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

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
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
