import argparse

def get_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type = str, default = 'densenet121', help = "CNN Model Architecture: default='densenet121' or 'vgg16'")
    parser.add_argument('--lr', type = float, default = .003, help = 'Learning rate for optimizer: default=.003')
    parser.add_argument('--hidden_layers', type = str, default = '512', help = 'Model hidden layers (comma separated list): default=512')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of training epochs: default=10')
    parser.add_argument('--gpu', type = str, default = 'True', help = 'Use GPU if available: default=True')
    parser.add_argument('--image_root', type = str, default = 'flowers', help = "Root directory of images folder: default='flowers'")
    parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size: default=16')
    parser.add_argument('--model_checkpoint', type = str, default = 'checkpoint.pth', help = "'Location to save checkpoint: default='checkpoint.pth'")

    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type = str, default = 'True', help = 'Use GPU if available: default=True')
    parser.add_argument('--image_path', type = str, default = './flowers/test/28/image_05214.jpg', help = "Path of file to predict: default='./flowers/test/28/image_05214.jpg'")
    parser.add_argument('--model_checkpoint', type = str, default = 'checkpoint.pth', help = "'Path to checkpoint file: default='checkpoint.pth'")
    parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes: default=5')

    return parser.parse_args()

# Ref https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')