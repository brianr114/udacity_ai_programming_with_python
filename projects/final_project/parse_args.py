import argparse

def get_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type = str, default = 'densenet121', help = "CNN Model Architecture: default='densenet121' or 'vgg16'")
    parser.add_argument('--lr', type = float, default = .003, help = 'Learning rate for optimizer: default=.003')
    parser.add_argument('--hidden_layers', type = str, default = '512', help = 'Model hidden layers (comma separated list): default=512')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of training epochs: default=10')
    parser.add_argument('--gpu', type = str, default = 'True', help = 'Use GPU if available: default=True')

    return parser.parse_args()

# Ref https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')