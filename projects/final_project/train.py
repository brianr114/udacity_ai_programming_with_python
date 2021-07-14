import torch
import argparse
import helper_functions
import parse_args



def main():
    # Read command line arguments and parse/format as needed
    args = parse_args.get_training_args()
    args.gpu = parse_args.str2bool(args.gpu)
    args.hidden_layers = [int(i) for i in args.hidden_layers.split(',')]
 
    # Determine device for model processing
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # Create model
    #model, optimizer, criterion = create_model(args.arch, args.hidden_layers, args.lr, train_data.class_to_idx)
    
   
if __name__ == "__main__":
    main()