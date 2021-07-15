import torch
import helper_functions
import parse_args
import model_support

def main():
    # Read command line arguments and parse/format as needed
    args = parse_args.get_training_args()
    args.gpu = parse_args.string_to_bool(args.gpu)
    args.hidden_layers = [int(i) for i in args.hidden_layers.split(',')]
    
     # Determine device for model processing
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # Load training data
    data, class_to_idx = model_support.load_train_data(args.image_root, args.batch_size)
    
    # Create model
    model, optimizer, criterion = model_support.create_model(args.arch, args.hidden_layers, args.lr, class_to_idx)

    # Train the model
    model_support.train(model, optimizer, criterion, args.epochs, device, data)

    # Save the model
    model_support.save_model(args.model_checkpoint, model, args.hidden_layers, class_to_idx, optimizer, args.epochs)
      
   
if __name__ == "__main__":
    main()