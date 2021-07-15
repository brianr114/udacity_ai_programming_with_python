import torch
import json
import helper_functions
import parse_args
import model_support

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def main():
    # Read command line arguments and parse/format as needed
    args = parse_args.get_predict_args()
    args.gpu = parse_args.string_to_bool(args.gpu)

    # Determine device for model processing
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    model, optimizer, criterion = model_support.load_model(args.model_checkpoint)

    # Run image against model
    probs, classes = helper_functions.predict(args.image_path, model, args.topk, device)

    # Get class names for most likely classes
    class_names = [cat_to_name[i] for i in classes]

    for i in range(args.topk):
        print(f"Probability: {probs[i]:.4f} | Classification: {class_names[i].title()}")

if __name__ == "__main__":
    main()