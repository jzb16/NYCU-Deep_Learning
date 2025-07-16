import argparse
import torch
from train import train
from oxford_pet import load_dataset, transform_normalize
from torch.utils.data import DataLoader
from models.resnet34_unet import ResNetUNet
from models.unet import UNet
import warnings
import torch.nn as nn
from utils import dice_score

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# params
def get_args():
    parser = argparse.ArgumentParser(description =' Predict masks from input images')
    parser.add_argument('--data_path', '-d', type = str, help = 'path to the input data')
    parser.add_argument('--batch_size', '-b', type = int, default = 1, help = 'batch size')
    parser.add_argument('--learning_rate', '-l', type = float, default = 0.05, help = 'learning_rate')
    parser.add_argument('--epochs', '-e', type = int, default = 50, help = 'epochs')
    parser.add_argument('--model_type', '-t', type = str, required = True, help = 'Model type: UNet or ResNet34_UNet')
    parser.add_argument('--model_weight', '-m', default =' MODEL.pth ', help = 'path to the stored model weight')
    parser.add_argument('--in_channels', type = int, default = 3, help='Number of input channels')
    parser.add_argument('--out_channels', type =int, default = 1, help='Number of output channels')
    parser.add_argument('--weight_path', type = str, default = 1, help='path to the weights')

    return parser.parse_args()

# testing
def test(model, weight_path, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing...")
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()

    # Initialize variables to store evaluation metrics
    losses = []
    dice_scores = []

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Iterate
    with torch.no_grad():
        for batch in test_loader:
            inputs, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            losses.append(loss.item())

            # Compute Dice score
            pred_masks = torch.sigmoid(outputs) > 0.5
            batch_dice = dice_score(pred_masks.float(), masks.float())
            dice_scores.append(batch_dice.item())

    # Print and return results
    avg_loss = sum(losses) / len(losses)
    avg_dice = sum(dice_scores) / len(dice_scores)
    print(f"Average Loss: {avg_loss:.4f}, Average Dice: {avg_dice:.4f}")

def main():
    print("Start...")
    args = get_args()
    
    # Load datasets
    train_dataset = load_dataset(args.data_path, 'train', transform=transform_normalize)
    test_dataset = load_dataset(args.data_path, 'test', transform=transform_normalize)
    valid_dataset = load_dataset(args.data_path, 'valid', transform=transform_normalize)
    
    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=True)

    if args.model_type == 'UNet':
        model = UNet(in_channels=args.in_channels, out_channels=args.out_channels)
        model_name = 'UNet'
    elif args.model_type == 'ResNet34_UNet':
        model = ResNetUNet()
        model_name = 'ResNet34_UNet'

    # Train the model
    #train(model, model_name, train_loader, valid_loader, args.learning_rate, args.epochs)

    # Test the model
    test(model, args.weight_path, test_loader)

if __name__ == '__main__':
    main()


