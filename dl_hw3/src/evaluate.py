import torch
import torch.nn as nn
from utils import dice_score

def evaluate(model, valid_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Initialize variables to store evaluation metrics
    total_loss = 0.0
    total_dice = 0.0
    num_samples = len(valid_loader)

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # Iterate
    with torch.no_grad():
        for batch in valid_loader:
            # Move inputs and masks to device
            inputs, masks = batch['image'].to(device), batch['mask'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Compute Dice score
            pred_masks = torch.sigmoid(outputs) > 0.5
            batch_dice = dice_score(pred_masks.float(), masks.float())
            total_dice += batch_dice.item()

    # Calculate average loss and Dice score
    avg_loss = total_loss / num_samples
    avg_dice = total_dice / num_samples

    print(f"Valid dice: {avg_dice}")

    return avg_dice, avg_loss
