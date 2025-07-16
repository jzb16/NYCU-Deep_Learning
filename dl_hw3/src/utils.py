import torch
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask):
    # Compute intersection and union
    intersection = torch.sum(pred_mask * gt_mask)
    pred_sum = torch.sum(pred_mask)
    gt_sum = torch.sum(gt_mask)
    union = pred_sum + gt_sum

    # Compute Dice score
    dice = (2.0 * intersection) / union

    return dice

# Plot dice score
def plot_dice(epochs, train_dice_list, valid_dice_list, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_dice_list, label='Train Dice Score')
    plt.plot(range(1, epochs + 1), valid_dice_list, label='Valid Dice Score', linestyle='--')
    plt.title(f'{model_name} Dice Score: Training vs Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_dice.png')

    plt.show()

# Plot Loss
def plot_loss(epochs, train_loss_list, valid_loss_list, model_name):
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, epochs + 1), valid_loss_list, label='Valid Loss', linestyle='--')
    plt.title(f'{model_name} Loss: Training vs Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss.png')

    plt.show()