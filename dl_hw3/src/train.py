import os
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate
from utils import dice_score, plot_dice, plot_loss

def train(model, model_name, train_loader, valid_loader, learning_rate, epochs):
    # Set device
    print("Training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ckpt_root =  os.path.join('saved_models', model_name)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Record accuracy and loss
    train_dice_list, train_loss_list = [], []
    valid_dice_list, valid_loss_list = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        best_valid_dice = 0.0

        for i, data in enumerate(train_loader):
            inputs, masks = data['image'].to(device), data['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            pred_masks = torch.sigmoid(outputs) > 0.5
            batch_dice = dice_score(pred_masks.float(), masks.float())
            epoch_dice += batch_dice.item()
        
        epoch_dice /= len(train_loader)
        train_dice_list.append(epoch_dice)
        epoch_loss /= len(train_loader)
        train_loss_list.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}")

        # evalute
        valid_dice_score, valid_dice_loss = evaluate(model, valid_loader)
        valid_dice_list.append(valid_dice_score)
        valid_loss_list.append(valid_dice_loss)

        if valid_dice_score > best_valid_dice:
            torch.save(model.state_dict(), 
                       os.path.join(ckpt_root, f'epoch{epoch}_{int(100*valid_dice_score)}.pth'))
            best_valid_dice = valid_dice_score

        # save data 
        with open(os.path.join(ckpt_root, f'{model_name}.txt'), 'a') as f:
            f.write(f'{epoch},{epoch_dice},{valid_dice_score}\n')
    
    plot_dice(epochs, train_dice_list, valid_dice_list, model_name)
    plot_loss(epochs, train_loss_list, valid_loss_list, model_name)

    print("Training finished.")
    print()
