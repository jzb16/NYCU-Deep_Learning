import argparse
import os
import torch.optim as optim
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from dataloader import create_dataloader
from GAN import GAN_Generator, GAN_Discriminator
from evaluator import evaluation_model
from util import plot_losses, plot_scores


#import torchsummary

class GAN_TrainTest():
    def __init__(self, args, generator, discriminator):
        super(GAN_TrainTest).__init__()

        self.args = args
        self.n_epoch = args.num_epochs
        self.device = args.device
        self.generator = generator   
        self.discriminator = discriminator
        self.evaluator = evaluation_model()
        self.dim_z = args.dim_z
        self.dim_c = args.dim_c
    
    
    def train_gan(self, train_loader, test_loader, new_test_loader):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=args.lr_generator, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=args.lr_discriminator, betas=(0.5, 0.999))

        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='max', factor=0.95, patience=5, min_lr=0)
        lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='max', factor=0.95, patience=5, min_lr=0)

        criterion = nn.BCEWithLogitsLoss()
        best_test_score, best_new_test_score = 0, 0

        Generator_Losses, Discriminator_Losses = [], []
        Test_Scores, New_Test_Scores = [], []

        for epoch in tqdm(range(args.num_epochs)):
            generator_loss, discriminator_loss = 0, 0

            for images, labels in tqdm(train_loader):
                self.generator.train()
                self.discriminator.train()

                images = images.to(args.device)
                labels = labels.to(args.device)
                batch_size = images.size(0)

                # Train Discriminator
                noise = torch.randn(batch_size, args.dim_z, 1, 1, device=args.device)
                fake_images = self.generator(noise, labels)

                real_outputs = self.discriminator(images.detach(), labels)
                fake_outputs = self.discriminator(fake_images.detach(), labels)

                real_label = torch.ones_like(real_outputs, dtype=torch.float)
                fake_label = torch.zeros_like(fake_outputs, dtype=torch.float)

                loss_real = criterion(real_outputs, real_label)
                loss_fake = criterion(fake_outputs, fake_label)
                loss_d = loss_real + loss_fake

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

                # ---------------------------------------------------------
                # Train Generator for 3 times
                for i in range(3):
                    noise = torch.randn(batch_size, args.dim_z, 1, 1, device=args.device)
                    fake_image_test = self.generator(noise, labels)

                    outputs = self.discriminator(fake_image_test, labels)
                    generator_label = torch.ones_like(outputs, dtype=torch.float)

                    loss_g = criterion(outputs, generator_label)

                    optimizer_g.zero_grad()
                    loss_g.backward()
                    optimizer_g.step()

                discriminator_loss += loss_d.item()
                generator_loss += loss_g.item()    
                

            """testing"""
            test_score, grid = self.test_gan(test_loader)
            path = os.path.join(args.test_result_path, f"test_{epoch}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(grid, path)

            new_test_score, grid = self.test_gan(new_test_loader)
            path = os.path.join(args.new_test_result_path, f"test_{epoch}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(grid, path)

            if test_score > best_test_score or new_test_score > best_new_test_score:
                if test_score > best_test_score:
                    best_test_score = test_score
                else:
                    best_new_test_score = new_test_score
                
                generator_path = os.path.join(args.model_path, 'g', f"epoch{epoch}_t{test_score:.3f}_nt{new_test_score:.3f}.pth")
                os.makedirs(os.path.dirname(generator_path), exist_ok=True)
                torch.save(self.generator.state_dict(), generator_path)
                
                discriminator_path = os.path.join(args.model_path, 'd', f"epoch{epoch}_t{test_score:.3f}_nt{new_test_score:.3f}.pth")
                os.makedirs(os.path.dirname(discriminator_path), exist_ok=True)
                torch.save(self.discriminator.state_dict(), discriminator_path)
                
                print("Saved best test model")
            

            lr_scheduler_d.step(test_score)
            lr_scheduler_g.step(test_score)
            generator_loss /= len(train_loader)
            discriminator_loss /= len(train_loader)

            Generator_Losses.append(generator_loss)
            Discriminator_Losses.append(discriminator_loss)
            Test_Scores.append(test_score)
            New_Test_Scores.append(new_test_score)

            print(f'Epoch: {epoch} train generator loss: {generator_loss} train discriminator loss: {discriminator_loss} test score: {test_score} new_test_score: {new_test_score}')
            print()
        
        return Generator_Losses, Discriminator_Losses, Test_Scores, New_Test_Scores


    def test_gan(self, test_loader):
        self.generator.eval()
        self.discriminator.eval()
        x_gen = []
        scores = 0
        with torch.no_grad():
            for labels in tqdm(test_loader):
                labels = labels.to(args.device)
                batch_size = labels.size(0)
                noise = torch.randn(batch_size, args.dim_z, 1, 1, device=args.device)
                fake_image = self.generator(noise, labels)
                x_gen.append(fake_image)
                score = self.evaluator.eval(fake_image, labels)
                scores += score

            x_gen = torch.stack(x_gen, dim=0).squeeze()#.view(-1, 3, 64, 64) # [32, 3, 64, 64]
            grid = make_grid(x_gen, nrow=8, normalize=True)

        return scores, grid


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # label_dim = 24
    
    if args.train:
        train_loader, n_classes = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='train')
        test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='test')
        new_test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='new_test')

        generator = GAN_Generator(args.dim_z, args.dim_c).to(device)
        discriminator = GAN_Discriminator(num_classes=n_classes, image_size=64).to(device)
        generator.initialize_weights()
        discriminator.initialize_weights()
        train_test = GAN_TrainTest(args, generator, discriminator)
        Generator_Losses, Discriminator_Losses, Test_Scores, New_Test_Scores = train_test.train_gan(train_loader, test_loader, new_test_loader)

        # Plot Training and Test
        plot_losses('gan', Generator_Losses, Discriminator_Losses)
        plot_scores('gan', Test_Scores, New_Test_Scores)

    if args.test:
        test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='test')
        new_test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='new_test')
        generator = GAN_Generator(args.dim_z, args.dim_c).to(device)
        discriminator = GAN_Discriminator(num_classes=n_classes, image_size=64).to(device)
        generator.initialize_weights()
        discriminator.initialize_weights()

        g_path_test = os.path.join(args.model_path, 'gan', 'g', args.weight_path)
        d_path_test = os.path.join(args.model_path, 'gan', 'd', args.weight_path)
        generator.load_state_dict(torch.load(g_path_test))
        discriminator.load_state_dict(torch.load(d_path_test))
        score, grid, grid_denoise = train_test.test_gan(test_loader)
        path = os.path.join(args.test_result_path, "test_load_weight.png")
        save_image(grid, path)
        print(f'Test score: {score}')

        g_path_new_test = os.path.join(args.model_path, 'gan', 'g', args.weight_path)
        d_path_new_test = os.path.join(args.model_path, 'gan', 'd', args.weight_path)
        generator.load_state_dict(torch.load(g_path_new_test))
        discriminator.load_state_dict(torch.load(d_path_new_test))
        score, grid, grid_denoise = train_test.test_gan(new_test_loader)
        path = os.path.join(args.new_test_result_path, "new_test_load_weight.png")
        save_image(grid, path)
        print(f'New Test score: {score}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GAN or DDPM models")
    parser.add_argument('--device', type=str, default='cuda', help="GPU")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--test', action='store_true', help="Test the model")
    parser.add_argument("--batch_size", default = 32, type=int, help="batch size")
    parser.add_argument("--image_size", default = 64, type=int, help="image size")
    parser.add_argument("--num_classes", default = 24, type=int, help="number of classes")
    parser.add_argument("--num_epochs", default = 300, type=int, help="training epochs")
    parser.add_argument('--num_workers', default = 2 , type=int, help='workers of Dataloader')
    # GAN
    parser.add_argument('--lr_discriminator', default=0.0002, type=float, help='Learning rate of discriminator')
    parser.add_argument('--lr_generator', default=0.0002, type=float, help='Learning rate of generator')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--warm_up', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--dim_z', default=100, type=int, help='dimensionality of the latent space')
    parser.add_argument('--dim_c', default=200, type=int, help='number of condition embedding dim')
    # path
    parser.add_argument('--dataset_path', type=str, default='iclevr', help="Model root")
    parser.add_argument("--model_path", default="model/gan/", type=str, help="model ckpt path")
    parser.add_argument("--test_result_path", default="test_result/gan/", type=str, help="save img path")
    parser.add_argument("--new_test_result_path", default="new_result/gan/", type=str, help="save img path")
    parser.add_argument("--weight_path", default="ddpm_58_t0.583_nt0.702.pth", type=str, help="weight path")

    args = parser.parse_args()
    main(args)







