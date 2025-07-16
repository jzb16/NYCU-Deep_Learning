import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from DDPM import ConditionalDDPM
from dataloader import create_dataloader
from UNet import build_model
import os
from torchvision.utils import save_image, make_grid
from evaluator import evaluation_model
from util import plot_losses_ddpm, plot_scores


class DDPM_TrainTest():
    def __init__(self, args, ddpm):
        super(DDPM_TrainTest).__init__()

        self.args = args
        self.lr = args.lr
        self.n_epoch = args.num_epochs
        self.device = args.device
        self.noise_type = args.noise_type
        self.sample_method = args.sample_method
        self.ddpm = ddpm    
        self.evaluator = evaluation_model()

    def train_ddpm(self, train_loader, test_loader, new_test_loader):
        optimizer = optim.Adam(self.ddpm.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=3, min_lr=0)
        best_test_score, best_new_test_score = 0, 0

        Losses = []
        Test_Scores, New_Test_Scores = [], []

        for epoch in tqdm(range(args.num_epochs)):
            self.ddpm.train()
            optimizer.param_groups[0]['lr'] = args.lr * (1 - epoch / args.num_epochs)  # linear lr decay

            for images, labels in tqdm(train_loader):
                optimizer.zero_grad()
                images = images.to(args.device)
                labels = labels.to(args.device)
                loss = self.ddpm(images, labels)
                loss.backward()
                optimizer.step()

            """testing"""
            test_score, grid, grid_denoise = self.test_ddpm(test_loader)
            path = os.path.join(args.test_result_path, self.sample_method, self.noise_type, f"test_{epoch}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(grid, path)

            save_path = os.path.join(self.args.denoise_path, self.noise_type, f"denoise_process_{epoch}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(grid_denoise, save_path)

            """new testing"""
            new_test_score, grid, _ = self.test_ddpm(new_test_loader)
            path = os.path.join(args.new_test_result_path, self.sample_method, self.noise_type, f"test_{epoch}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(grid, path)

            if test_score > best_test_score or new_test_score > best_new_test_score:
                if test_score > best_test_score:
                    best_test_score = test_score
                else:
                    best_new_test_score = new_test_score
                    
                path = os.path.join(args.model_path, self.sample_method, self.noise_type, 
                                    f"ddpm_{epoch}_t{test_score:.3f}_nt{new_test_score:.3f}.pth")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.ddpm.state_dict(), path)
                print("saved best test model")

            lr_scheduler.step(test_score)

            Losses.append(loss.item())
            Test_Scores.append(test_score)
            New_Test_Scores.append(new_test_score)

            print(f'Epoch: {epoch} train loss: {loss.item()} test score: {test_score} new_test_score: {new_test_score}')
            print()

        return Losses, Test_Scores, New_Test_Scores


    def test_ddpm(self, test_loader):
        self.ddpm.eval()
        x_gen, label, sample = [], [], []
        with torch.no_grad():
            for i, cond in enumerate(tqdm(test_loader)):
                cond = cond.to(self.device)
                size = (3, 64, 64)
                generated_images, samples = self.ddpm.sample(cond, size, self.device)

                x_gen.append(generated_images)
                sample.append(samples)
                label.append(cond)

            sample = torch.stack(samples, dim=0).squeeze() 
            x_gen = torch.stack(x_gen, dim=0).squeeze()#.view(-1, 3, 64, 64) # [32, 3, 64, 64]
            label = torch.stack(label, dim=0).squeeze()#.view(-1, 24) # [32, 24]

            score = self.evaluator.eval(x_gen, label)

            grid = make_grid(x_gen, nrow=8, normalize=True)
            grid_denoise = make_grid(sample, nrow=10, normalize=True)

        return score, grid, grid_denoise


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # label_dim = 24
    
    if args.train:
        train_loader, n_classes = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='train')
        test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='test')
        new_test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='new_test')
        train_model = build_model(args, n_classes, device, mode="train")
        ddpm = ConditionalDDPM(unet_model=train_model, betas=(args.beta_start, args.beta_end), noise_steps=args.noise_steps, 
                               device=device, noise_type=args.noise_type).to(device)
        train_test = DDPM_TrainTest(args, ddpm)
        Losses, Test_Scores, New_Test_Scores = train_test.train_ddpm(train_loader, test_loader, new_test_loader)

        plot_losses_ddpm('ddpm', Losses)
        plot_scores('ddpm', Test_Scores, New_Test_Scores)

    if args.test:
        test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='test')
        new_test_loader, _ = create_dataloader(args.dataset_path, args.batch_size, args.num_workers, mode='new_test') 
        ddpm = ConditionalDDPM(unet_model=train_model, betas=(args.beta_start, args.beta_end), noise_steps=args.noise_steps, 
                               device=device, noise_type=args.noise_type).to(device)
        train_test = DDPM_TrainTest(args, ddpm)
        
        path_test = os.path.join(args.model_path, 'ddpm', args.noise_type, args.weight_path)
        ddpm.load_state_dict(torch.load(path_test))
        score, grid, grid_denoise = train_test.test_ddpm(test_loader)
        path = os.path.join(args.test_result_path, "test_load_weight.png")
        save_image(grid, path)
        print(f'Test score: {score}')

        path_new_test = os.path.join(args.model_path, 'ddpm', args.noise_type, args.weight_path)
        ddpm.load_state_dict(torch.load(path_new_test))
        score, grid, grid_denoise = train_test.test_ddpm(new_test_loader)
        path = os.path.join(args.new_test_result_path, "new_test_load_weight.png")
        save_image(grid, path)
        print(f'New Test score: {score}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test GAN or DDPM models")
    parser.add_argument('--device', type=str, default='cuda', help="GPU")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--test', action='store_true', help="Test the model")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batch_size", default = 32, type=int, help="batch size")
    parser.add_argument("--num_epochs", default = 100, type=int, help="training epochs")
    parser.add_argument('--num_workers', default = 2 , type=int, help='workers of Dataloader')
    parser.add_argument("--in_channels", default = 3, type=int, help="channels of input images")
    parser.add_argument('--n_feature', default = 256, type=int, help='condition embedding')
    # Diffusion
    parser.add_argument('--beta_start', default=1e-4, type=float, help='start beta value')
    parser.add_argument('--beta_end', default=0.02, type=float, help='end beta value')
    parser.add_argument('--noise_steps', default=1000, type=int, help='frequency of sampling')
    parser.add_argument('--noise_type', type=str, default='linear', choices=['linear', 'cosine'], help="Type of noise schedule to use")
    parser.add_argument('--sample_method', type=str, default='ddpm', help="Sampling method to use during generation")
    # path
    parser.add_argument('--dataset_path', type=str, default='iclevr', help="Model root")
    parser.add_argument("--model_path", default="model/", type=str, help="model ckpt path")
    parser.add_argument("--test_result_path", default="test_result/", type=str, help="save img path")
    parser.add_argument("--new_test_result_path", default="new_result/", type=str, help="save img path")
    parser.add_argument("--denoise_path", default="denoise/", type=str, help="save denoise img path")
    parser.add_argument("--weight_path", default="ddpm_58_t0.583_nt0.702.pth", type=str, help="weight path")

    args = parser.parse_args()
    main(args)