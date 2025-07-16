import matplotlib.pyplot as plt

def plot_losses(model, generator_losses, discriminator_losses):
    plt.figure(figsize=(14, 7))
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./result_image/{model}_training_losses.png')
    plt.close()
 

def plot_scores(model, test_scores, new_test_scores):
    plt.figure(figsize=(14, 7))
    plt.plot(test_scores, label="Test Scores")
    plt.plot(new_test_scores, label="New Test Scores")
    plt.title("Test and New Test Scores")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./result_image/{model}_test_scores.png')
    plt.close()

def plot_losses_ddpm(model, losses):
    plt.figure(figsize=(14, 7))
    plt.plot(losses, label="DDPM Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./result_image/{model}_training_losses.png')
    plt.close()
