import torch

from network.vae_cnn import VAE_CNN_P3D

reconstruction_function = torch.nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD


class VAE_Learner:
    def __init__(self, epochs=1000, lr=0.01):
        self.model = VAE_CNN_P3D()
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def learn(self, data):
        self.optimizer.zero_grad()
        mu, logvar, predictions = self.model(data)
        loss = loss_function(predictions, data, mu, logvar) / data.size(0)

        loss.backward()
        self.optimizer.step()
        print(' - loss : ', str(loss.item()))
        return mu

    def predict(self, data):
        mu, logvar, predictions = self.model(data)
        return mu
