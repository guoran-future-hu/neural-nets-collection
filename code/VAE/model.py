import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0),  # 64x64x3 -> 31x31x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # 31x31x32 -> 14x14x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),  # 14x14x64 -> 6x6x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),  # 6x6x128 -> 2x2x256
            nn.ReLU(),
            nn.Flatten(),  # 2x2x256 = 1024
        )
        
        # Latent space
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2, padding=0),  # 1x1x1024 -> 5x5x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0),  # 5x5x128 -> 13x13x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0),  # 13x13x64 -> 30x30x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0),  # 30x30x32 -> 64x64x3
            nn.Sigmoid()
        )


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        x = self.encoder(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        
        z = self.reparameterize(mu, log_var)
        
        z = self.decoder_input(z)
        z = z.view(-1, 1024, 1, 1)
        out = self.decoder(z)
        
        return out, mu, log_var
    
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        # Normalize by batch size
        MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Apply beta for KL scaling
        return MSE + beta * KLD
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
