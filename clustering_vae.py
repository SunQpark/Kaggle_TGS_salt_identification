import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from data_loader import SaltDataset



class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(1, 4, 5) # 10, 24, 24
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 10, 3, padding=1) # 20, 10, 10
        self.bn2 = nn.BatchNorm2d(10)
        self.maxpool1 = nn.MaxPool2d(2, 2) #20, 12, 12

        self.conv3 = nn.Conv2d(10, 10, 3, padding=1) # 20, 4, 4
        self.bn3 = nn.BatchNorm2d(10)
        self.maxpool2 = nn.MaxPool2d(2, 2) # 20, 6, 6

        self.fc1 = nn.Linear(360, 100)
        self.fc_mu = nn.Linear(100, hidden_size)
        self.fc_var = nn.Linear(100, hidden_size)
        
    def forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        x = self.elu(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        x = self.elu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)
        x = x.view(-1, 360)
        x = F.tanh(F.dropout(self.fc1(x), 0.3))
        mu  = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar#, idx1, idx2

class Reparametrize(nn.Module):
    def __init__(self):
        super(Reparametrize, self).__init__()

    def forward(self, mu, logvar):
        logstd = 0.5 * logvar
        std = torch.exp_(logstd)
        if self.training:
            z = torch.randn_like(std, dtype=torch.float32) * std + mu
        else: 
            z = mu
        return z

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 360)
        
        self.upsample1 = nn.ConvTranspose2d(10, 10, 2, stride=2)
        self.conv1 = nn.Conv2d(10, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)

        self.upsample2 = nn.ConvTranspose2d(10, 10, 2, stride=2)
        self.conv2 = nn.Conv2d(10, 4, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(4)

        self.upsample3 = nn.ConvTranspose2d(4, 4, 5)
        self.conv3 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, z):
        z = F.elu(self.fc1(z), inplace=True)
        z = F.elu(self.fc2(z), inplace=True)
        x = z.view(-1, 10, 6, 6)

        x = self.upsample1(x)
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.upsample2(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.upsample3(x)
        x = self.conv3(x)
        output = F.sigmoid(x)
        return output
    

class VAE(nn.Module):
    def __init__(self, hidden_size=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparam(self, mu, logvar):
        logstd = 0.5 * logvar
        std = torch.exp_(logstd)
        if self.training:
            z = torch.randn_like(std) * std + mu
        else: 
            z = mu
        return z


def KL_divergence(mu, logvar):
    KLD = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = - (mu.pow(2) + logvar.exp()) + 1 + logvar
    KLD *= -0.5
    return torch.mean(KLD)

def reconst_loss(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, size_average=False) / x.shape[0]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tsfm = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor()
    ])
    dset = SaltDataset('./input', transform=tsfm, train=False)
    loader = DataLoader(dset, 1024, shuffle=True)
    
    model = VAE()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    
    writer = SummaryWriter(f"saved/runs/Clustering_VAE/{datetime.now().strftime('%m%d%H%M%S')}")
    train_iter = 0
    num_epochs = 64
    for epoch in range(num_epochs):
        print(f'epoch {epoch} / {num_epochs}: ')
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.mean(dim=1, keepdim=True)
            data = data.to(device)

            optimizer.zero_grad()
            output, mu, logvar = model(data)
            
            loss = KL_divergence(mu, logvar) + reconst_loss(output, data)
            loss.backward()
            optimizer.step()

            writer.add_scalar('VAE/loss', loss.item(), train_iter)
            writer.add_image('VAE/original', make_grid(data[:32], nrow=4), train_iter)
            writer.add_image('VAE/output', make_grid(output[:32], nrow=4), train_iter)
            train_iter += 1
        

    torch.save(model.state_dict, 'saved/VAE/model.pt')
    num_sample = 2048
    