import sys, os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from model.model import Unet
from data_loader import SaltDataset
from utils import rle_encode, rle_decode
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resume_path = 'saved/TGS_Unet_128/model_best.pth.tar'
output_path = 'saved/submission.csv'
threshold = 0.9

trfm = transforms.Compose([
    transforms.Pad((13, 13, 14, 14), padding_mode='reflect'),
    transforms.ToTensor(),
    ])
dataset = SaltDataset('input', transform=trfm, train=False)

# load trained weights
model = Unet(n_features=8)
print(f"Loading checkpoint: {resume_path} ...")
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

batch_size = 2048
batch = []
ids = []

rle_vec = np.vectorize(rle_encode)
n_data = len(dataset)
with open(output_path, 'wt') as f:
    f.write('id,rle_mask\n')
    with torch.no_grad():
        for i in tqdm(range(n_data)):
            data, fname = dataset[i]
            data = data.unsqueeze(0)
            batch.append(data)
            ids.append(fname)
            if len(batch) < batch_size and i != n_data - 1:
                continue
            else:
                output = model(torch.cat(batch, dim=0).to(device))
                output = output[:, 0, 13:-14, 13:-14] > threshold
                for mask, fname in zip(torch.unbind(output, dim=0), ids):
                    rle = rle_encode(mask.cpu().numpy().astype(np.bool))
                    f.write(f'{fname},{rle}\n')
                batch = []
                ids = []
    # process last batch
    # output = model(torch.cat(batch, dim=0).to(device))
    # output = output[:, 0, 13:-14, 13:-14] > threshold
    # for mask, fname in zip(torch.unbind(output, dim=0), ids):
    #     rle = rle_encode(mask.cpu().numpy().astype(np.bool))
    #     f.write(f'{fname},{rle}\n')

            # mask_recon = rle_decode(rle, (101, 101))
        # print(output[:, :, 13:-14, 13:-14].shape)
        # data = transforms.ToPILImage()(data[0, :, 13:-14, 13:-14].cpu())
        # mask = ((output[0, :, 13:-14, 13:-14] > 0).cpu())
        # data = (data[0, 0, 13:-14, 13:-14] > threshold).cpu().numpy()
        # print(mask.astype(np.uint8)[:, 0:6])
        # data = Image.fromarray(data*255, 'L')
        
        # mask_recon = Image.fromarray(mask_recon*255, 'L')
        # print(np.max(mask - mask_recon))
        # if i <= 10:
        #     data.save(f'sample/{i}_orig.png')
            # mask_recon.save(f'sample/{i}_out.png')
        #     # Image.fromarray(data.astype(np.uint8), 'RGB').save(f'sample/{i}_orig.png')
        #     # Image.fromarray(mask.astype(np.uint8), '1').save(f'sample/{i}_out.png')
        # else:
        #     break
    # print(f'{fname},{rle}\n')