import sys, os
import shutil
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchvision import transforms
from model.model import Unet, ResnetUnet, Generator
from data_loader import SaltDataset
from PIL import Image
from utils import rle_encode, rle_decode
from apply_crf import crf
from skimage.io import imread


def evaluate(model_path, dataset, device):
    # load trained weights
    model = Unet(n_fts=16)
    print(f"Loading checkpoint: {model_path} ...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    batch_size = 512
    batch = []
    ids = []

    n_data = len(dataset)
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
                output = output[:, 0, 13:-14, 13:-14].transpose(1, 2)
                for mask, fname in zip(torch.unbind(output, dim=0), ids):
                    temp_path = f'saved/temp/{fname}.npy'
                    if os.path.isfile(temp_path):
                        mask_cum = np.load(temp_path)
                        mask += torch.from_numpy(mask_cum).to(device)
                    
                    np.save(temp_path, mask)
                batch = []
                ids = []

if __name__ == '__main__':
    folds = [0]
    num_folds = len(folds)    
    output_path = 'saved/submission.csv'
    threshold = 0.8

    # delete temporary files 
    shutil.rmtree('saved/temp')
    os.mkdir('saved/temp')
    

    pad = (13, 13, 14, 14)
    trfm = transforms.Compose([
        transforms.Pad(pad, padding_mode='reflect'),
        transforms.ToTensor(),
        ])
    dataset = SaltDataset('input', transform=trfm, train=False)
    
    # Evaluate models trained on each fold of cross-validation
    # for fold_idx in range(num_folds):
    for fold_idx in folds:
        resume_path = f'saved/Unet_naive/model_best.pth.tar'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        evaluate(resume_path, dataset, device=device)
        # break

    test_path = 'input/test/images/'
    subm = pd.read_csv('input/sample_submission.csv')
    print('applying CRF to the mask')
    for i, fname in enumerate(tqdm(subm.id)):
        temp_path = f'saved/temp/{fname}.npy'
        # if not os.path.isfile(temp_path):    
        #     continue
        output = np.load(temp_path).T
        mask = (output / num_folds) > threshold
        orig_img = imread(f'{test_path}/{fname}.png')
        crf_output = crf(orig_img, mask)

        subm.loc[i,'rle_mask'] = rle_encode(crf_output.T)

    subm.to_csv(output_path, index=False)
    print(f'result saved at {output_path}')