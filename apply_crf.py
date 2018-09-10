# code ref: https://www.kaggle.com/meaninglesslives/apply-crf
import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from tqdm import tqdm
from utils import rle_decode, rle_encode


"""
Function which returns the labelled image after applying CRF

"""
#Original_image = Image which has to labelled
#Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


if __name__ == '__main__':

    df = pd.read_csv('saved/submission.csv')
    test_path = 'input/test/images/'

    """
    Applying CRF on the predicted mask 

    """
    for i in tqdm(range(df.shape[0])):
        if str(df.loc[i,'rle_mask'])!=str(np.nan):        
            decoded_mask = rle_decode(df.loc[i,'rle_mask'], shape=(101, 101))
            # print(decoded_mask.T)
            orig_img = imread(test_path+df.loc[i,'id']+'.png')
            crf_output = crf(orig_img,decoded_mask)
            # assert(False)
            df.loc[i,'rle_mask'] = rle_encode(crf_output.T)

    df.to_csv('saved/submission_CRF.csv', index=False)