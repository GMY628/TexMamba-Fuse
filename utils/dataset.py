import torch.utils.data as Data
import h5py
import numpy as np
import torch
import os
import torchvision.transforms as transforms

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                                         
from lavis.models import load_model_and_preprocess
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                                                                         
class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['A_patchs'].keys())
        h5f.close()
        self.transform = transforms.Compose([
             transforms.Resize((128, 128)), 
            ]) 
        self.transform2 = transforms.Compose([
             transforms.Resize((128, 128)), 
            ])        

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        imageA = np.array(h5f['A_patchs'][key])
        imageB = np.array(h5f['B_patchs'][key])
        mask = np.array(h5f['mask_patchs'][key])
        seg = np.array(h5f['seg_patchs'][key])
        h5f.close()
        imageA, imageB , mask, seg = torch.Tensor(imageA), torch.Tensor(imageB), torch.Tensor(mask), torch.Tensor(seg)

        if imageA.shape[1] > imageA.shape[2]:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
            mask = self.transform(mask)
            seg = self.transform(seg)
            
        else:
            imageA = self.transform2(imageA)
            imageB = self.transform2(imageB)
            mask = self.transform2(mask)
            seg = self.transform2(seg)
            
            

        return imageA, imageB, mask, seg                                                                                   
    
class H5Dataset_withtext(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['A_patchs'].keys())
        h5f.close()
        print("blip2")
        self.transform = transforms.Compose([
             transforms.Resize((128, 128)), 
            ])       

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        imageA = np.array(h5f['A_patchs'][key])
        imageB = np.array(h5f['B_patchs'][key])
        mask = np.array(h5f['mask_patchs'][key])
        seg = np.array(h5f['seg_patchs'][key])
     
        text_nearunique =  torch.from_numpy(np.array(h5f['text_nearunique_patchs'][key]))
        text_farunique =  torch.from_numpy(np.array(h5f['text_farunique_patchs'][key]))
        text_near_far_common =  torch.from_numpy(np.array(h5f['text_near_far_common_patchs'][key]))
        h5f.close()
        
        imageA, imageB , mask, seg  = torch.Tensor(imageA), torch.Tensor(imageB), torch.Tensor(mask), torch.Tensor(seg)
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)
        mask = self.transform(mask)
        seg = self.transform(seg)

    
        return imageA, imageB, mask, seg, text_nearunique, text_farunique, text_near_far_common

