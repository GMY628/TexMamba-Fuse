from utils.img_read_save import img_save, image_read_cv2
import matplotlib.cm as cm
from lavis.models import load_model_and_preprocess
import time
from torchvision.utils import save_image
import logging
import warnings
import torch.nn as nn
import torch
import numpy as np
from mambablock_open import Mamba_Encoder
from mambablock_open import Mamba_Decoder_textguide as Mamba_Decoder
import torch.utils
import os

os.environ['TRANSFORMERS_OFFLINE'] = '0'

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path = r"checkpoints\stage2.pth"
for dataset_name in ["lytro"]:
    print("The test result of "+dataset_name+' :')
    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('fusion_result_withtext', dataset_name)
    os.makedirs(test_out_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Encoder_common = nn.DataParallel(Mamba_Encoder()).to(device)
    Encoder_diff = nn.DataParallel(Mamba_Encoder(inp_channels=3,)).to(device)

    Mamba_decoder = nn.DataParallel(Mamba_Decoder()).to(device)

    Encoder_common.load_state_dict(
        torch.load(ckpt_path)['DIDF_Encoder_common'])
    Encoder_diff.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder_diff'])
    Mamba_decoder.load_state_dict(torch.load(ckpt_path)['Mamba_decoder'])

    Encoder_common.eval()
    Encoder_diff.eval()

    Mamba_decoder.eval()

    with torch.no_grad():
        times = 0
        for img_name in os.listdir(os.path.join(test_folder, "blur1")):

            imageA = image_read_cv2(os.path.join(
                test_folder, "blur1", img_name), mode='RGB').transpose(2, 0, 1)[np.newaxis, ...]/255.0
            imageB = image_read_cv2(os.path.join(
                test_folder, "blur2", img_name), mode='RGB').transpose(2, 0, 1)[np.newaxis, ...]/255.0
            name = img_name[:-4]
            imageA, imageB = torch.FloatTensor(
                imageA), torch.FloatTensor(imageB)
            imageA, imageB = imageA.cuda(), imageB.cuda()
            print(name)

            with open(os.path.join(test_folder, "text_nearunique", str(name)+'.txt'), 'r') as file:
                T_nearfocus = file.read()
                text_nearunique = txt_processors["eval"](T_nearfocus)
                sample1 = {"text_input": [text_nearunique]}
                with torch.no_grad():
                    text_nearunique = model.extract_features(
                        sample1, mode="text").text_embeds
            with open(os.path.join(test_folder, "text_farunique", str(name)+'.txt'), 'r') as file:
                T_farfocus = file.read()
                text_farunique = txt_processors["eval"](T_farfocus)
                sample2 = {"text_input": [text_farunique]}
                with torch.no_grad():
                    text_farunique = model.extract_features(
                        sample2, mode="text").text_embeds

            with open(os.path.join(test_folder, "text_near_far_common", str(name)+'.txt'), 'r') as file:
                T_near_far_common = file.read()
                text_near_far_common = txt_processors["eval"](
                    T_near_far_common)
                sample3 = {"text_input": [text_near_far_common]}
                with torch.no_grad():
                    text_near_far_common = model.extract_features(
                        sample3, mode="text").text_embeds

            feature_common = Encoder_common(torch.cat((imageA, imageB), dim=1))
            feature_image1 = Encoder_diff(imageA)
            feature_image2 = Encoder_diff(imageB)
            time1 = time.time()
            fuse = Mamba_decoder(feature_image1, feature_image2, feature_common,
                                 text_nearunique, text_farunique, text_near_far_common)
            time2 = time.time()
            time3 = time2 - time1

            times = times + time3

            save_image(fuse, test_out_folder+'/'+name+'.bmp')
