import cv2
from skimage.metrics import structural_similarity
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from util.run_test import run_main
from util.util import denormalize


def ssim_main(ori_path, save_path, reconst_path, device, model_mae, mask_image, mask_size, is_multiss=False):
    txt_name = os.path.join(save_path, "ssim.txt")
    diffs =[]
    oris = []
    f = open(txt_name, 'w')
    tplt = "{:<20}\t{:<20}\t{:<20}\t{:<20}"
    recs = sorted(os.listdir(reconst_path))
    for num in tqdm(range(len(recs))):
        rec_num = 1
        oripath = os.path.join(ori_path, recs[num])
        recpath = os.path.join(reconst_path, recs[num])
        ori = cv2.imread(oripath,0)
        rec = cv2.imread(recpath,0)
        if ori.shape[0] != rec.shape[0]:
            ori = cv2.resize(ori, rec.shape)
        (score, diff) = structural_similarity(ori, rec, full=True, gaussian_weights=True)
        diff = np.clip(diff, 0, diff.max())
        while score < 0.94 and is_multiss and rec_num < 2:
            rec_num += 1
            transform_image = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            rec = Image.open(recpath).convert('RGB')
            rec = transform_image(rec).to(device)
            rec = rec.unsqueeze(dim=0)
            rect, _, _ = run_main(rec, model_mae, mask_image, mask_size)
            cv2.imwrite(recpath, denormalize(rect[0].cpu().detach().numpy()))
            rec = cv2.imread(recpath, 0)
            if ori.shape[0] != rec.shape[0]:
                ori = cv2.resize(ori, rec.shape)
            (score, diff) = structural_similarity(ori, rec, full=True, gaussian_weights=True)
        f.write(tplt.format(recs[num], str(score), str(rec_num), str(diff.min())))
        f.write('\n')
        diffs.append(diff)
        oris.append(ori)
    f.close()
    return diffs



