import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os


def blend(ori_path, anomaly_maps, savename, boxes):
    imagelist = sorted(os.listdir(ori_path))
    for num in tqdm(range(len(anomaly_maps))):
        savepath0 = os.path.join(savename, ("vis_" + imagelist[num]))
        savepath1 = os.path.join(savename, ("crop_" + imagelist[num]))
        savepath2 = os.path.join(savename, ("ano_" + imagelist[num]))
        a_m = (anomaly_maps[num] * 255).astype(np.uint8)
        if len(boxes) == 0:
            if a_m.shape[1] != 512:
                a_m = cv2.resize(a_m, (512, 512))
            anomaly_map = a_m
            cv2.imwrite(savepath2, anomaly_map)
            color_jet = cv2.applyColorMap((255 - anomaly_map), cv2.COLORMAP_JET)
            cv2.imwrite(savepath0, color_jet)
            src = cv2.imread(os.path.join(ori_path, imagelist[num]))
            if src.shape[1] != 512:
                src = cv2.resize(src, (512, 512))
            img = cv2.addWeighted(color_jet, 0.6, src, 0.4, 0)
        else:
            anomaly_map = a_m
            cv2.imwrite(savepath2, anomaly_map)
            color_jet = cv2.applyColorMap((255 - anomaly_map), cv2.COLORMAP_JET)
            cv2.imwrite(savepath0, color_jet)
            src = cv2.imread(os.path.join(ori_path, imagelist[num]))
            use_crop_am = color_jet[boxes[num][2]:boxes[num][3]:, boxes[num][0]:boxes[num][1]:]
            img = np.ones_like(color_jet)
            img[:,:,0] = img[:,:,0] * color_jet[0,0,0]
            img[:,:,1] = img[:,:,1] * color_jet[0,0,1]
            img[:,:,2] = img[:,:,2] * color_jet[0,0,2]
            img[boxes[num][2]:boxes[num][3]:, boxes[num][0]:boxes[num][1]:] = use_crop_am
            img = cv2.addWeighted(img, 0.5, src, 0.5, 0)
        cv2.imwrite(savepath1, img)
    return


def denormalize(images):
    mymean = np.array([0.485, 0.456, 0.406])
    mystd = np.array([0.229, 0.224, 0.225])
    images = ((images * mystd + mymean) * 255).astype(np.float32)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    return images