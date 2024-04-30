from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2
import os
import numpy as np
from tqdm import tqdm


def rocauc_pixel(gt_all, am_all, f, tplt):
    f.write("pixel_rocauc:")
    f.write('\n')
    img_roc_aucs = []
    f1bests = []
    pbests = []
    rbests = []
    seg_thresholds = []
    for num in tqdm(range(len(am_all))):
        img_gt = gt_all[num]
        img_am = am_all[num]
        if img_gt.max() == 0:
            continue
        img_roc_auc = roc_auc_score(img_gt.flatten(), img_am.flatten())
        img_roc_aucs.append(img_roc_auc)
        precision, recall, thresholds = precision_recall_curve(img_gt.flatten(), img_am.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        f1best = f1.max()
        seg_threshold = thresholds[np.argmax(f1)]
        pbest = precision[np.argmax(f1)]
        rbest = recall[np.argmax(f1)]
        f1bests.append(f1best)
        pbests.append(pbest)
        rbests.append(rbest)
        seg_thresholds.append(seg_threshold)
        f.write(tplt.format(str(img_roc_auc), str(pbest), str(rbest), str(f1best), str(seg_threshold)))
        f.write('\n')
    return img_roc_aucs, f1bests, pbests, rbests, seg_thresholds


def rocauc_image(img_gts, img_scores, f, tplt):
    f.write('\n')
    f.write("image_rocauc:")
    f.write('\n')
    gts = np.asarray(img_gts)
    scores = np.asarray(img_scores)
    img_roc_auc = roc_auc_score(gts, scores)
    precision, recall, thresholds = precision_recall_curve(gts, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    f1best = f1.max()
    pbest = precision[np.argmax(f1)]
    rbest = recall[np.argmax(f1)]
    seg_threshold = thresholds[np.argmax(f1)]
    f.write(tplt.format(str(img_roc_auc), str(pbest), str(rbest), str(f1best), str(seg_threshold)))
    f.write('\n')
    return img_roc_auc, pbest, rbest, f1best


def rocauc(am_all, rocauc_name, pixel_gts, img_gts):
    img_scores = []
    pxiel_scores = []
    for num in range(len(am_all)):

        img_num = np.squeeze(pixel_gts[num]).astype(np.uint8)
        if img_num.shape[0] != 512:
            img_num = cv2.resize(img_num, (512, 512))
        pixel_gts[num] = img_num.astype(np.uint8)[162:-162, 64:-64]

        img_am = am_all[num]
        if img_am.shape[0] != img_num.shape[0]:
            img_am = cv2.resize(img_am, (img_num.shape[0], img_num.shape[1]))
        img_am = (1 - img_am[162:-162, 64:-64])

        score = img_am.max()
        if score - img_am.min() != 0:
            img_am = (img_am - img_am.min()) / (score - img_am.min())

        img_scores.append(score)
        pxiel_scores.append(img_am)
    f = open(rocauc_name, 'w')
    tplt = "{:<20}\t{:<20}\t{:<20}\t{:<20}\t{:<20}"
    f.write(tplt.format("img_roc_auc", "p", "r", "f1", "seg_threshold"))
    f.write('\n')
    pixel_rocaucs, f1bests, pbests, rbests, seg_thresholds = rocauc_pixel(pixel_gts, pxiel_scores, f, tplt)
    mean_rocauc = sum(pixel_rocaucs) / len(pixel_rocaucs)
    mean_f1 = sum(f1bests) / len(f1bests)
    mean_p = sum(pbests) / len(pbests)
    mean_r = sum(rbests) / len(rbests)
    mean_threshold = sum(seg_thresholds) / len(seg_thresholds)
    f.write("mean:")
    f.write('\n')
    f.write(tplt.format(str(mean_rocauc), str(mean_p), str(mean_r), str(mean_f1), str(mean_threshold)))
    f.write('\n')
    img_roc_auc, pbest, rbest, f1best = rocauc_image(img_gts, img_scores, f, tplt)
    f.close()
    return mean_rocauc, mean_p, mean_r, mean_f1, img_roc_auc, pbest, rbest, f1best
