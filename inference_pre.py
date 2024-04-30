import torch
import cv2
import argparse
import time
import os
from tqdm import tqdm
import models_mae
from util.dataset import MyDataset
from util.run_test import run_main
from util.ssim import ssim_main
from util.util import blend, denormalize
from util.rocauc import rocauc
from util.ss_get_ret import get_mask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--mask_size', default=16, type=int)
    parser.add_argument('--class_name', default='insulator', type=str)
    parser.add_argument('--test_dir', default='./data', type=str)
    parser.add_argument('--model_dir', default='./checkpoints', type=str)
    parser.add_argument('--save_dir', default='./results_pre', type=str)
    parser.add_argument('--vitname', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--is_multiss', action='store_true')
    parser.add_argument('--test_fold', default='test_real', type=str)
    return parser.parse_args()


def main(args):
    start_time = time.perf_counter()
    print("Test start")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    modelname = str(args.input_size) + "-20.pth"
    print(device, "\t", modelname)
    modelpath = os.path.join(args.model_dir, "insulator", modelname)
    if args.is_multiss:
        save_dir = args.save_dir + "/" + args.test_fold + "-mss"
    else:
        save_dir = args.save_dir + "/" + args.test_fold
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, args.class_name)
    if not os.path.isdir(save_file):
        os.mkdir(save_file)
    save_path = os.path.join(save_file, (modelname[:-4] + "-" + str(args.mask_size)))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # build model
    model = getattr(models_mae, args.vitname)(img_size=args.input_size, run_mode="inference")
    # load model
    checkpoint = torch.load(modelpath, map_location='cuda')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model_mae = model.to(device)
    # load dataset
    test_dataset = MyDataset(args.test_dir, test_fold=args.test_fold, class_name=args.class_name, is_train=False, input_size=args.input_size)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    reconsts = []
    mask1s = []
    mask2s = []
    boxes = []
    test_imgs = []
    p_gts = []
    i_gts = []
    time_flag = 0
    print("Time of preprocessing: ", time.perf_counter()  - start_time)
    end_time = time.perf_counter()
    rec_time1 = 0
    print("Reconstructing")
    for (data, p_gt, i_gt) in tqdm(test_loader):
        mask_image, box = get_mask(data.cpu().numpy(), args)
        test_imgs.extend(data.cpu().numpy())
        p_gts.extend(p_gt.cpu().numpy())
        i_gts.extend(i_gt.cpu().numpy())
        data = data.to(device)
        mask_image = []
        rect, mask1, mask2 = run_main(data, model_mae, mask_image, args.mask_size)
        mask_path = os.path.join(save_path, "mask")
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)
        for i in range(rect.shape[0]):
            reconsts.append(rect[i].cpu().detach().numpy())
            mask1s.append(mask1[i].cpu().detach().numpy())
            mask2s.append(mask2[i].cpu().detach().numpy())
            boxes.append(box)
        if time_flag == 0:
            rec_time1 = time.perf_counter()
            time_flag += 1
    rec_time2 = time.perf_counter()
    rec_time = rec_time2 - rec_time1
    mean_rectime = rec_time / (len(test_imgs) - args.batch_size)
    print("Mean time of Reconstruction: ", mean_rectime)
    print("Total time of Reconstruction: ", time.perf_counter() - end_time)
    end_time = time.perf_counter()
    # save reconstruction
    reconst_path = os.path.join(save_path, "rec")
    if not os.path.isdir(reconst_path):
        os.mkdir(reconst_path)
    imgnames = []
    for num in range(len(reconsts)):
        name = str(num)
        while len(name) < 3:
            name = "0" + name
        name = name + ".png"
        imgnames.append(name)
        reconst_save = os.path.join(reconst_path, name)
        cv2.imwrite(reconst_save, denormalize(reconsts[num]))
        cv2.imwrite(os.path.join(mask_path, (name[:-4] + "_1" + name[-4:])), denormalize(mask1s[num]))
        cv2.imwrite(os.path.join(mask_path, (name[:-4] + "_2" + name[-4:])), denormalize(mask2s[num]))
    # save original images
    ori_path = os.path.join(save_path, "ori")
    if not os.path.isdir(ori_path):
        os.mkdir(ori_path)
    for num in range(len(test_imgs)):
        ori_save = os.path.join(ori_path, imgnames[num])
        cv2.imwrite(ori_save, denormalize(test_imgs[num].transpose(1, 2, 0)))
    mean_savetime = (time.perf_counter()  - end_time) / len(test_imgs)   
    print("Time of save images: ", time.perf_counter()  - end_time)
    end_time = time.perf_counter()
    print("Detecting and Locating")
    # Detecte and Locate
    anomaly_maps = ssim_main(ori_path, save_path, reconst_path, device, model_mae, args.mask_size, args.is_multiss)
    mean_delotime = (time.perf_counter()  - end_time) / len(test_imgs) 
    print("Time of Detecting and Locating: ", time.perf_counter()  - end_time)
    end_time = time.perf_counter()
    # Visualize
    print("Visualizing")
    savename = os.path.join(save_path, "defect")
    if not os.path.isdir(savename):
        os.mkdir(savename)
    blend(ori_path, anomaly_maps, savename, boxes)
    mean_vistime = (time.perf_counter()  - end_time) / len(test_imgs) 
    print("Time of Visualizing: ", time.perf_counter()  - end_time)
    end_time = time.perf_counter()
    # Calculate indexes
    print("Calculating indexes")
    rocauc_name = os.path.join(save_path, "rocauc.txt")
    mean_rocauc, mean_p, mean_r, mean_f1, img_roc_auc, pbest, rbest, f1best = rocauc(anomaly_maps, rocauc_name, p_gts, i_gts)
    mean_caltime = (time.perf_counter()  - end_time) / len(test_imgs) 
    print("Time of Calculate indexes: ", time.perf_counter()  - end_time)
    end_time = time.perf_counter()
    print("Total time:", time.perf_counter()  - start_time)
    mean_time = mean_rectime + mean_savetime + mean_delotime + mean_vistime + mean_caltime
    print("Cost time of per image: ", mean_time, mean_rectime, mean_savetime, mean_delotime, mean_vistime, mean_caltime)
    headers = (" ", "AUROC", "Precision", "Recall", "F1-score")
    image_indexes = ("Image_level", img_roc_auc, pbest, rbest, f1best)
    pixel_indexes = ("Pixel_level", mean_rocauc, mean_p, mean_r, mean_f1)
    print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format(*headers))
    print('{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(*image_indexes))
    print('{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(*pixel_indexes))


if __name__ == '__main__':
    args = parse_args()
    print(args.input_size, args.mask_size, args.batch_size)
    main(args)
