import os.path
from tqdm import tqdm
import cv2
import numpy as np

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import cv2
import os
from tqdm import tqdm


def generate_segments(im_orig, scale, sigma, min_size):
    im_mask = skimage.segmentation.felzenszwalb(skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
                                                min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_orig = np.append(im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig


def extract_regions(img):
    R = {}
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    # pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {"min_x": 0xffff, "min_y": 0xffff,
                        "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: calculate texture gradient
    tex_grad = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        tex_grad[:, :, colour_channel] = skimage.feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)

    # pass 3: calculate color histogram of each region
    for k, v in list(R.items()):
        # color histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        img_t = tex_grad[:, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        hist_c = np.array([])
        hist_t = np.array([])
        for colour_channel in (0, 1, 2):
            # extracting one color channel
            c = img[:, colour_channel]
            t = img_t[:, colour_channel]
            # calculate histogram for each colour and join to the result
            hist_c = np.concatenate([hist_c] + [np.histogram(c, 25, (0.0, 255.0))[0]])
            hist_t = np.concatenate([hist_t] + [np.histogram(t, 10, (0.0, 1.0))[0]])
        # L1 normalize
        R[k]["hist_c"] = hist_c / len(img)
        R[k]["hist_t"] = hist_t / len(img_t)
    return R


def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def calc_sim(r1, r2, imsize):
    sim_color = sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])
    sim_texture = sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])
    sim_size = 1.0 - (r1["size"] + r2["size"]) / imsize
    bbsize = ((max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"])) * (
                max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"])))
    sim_fill = 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize
    return sim_color + sim_texture + sim_size + sim_fill


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    assert im_orig.shape[2] == 3, "3ch image is expected"
    img = generate_segments(im_orig, scale, sigma, min_size)
    if img is None:
        return None, {}
    imsize = img.shape[0] * img.shape[1]
    R = extract_regions(img)
    # extract neighbouring information
    neighbours = extract_neighbours(R)
    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)
    # hierarchal search
    while S != {}:
        # get the highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions

def get_mask(image_all, args):
    # mask_images = np.zeros((image_all.shape[0]))
    mask_images = []
    mask_size = 16
    image_size = args.input_size
    for image_id in range(image_all.shape[0]):
        image = image_all[image_id]
        img = image.transpose(1, 2, 0)
        img_lbl, regions = selective_search(img, scale=400, sigma=0.9, min_size=10)
        xs = []
        ys = []
        ws = []
        hs = []
        for i in range(len(regions)):
            region = regions[i]['rect']
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            if w * h < 50:
                continue
            if x >= 15 and x + w <= img.shape[1] - 15:
                xs.append(x)
                ys.append(y)
                ws.append(x + w)
                hs.append(y + h)
        xmin = min(xs)
        xmax = max(ws)
        ymin = min(ys)
        ymax = max(hs)
        
        # 找出陶瓷片区域
        mask_image = np.zeros(img.shape, dtype=np.float32)
        cv2.rectangle(mask_image, (xmin, mask_size), (xmax, mask_image.shape[1] - mask_size), (255, 255, 255), -1)
        # cv2.rectangle(mask_image, (xmin, ymin - mask_size), (xmax, ymax + mask_size), (255, 255, 255), -1)
        # 拆分成patches
        patches_num = int(image_size / mask_size)
        # print(patches_num)
        image_patches = []
        for row in range(patches_num):
            for col in range(patches_num):
                image_patch = mask_image[row * mask_size:(row + 1) * mask_size, col * mask_size:(col + 1) * mask_size]
                image_area_all = np.sum(image_patch == 255) + np.sum(image_patch == 0)
                image_area = np.sum(image_patch == 255)
                if image_area > image_area_all / 4:
                    image_patches.append(1)
                else:
                    image_patches.append(0)
        image_patches = np.array(image_patches)
        mask_images.append(image_patches)
    return mask_images, (xmin, xmax, ymin, ymax)


def get_GT(image_all, args):
    # mask_images = np.zeros((image_all.shape[0]))
    mask_images = []
    mask_size = 16
    image_size = args.input_size
    for image_id in range(image_all.shape[0]):
        image = image_all[image_id]
        # img = image.transpose(1, 2, 0)
        img = image[0]
        img = cv2.resize(img, (image_size, image_size))
        # print(img.max())
        # 找出GT区域，并拆分成patches
        patches_num = int(image_size / mask_size)
        # print(patches_num)
        image_patches = []
        for row in range(patches_num):
            for col in range(patches_num):
                image_patch = img[row * mask_size:(row + 1) * mask_size, col * mask_size:(col + 1) * mask_size]
                image_area_all = np.sum(image_patch == 1) + np.sum(image_patch == 0)
                image_area = np.sum(image_patch == 1)
                if image_area > 0:
                    image_patches.append(1)
                else:
                    image_patches.append(0)
        image_patches = np.array(image_patches)
        # print(image_patches.max())
        mask_images.append(image_patches)
    return mask_images


if __name__ == '__main__':
    img_folder = r"G:\insulator_data\final\insulator\test"
    # img_folder = r"G:\insulator_data\old\insulator_data\test"
    # img_folder = r"F:\Code\Python\FILE\anomaly_detection\result\fre\noblack2"
    names = os.listdir(img_folder)
    for name in names:
        # if name != "breakage":
        #     continue
        image_file = os.path.join(img_folder, name)
        image_names = os.listdir(image_file)
        # 创建文件夹
        # 检查文件夹是否存在
        if not os.path.exists("./result/exp/black1d/" + name):
            # 如果不存在，则创建文件夹
            os.makedirs("./result/exp/black1d/" + name)
        for image_name in tqdm(image_names):
            image_name = os.path.join(image_file, image_name)
            # img_man = cv2.imread(image_name, 0)  # 直接读为灰度图像
            # img_man = cv2.resize(img_man, (224, 224))
            img = skimage.io.imread(image_name)
            img = skimage.transform.resize(img, (224, 224))
            img_lbl, regions = selective_search(img, scale=400, sigma=0.9, min_size=10)
            xs = []
            ys = []
            ws = []
            hs = []
            for i in range(len(regions)):
                region = regions[i]['rect']
                x = region[0]
                y = region[1]
                w = region[2]
                h = region[3]
                if w * h < 50:
                    continue
                if x >= 15 and x + w <= img.shape[1] - 15:
                    xs.append(x)
                    ys.append(y)
                    ws.append(x + w)
                    hs.append(y + h)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            xmin = min(xs)
            xmax = max(ws)
            ymin = min(ys)
            ymax = max(hs)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # print(regions[:10])
            cv2.imwrite("./result/exp/black1d/" + name + "/" + image_name[-7:-4] + "_all" + image_name[-4:], img * 255)