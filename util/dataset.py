import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, dataset_path='./data', test_fold="test", class_name='insulator', is_train=True, input_size=224):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.test_fold = test_fold  # 修改test_real、test_sim、test_all
        self.is_train = is_train
        self.input_size = input_size
        # load dataset
        self.images, self.img_gts, self.pixel_gts = self.load_dataset_folder()
        # set transforms
        self.transform_image = transforms.Compose([transforms.Resize(self.input_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        self.transform_gts = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        images, img_gts, pixel_gts = self.images[idx], self.img_gts[idx], self.pixel_gts[idx]

        images = Image.open(images).convert('RGB')
        w, h = images.size
        images = self.transform_image(images)

        if img_gts == 0:
            pixel_gts = Image.fromarray(np.zeros((w, h)))
        else:
            pixel_gts = Image.open(pixel_gts).convert('L')
        pixel_gts = self.transform_gts(pixel_gts)

        return images, pixel_gts, img_gts

    def __len__(self):
        return len(self.images)

    def load_dataset_folder(self):
        images = []
        pixel_gts = []
        img_gts = []
        test_imgdir = os.path.join(self.dataset_path, self.class_name, self.test_fold)
        test_labdir = os.path.join(self.dataset_path, self.class_name, 'ground_truth', self.test_fold)
        test_imgs = sorted(os.listdir(test_imgdir))
        for test_img in test_imgs:
            img_fold = os.path.join(test_imgdir, test_img)
            imglists = sorted(os.listdir(img_fold))  # images
            if test_img == "good":
                for i in range(len(imglists)):
                    imagelist = os.path.join(img_fold, imglists[i])
                    images.append(imagelist)
                    pixel_gts.append(None)
                    img_gts.append(0)
            else:
                for i in range(len(imglists)):
                    imagelist = os.path.join(img_fold, imglists[i])
                    gtlist = os.path.join(test_labdir, test_img, (imglists[i][:-4] + ".png"))
                    images.append(imagelist)
                    pixel_gts.append(gtlist)
                    img_gts.append(1)

        assert len(images) == len(img_gts), 'number of x and y should be same'
        return list(images), list(img_gts), list(pixel_gts)

