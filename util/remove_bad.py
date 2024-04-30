import os.path
from tqdm import tqdm
from PIL import Image
from numpy import average, dot, linalg
import shutil


def get_thum(image, size=(64, 64), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


# Calculate the cosine distance of the picture
def image_similarity_vectors_via_numpy(image, image1, image2):
    image = get_thum(image)
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image, image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b1, b2 = vectors
    a_norm, b_norm1, b_norm2 = norms
    res1 = dot(a / a_norm, b1 / b_norm1)
    res2 = dot(a / a_norm, b2 / b_norm2)
    return res1, res2


def remove(picpath, despath):
    # picpath = os.path.join(picpath, "insulator")
    # despath = os.path.join(despath, "insulator")
    if not os.path.isdir(despath):
        os.mkdir(despath)
    imagelist = []
    for f in os.listdir(picpath):
        if f.endswith(".jpg") or f.endswith(".png"):
            imagelist.append(f)
    image_ori1 = Image.open(os.path.join(picpath, "0002.jpg"))
    image_ori2 = Image.open(os.path.join(picpath, "0005.jpg"))
    cosins = []
    for i in tqdm(range(len(imagelist))):
        ori = os.path.join(picpath, imagelist[i])
        image1 = Image.open(ori)
        cosin1, cosin2 = image_similarity_vectors_via_numpy(image1, image_ori1, image_ori2)
        cosin = cosin1 if cosin1 > cosin2 else cosin2
        cosins.append(cosin)
    mean_cos = sum(cosins) / len(cosins)
    print(mean_cos)
    for i in tqdm(range(len(cosins))):
        if cosins[i] > mean_cos:
            ori = os.path.join(picpath, imagelist[i])
            dec = os.path.join(despath, imagelist[i])
            shutil.copy(ori, dec)