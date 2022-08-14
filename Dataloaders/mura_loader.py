from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import itertools
import random

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def download_class_mura(opt):
    opt.input_name = "mura_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
                     + "_indexdown" + str(opt.index_download) + ".png"
    class_to_idx = {
        "normal": 0,
        "defect": 1
    }
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images

    def imsave(img, i):
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((scale, scale)),
        ])
        im = transform(img)
        im.save("Input/Images/mura_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                    + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")

    def load_dataset_from_folder(path, c_transforms=None, bs=None, shuf=True):
        
        dataset = datasets.ImageFolder(
            root=path,
            transform=c_transforms,
            class_to_idx=class_to_idx,
        )
        # print(dataset.classes)
        # print(dataset.class_to_idx)
        trainset = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuf)
        
        trainset_data = []
        trainset_targets = []
        
        for images, labels in trainset:
            img = images.permute(1, 2, 0).numpy()
            img = convert(img, 0, 255, np.uint8)
            trainset_data.append(img)
            trainset_targets.append(labels)
            # print(images, labels)
    
        
        data = np.array(trainset_data)
        labels = np.array(trainset_targets)
        print("load dataset: ", path, "data", data.shape, "labels", labels.shape)
        
        return data, labels
    
    if opt.mode == "train":
        
        # trainset = datasets.MNIST('./mnist_data', download=True, train=True)
        
        train_data, train_labels = load_dataset_from_folder('Data/mura_march_clean/train_data', c_transforms=transforms.ToTensor())
        
        train_data = train_data[np.where(train_labels == int(pos_class))]

        dicty = {}
        
        random_index = random.sample(range(0, len(train_data)), opt.num_images)
        training_images = list(random_index)
        for i in range(len(training_images)):
            index = training_images[i]
            t = train_data[index]
            imsave(t, i)
            
        print("training images: ", training_images)

        genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,))
        genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3))
        genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,))
        genertator = itertools.chain(genertator0, genertator1, genertator3)
        lst = list(genertator)
        random.shuffle(lst)
        path_transform = "TrainedModels/" + str(opt.input_name)[:-4]
        if os.path.exists(path_transform) == False:
            os.mkdir(path_transform)
        np.save(path_transform +  "/transformations.npy", lst)

    path = "mura_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path) == False:
        os.mkdir(path)
        
    # mura_testset = datasets.MNIST('./mnist_data', download=True, train=False)
    
    
#     test_data = np.array(mura_testset.data)
#     test_labels = np.array(mura_testset.targets)
    
    test_data, test_labels = load_dataset_from_folder('Data/mura_march_clean/test_data', c_transforms=transforms.ToTensor())
    
    np.save(path + "/mura_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", test_data)
    np.save(path + "/mura_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", test_labels)

    opt.input_name = "mura_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
                     + "_indexdown" + str(opt.index_download) + ".png"
    return opt.input_name
