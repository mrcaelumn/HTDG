from PIL import Image
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import cv2
import os
import itertools
import random


def download_class_mnist(opt):
    opt.input_name = "mnist_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(opt.pos_class) \
                     + "_indexdown" + str(opt.index_download) + ".png"
    scale = opt.size_image
    pos_class = opt.pos_class
    num_images = opt.num_images

    def imsave(img, i):
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((scale, scale))])
        im = transform(img)
        im.save("Input/Images/mnist_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy) + "_" + str(pos_class)
                    + "_indexdown" +str(opt.index_download) + "_" + str(i) + ".png")

    if opt.mode == "train":
        trainset = datasets.MNIST('./mnist_data', download=True, train=True)
        train_data = np.array(trainset.data)
        train_labels = np.array(trainset.targets)
        
        print("train_data: ", train_data.shape)
        print("train_labels: ", train_labels.shape)
    
        train_data = train_data[np.where(train_labels == int(pos_class))]

        dicty = {}
        if opt.random_images_download == False:
            count_images, step_index = 0, 0
            for i in range(len(train_data)):
                t = train_data[i]
                imsave(t, count_images)
                dicty[count_images] = i
                count_images += 1
                if count_images == num_images: step_index += 1
                if step_index == opt.index_download: break
                if count_images == num_images and step_index != opt.index_download: count_images = 0
            training_images = list(dicty.values())
        else:
            random_index = random.sample(range(0, len(train_data)), opt.num_images)
            training_images = list(random_index)
            for i in range(len(training_images)):
                index = training_images[i]
                t = train_data[index]
                imsave(t, i)
        print("training imgaes: ", training_images)

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

    path = "mnist_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if os.path.exists(path) == False:
        os.mkdir(path)
    mnist_testset = datasets.MNIST('./mnist_data', download=True, train=False)
    
    test_data = np.array(mnist_testset.data)
    test_labels = np.array(mnist_testset.targets)
    
    print("test_data: ", test_data.shape)
    print("test_labels: ", test_labels.shape)
    
    test_data = torch.from_numpy(test_data).unsqueeze(3)
    test_data = test_data.repeat(1, 1,1, 3).numpy()
    test_data = test_data.transpose((0, 3, 1, 2)) / 255
    test_data = np.array(test_data)
    mnist_target_new = np.zeros((test_labels.shape))
    mnist_target_new[test_labels == int(pos_class)] = 1
    mnist_target_new[test_labels != int(pos_class)] = 0
    np.save(path + "/mnist_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", test_data)
    np.save(path + "/mnist_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy", mnist_target_new)

    opt.input_name = "mnist_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_" + str(pos_class) \
                     + "_indexdown" + str(opt.index_download) + ".png"
    return opt.input_name
