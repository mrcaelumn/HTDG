from FewShot_models.training_parallel import *
import FewShot_models.functions as functions
import numpy as np
import torch.utils.data
from datetime import datetime
from Dataloaders.mura_loader import download_class_mura
from Dataloaders.folder_loader import download_class_folder

from anomaly_detection_evaluation import anomaly_detection
from defect_detection_evaluation import defect_detection

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--dataset', help='mura/leather/tile/carpet', default='mura')
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--pos_class', help='normal class', default=0)
    parser.add_argument('--random_images_download', help='random selection of images', default=False)
    parser.add_argument('--num_images', type=int, help='number of images to train on', default=20)
    parser.add_argument('--if_download', type=bool, help='do you want to download class', default=False)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--size_image', type=int, help='size orig image', default=64)
    parser.add_argument('--num_epochs', type=int, help='num epochs', default=1)
    parser.add_argument('--policy', default='')
    parser.add_argument('--niter_gray', help='number of iterations in each scale', type=int, default=500)
    parser.add_argument('--niter_rgb', help='number of iterations in each scale', type=int, default=1000)
    parser.add_argument('--index_download', help='index in dataset for starting download', type=int, default=1)
    # parser.add_argument('--use_internal_load', help='using another dataset', default=False)
    parser.add_argument('--experiment', help='task to be done', default='stop_signs')
    parser.add_argument('--test_size', help='test size', type=int, default=15000)
    parser.add_argument('--num_transforms', help='54 for rgb, 42 for grayscale', type=int, default=42)
    parser.add_argument('--device_ids', help='gpus ids in format: 0/ 0 1/ 0 1 2..', nargs='+', type=int, default=0)
    parser.add_argument('--fraction_defect', help='fraction of patches to consider in each scale', nargs='+', type=float, default=0.1)


    opt = parser.parse_args()
    
    start_time = datetime.now()
    
    scale = opt.size_image
    # pos_class = opt.pos_class
    random_images = opt.random_images_download
    num_images = opt.num_images
    opt.num_transforms = opt.num_transforms
    dataset = opt.dataset
    opt.niter = 54
    opt.niter_rgb
    # opt.size_image = 128
    # opt.input_dir = 
    # if opt.if_download == True:
    if dataset == 'mura':
        opt.num_transforms, opt.niter = 42, opt.niter_rgb
        opt.input_name = download_class_mura(opt)
    else:
        opt.num_transforms, opt.niter = 42, opt.niter_gray
        opt.input_name = download_class_folder(opt)
 

    print(opt.input_name)
    opt = functions.post_config(opt)
    Gs = []
    Zs,NoiseAmp = {}, {}
    
    reals_list = torch.FloatTensor(num_images,1,int(opt.nc_im), int(opt.size_image), int(opt.size_image)).cuda()
    
    for i in range(num_images):
        real = img.imread("%s/%s_%d.png" % (opt.input_dir, opt.input_name[:-4], i))    
        real = functions.np2torch(real, opt)
        real = real[:, 0:3, :, :]
        # print(real.shape)
        functions.adjust_scales2image(real, opt)
        
    dir2save = functions.generate_dir2save(opt)
    reals = {}
    lst =  np.load("TrainedModels/" + str(opt.input_name)[:-4] +  "/transformations.npy")
    opt.list_transformations = lst
    print(real.shape)
    
    if opt.mode == 'train':
        train(opt, Gs, Zs, reals, NoiseAmp)
    
    f = open(f"{str(opt.input_name)[:-4]}_duration_training.txt",'w')
    end_time = datetime.now()
    TRAINING_DURATION = end_time - start_time
    print(f'Duration of Training: {TRAINING_DURATION}', file=f)
    
    # testing    
    anomaly_detection(opt.input_name, opt.test_size, opt)
