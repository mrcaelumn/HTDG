from __future__ import print_function
from FewShot_models.manipulate import *
from FewShot_models.training_parallel import *
from FewShot_models.imresize import imresize, imresize_to_shape
import FewShot_models.functions as functions
import FewShot_models.models as models
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import os, sys
from tqdm import tqdm

def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(result_folder + name_model+'_roc_curve.png')
    plt.show()
    plt.clf()

''' calculate the auc value for lables and scores'''
def roc(labels, scores, name_model):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    # print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshod that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_threshold

def plot_anomaly_score(score_ano, labels, name, model_name):
    
    df = pd.DataFrame(
    {'predicts': score_ano,
     'label': labels
    })
    
    df_normal = df[df.label == 0]
    sns.distplot(df_normal['predicts'],  kde=False, label='normal')

    df_defect = df[df.label == 1]
    sns.distplot(df_defect['predicts'],  kde=False, label='defect')
    
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Number of samples')
    plt.legend(prop={'size': 12})
    plt.savefig(model_name+ '_'+name+'_anomay_scores_dist.png')
    plt.show()
    plt.clf()
    
def calculate_metrics(y_true, y_preds, name):
    auc_out, threshold = roc(y_true, y_preds, name)     
            
    # histogram distribution of anomaly scores
    plot_anomaly_score(y_preds, y_true, "anomaly_score_dist", name)

    y_preds = (y_preds > threshold).astype(int)

    TN, FP, FN, TP = confusion_matrix(y_true, y_preds).ravel()


    arr_result = [
        f"Model Spec: {name}",
        f"AUC: {auc_out}",
        f"Threshold: {threshold}",
        f"False Alarm Rate (FPR): {(FP/(FP+TN))}", 
        f"TNR: {(TN/(FP+TN))}", 
        f"Precision Score (PPV): {(TP/(TP+FP))}", 
        f"Recall Score (TPR): {(TP/(TP+FN))}", 
        f"NPV: {(TN/(FN+TN))}", 
        f"F1-Score: {(f1_score(y_true, y_preds))}", 
    ]
    # print("\n".join(arr_result))
    return "\n".join(arr_result)

def anomaly_detection(input_name_model,test_size, opt):
    scale = int(opt.size_image)
    pos_class = opt.pos_class

    alpha = int(opt.alpha)
    data = opt.dataset
    num_images = opt.num_images

    path = str(data) + "_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    if (os.path.exists(path)==True):

        xTest_input = np.load(path + "/" + str(data) + "_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
        yTest_input = np.load(path + "/" + str(data) + "_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")

    else:
        if os.path.exists(path) == False:
            print("path not exists")
            exit()

    xTest_input = xTest_input[:test_size]
    yTest_input = yTest_input[:test_size]

    num_samples = xTest_input.shape[0]
    batch_size = 1
    batch_n = num_samples // batch_size

    opt.input_name = input_name_model
    opt.num_images = 1

    path = "TrainedModels/" + str(opt.input_name)[:-4] + \
           "/scale_factor=0.500000,alpha=" + str(alpha)
    transformations_list = np.load("TrainedModels/" + str(opt.input_name)[:-4] +  "/transformations.npy")
    probs_predictions = []
    
    # print("xTest_input.shape:", xTest_input.shape)
    
    real = torch.from_numpy(xTest_input[0]).cuda().unsqueeze(0).permute((0, 3, 1, 2))
    # print("real.size():", real.size())
    real = imresize_to_shape(real, (scale, scale), opt)

    functions.adjust_scales2image(real, opt)


    scores_per_scale_dict = torch.from_numpy(np.zeros((opt.stop_scale+1,batch_n))).cuda()

    def compute_normalized_dict(scores_per_scale_dict):
        for scale in range(0, opt.stop_scale + 1):
            maxi = torch.max(scores_per_scale_dict[scale])
            mini = torch.min(scores_per_scale_dict[scale])
            scores_per_scale_dict[scale] = (scores_per_scale_dict[scale] - mini) / (maxi - mini)
        return scores_per_scale_dict

    with torch.no_grad():
        for i in tqdm(range(batch_n), desc='testing stages'):
        # for i in range(batch_n):
            reals = {}
            # print("xTest_input.shape: ", xTest_input.shape)
            real = torch.from_numpy(xTest_input[i]).unsqueeze(0).cuda()
            # print("real batch", real.size())
            real = functions.norm(real)

            # real = real[:, 0:3, :, :]
            # print("real 1", real.size())
            # print("real 2", real.shape[-1])
            if real.shape[-1] == 3:
                real = real.permute(0, 3, 1, 2)
                # real = real.transpose(0, 3, 1, 2)
            real = imresize_to_shape(real, (scale,scale), opt)
            functions.adjust_scales2image(real, opt)
            real = imresize(real, opt.scale1, opt)
            for index_image in range(int(opt.num_images)):
                reals[index_image] = []
                reals = functions.creat_reals_pyramid(real, reals, opt,index_image)

            err_total,err_total_avg, err_total_abs = [],[],[]

            for scale_num in range(0, opt.stop_scale+1, 1):
                opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                netD = models.WDiscriminatorMulti(opt)
                if torch.cuda.device_count() > 1:
                    netD = DataParallelModel(netD, device_ids=opt.device_ids)
                netD.to(opt.device)
                netD.load_state_dict(torch.load('%s/%d/netD.pth' % (path, scale_num)))
                netD.eval()

                err_scale = []

                for index_image in range(int(opt.num_images)):
                    score_image_in_scale = 0
                    reals_transform = []
                    for index_transform, pair in enumerate(transformations_list):
                        real = reals[index_image][scale_num].to(opt.device)
                        flag_color,is_flip, tx, ty, k_rotate = pair
                        real_augment = apply_augmentation(real, is_flip, tx, ty, k_rotate,flag_color).to(opt.device)
                        real_augment = torch.squeeze(real_augment)
                        reals_transform.append(real_augment)
                    real_transform = torch.stack(reals_transform)
                    output = netD(real_transform)
                    if isinstance(output, list):
                        output = [tens.to(opt.device) for tens in output]
                        output = torch.cat(output).detach()
                    else:
                        output = output.to(opt.device)
                    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
                    shape = reshaped_output.shape
                    reshaped_output = reshaped_output.view(-1, shape[3])  # 25,73
                    reshaped_output = reshaped_output[:, :opt.num_transforms]  # 1,72,5,5
                    m = nn.Softmax(dim=1)
                    score_temp = m(reshaped_output)
                    score_all = score_temp.reshape(opt.num_transforms, -1, opt.num_transforms)
                    for j in range(opt.num_transforms):
                            current = score_all[j]
                            score_temp = current[:, j]
                            score_temp = torch.mean(score_temp)
                            score_image_in_scale += score_temp
                    err_scale.append(score_image_in_scale)

                err_scale = torch.stack(err_scale)

                err = torch.max(err_scale, dim=0)[0]
                err = torch.mean(err).item()
                scores_per_scale_dict[scale_num][i] = (err)
                err_total.append(err)
                del netD
            avg_err_total = np.mean(err_total)

            probs_predictions.append(avg_err_total)

            if i > 99 and i % 100 == 0:
                # print(i)
                try:
                    # print("name model: ", opt.input_name)
                    with open(opt.input_name + ".txt", "w") as text_file:
                        auc1 = roc_auc_score(yTest_input[:i], probs_predictions[:i])
                        print("roc_auc_score  all ={}".format(auc1), file=text_file)
                except Exception as e:
                    print(e)

        with open(opt.input_name + ".txt", "w") as text_file:

            print(pos_class, "results: ", file=text_file)
            print(" ", file=text_file)
            print("results without norm, without top_k: ", file=text_file)
            
            result = calculate_metrics(yTest_input, probs_predictions, opt.input_name)
            print("results without norm: ", result, file=text_file)
            
            
            # auc1 = roc_auc_score(yTest_input, probs_predictions)
            # print("roc_auc_score (not normal) all ={}".format(auc1), file=text_file)
            
            # # false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest_input, probs_predictions)
            # print(f'TNR={false_positive_rate}, TPR={true_positive_rate}, threshold={thresholds}', file=text_file)
            # roc_score = auc(false_positive_rate, true_positive_rate)
            
            # print(f"roc_auc_score (not normal) all={auc_out}, threshold={threshold}", file=text_file)
            
            
            
            
            
            scores_per_scale_dict_norm = compute_normalized_dict(scores_per_scale_dict)
            scores_per_scale_dict_norm = scores_per_scale_dict_norm.cpu().clone().numpy()
            scores_per_scale_dict_norm = np.nan_to_num(scores_per_scale_dict_norm)
            # print(scores_per_scale_dict_norm)
            print(scores_per_scale_dict_norm.shape)
            print(" ", file=text_file)
            print("results with normalization, without top_k: ", file=text_file)

            probs_predictions_norm_all = np.mean(scores_per_scale_dict_norm, axis=0)
            
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest_input, probs_predictions_norm_all)
            # # print(f'TNR={false_positive_rate}, TPR={true_positive_rate}, threshold={thresholds}', file=text_file)
            # roc_score = auc(false_positive_rate, true_positive_rate)
            
            result = calculate_metrics(yTest_input, probs_predictions_norm_all, opt.input_name)
            print("results with normalization: ", result, file=text_file)
            
            # print("roc_auc_score T1 normalize all ={}".format(roc_score), file=text_file)

    path = str(data) + "_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(num_images)
    os.remove(path + "/" + str(data) + "_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
    os.remove(path + "/" + str(data) + "_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
    del xTest_input, yTest_input
