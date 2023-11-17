# Adopted from: https://github.com/allenai/elastic/blob/master/multilabel_classify.py
# special thanks to @hellbell

import os
import cv2
import time
import tqdm
import argparse
import numpy as np
from curses import raw
from PIL import ImageFile
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torchvision import models
import torchvision.transforms as transforms

from src.helper_functions.helper_functions_doublenet import mAP, AverageMeter, OCTDataset, OCTDataset_CAM, evaluation_cal_multilabel
from src.models import create_model
from src.models.resnet import *
from src.models.octnet import *
from src.models.resoctnet import *
from src.models.vggnet import *
from efficientnet_pytorch import EfficientNet

from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_auc_score
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from torchsummary import summary

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='../multi-label-classifier/data/OCT/total_images')
parser.add_argument('--num-classes', default=4)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
					metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.4, type=float,
					metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
					metavar='N', help='print frequency (default: 64)')


def main(backbone):
	args = parser.parse_args()
	args.batch_size = args.batch_size
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	model_name = 'vgg-doublenet-224-448-rotate-lr003.ckpt'
	print('Using {} with threshold {}'.format(model_name, args.thre))

	# setup model
	print('Creating and loading the model...')
	# model = ResNet18_Dual(num_classes=3)
	# model = OCTNet_Dual(num_classes=3)
	# model = ResOCTNet_Dual_1Input(num_classes=3)
	# model = ResOCTNet_Dual()
	model = VGG_Dual(num_classes=3)

	model.load_state_dict(torch.load('./models/8020_42/{}'.format(model_name)))
	model = model.cuda()
	model.eval()
	# print('done\n')
	
	# x=torch.rand((1,1,448,448))
	# flops = FlopCountAnalysis(model, x.cuda())
	# acts = ActivationCountAnalysis(model, x.cuda())
	# print(f"total flops : {(flops.total()+acts.total())/1e9}",'G')

	# summary(model, (1, 448, 448))
	# repetitions = 1000
	# total_time_1 = 0
	# for i in range(repetitions):
	# 	starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
	# 	starter.record()
	# 	_ = model(x.cuda())
	# 	ender.record()
	# 	torch.cuda.synchronize()
	# 	curr_time = starter.elapsed_time(ender)/1000
	# 	total_time_1 += curr_time
	# print(total_time_1/repetitions*1000)
	
	# total_time_2 = 0
	# for i in range(repetitions):
	# 	start_time= time.time() # set the time at which inference started
	# 	_ = model(x.cuda())
	# 	stop_time=time.time()
	# 	duration =stop_time - start_time
	# 	hours = duration // 3600
	# 	minutes = (duration - (hours * 3600)) // 60
	# 	seconds = duration - ((hours * 3600) + (minutes * 60))
	# 	total_time_2 += seconds
	# print(total_time_2/repetitions*1000)
	
	instances_path = os.path.join('./data', 'test_6_80_20_updated.csv')
	data_path = args.data

	if backbone in ['resnet', 'effnet', 'inception']:
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
	else:
		mean = [0.5]
		std = [0.5]

	print('Mead and std used: ', mean, std)

	val_transform_b = transforms.Compose([transforms.Resize((args.image_size*2, args.image_size*2)),
						transforms.ToTensor(), transforms.Normalize(mean, std)])
	
	val_transform_S = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
						transforms.ToTensor(), transforms.Normalize(mean, std)])

	val_dataset = OCTDataset(data_path, instances_path, val_transform_b, val_transform_S)

	cam_dataset = OCTDataset_CAM(data_path, instances_path,	
									val_transform_b, val_transform_S,
									transforms.Compose([
									transforms.ToPILImage(),
									transforms.Resize((448, 448)),
									transforms.ToTensor()
								]))

	print("len(test_dataset)): ", len(cam_dataset))

	val_loader = torch.utils.data.DataLoader(
					val_dataset, batch_size=args.batch_size, shuffle=False,
					num_workers=args.workers, pin_memory=True)

	validate_multi(val_loader, model, args)


def validate_multi(val_loader, model, args):
	print("Starting validation ...")
	classes = ['AMD', 'ERM', 'M.Edema', 'Normal']
	# font                   = cv2.FONT_HERSHEY_SIMPLEX
	# bottomLeftCornerOfText = (125,492)
	# topLeftCornerOfText    = (125,20)
	# fontScale              = 0.75
	# fontColor              = (255,255,255)
	# thickness              = 2
	# lineType               = 2
	
	# target_layer_b = [model.resoctnet1.block6]
	# target_layer_s = [model.resoctnet2.block6]
	# cam_b = GradCAMPlusPlus(model=model, target_layers=target_layer_b, use_cuda=False)
	# cam_s = GradCAMPlusPlus(model=model, target_layers=target_layer_s, use_cuda=True)
	
	sigmoid = torch.nn.Sigmoid()
	# criterion = nn.MultiLabelSoftMarginLoss()

	preds = []
	targets = []
	total_time = 0.0
	image_lists = []
	# for input_b, input_s, raw_input, target, img_id in tqdm.tqdm(val_loader):
	for input_b, input_s, target in tqdm.tqdm(val_loader):
		target = target.cuda()
		input_b, input_s = input_b.cuda(), input_s.cuda()
		# pred_names = ['Prediction: ']
		# gt_names = ['Label: ']
		# image_lists += img_id

		# compute output
		time1 = time.time()
		with torch.no_grad():
			# raw_input = raw_input.cpu().detach().numpy()
			# raw_input = raw_input.squeeze(0)
			with autocast():
				output = sigmoid(model(input_b, input_s)) #.cpu()
		total_time += (time.time() - time1)

		# output_indice = output.cpu().detach().numpy() > args.thre
		# if not (target.cpu().numpy().astype('double') == output_indice.astype('double')).all():
		# 	for i in range(len(output_indice[0])):
		# 		if output_indice[0][i] == True:
		# 			pred_names.append(classes[i])
		# 		if target.cpu().numpy()[0][i] == 1.0:
		# 			gt_names.append(classes[i])
		# 	if len(pred_names) == 1:
		# 		pred_names.append('Normal')
		# 	if len(gt_names) == 1:
		# 		gt_names.append('Normal')

		# 	grad_maps = [cam_b(input_tensor=[input_b, input_s], targets=[ClassifierOutputTarget(i)]) for i in range(3)]
		# 	for i in range(1, 3):
		# 		grad_maps[0][0,:] += grad_maps[i][0,:]

		# 	heatmap = cv2.applyColorMap(np.uint8(255/3*grad_maps[0][0,:]), cv2.COLORMAP_JET)
		# 	heatmap = np.float32(heatmap) / 255
		# 	cam_output = cv2.resize(heatmap, (448, 448)) + raw_input.transpose(1,2,0)
		# 	cam_output = cam_output / np.max(cam_output)
		# 	img = np.uint8(255 * cam_output)
			
			# raw_input = raw_input.transpose(1,2,0)
			# raw_input = np.uint8(raw_input*255)

			# cv2.putText(img, ' '.join(pred_names),
			# 			bottomLeftCornerOfText,
			# 			font,
			# 			fontScale,
			# 			fontColor,
			# 			thickness,
			# 			lineType)

			# cv2.putText(raw_input, ' '.join(gt_names),
			# 			topLeftCornerOfText,
			# 			font,
			# 			fontScale,
			# 			fontColor,
			# 			thickness,
			# 			lineType)
			
			# cv2.imwrite('./grad_cam/{}/{}'.format('doublenet_gradcam', img_id[0]), raw_input)
			# del pred_names
			# del gt_names

		# for mAP calculation
		preds.append(output)
		targets.append(target)

		# measure accuracy and record loss
	# time2 = time.time()
	preds = torch.cat(preds).cpu().numpy()
	targets = torch.cat(targets).cpu().numpy()
	print(roc_auc_score(targets, preds, average=None))
	final_preds = preds > args.thre
	acc_list, sen_list, spe_list, _, _, _ = evaluation_cal_multilabel(targets, final_preds, is_train=False)
	print('Acc list: ', acc_list)
	print('Sen list: ', sen_list)
	print('Spe list: ', spe_list)
	print("Total inference time: ", total_time)
	
	# outputs_list = preds.tolist()

	# with open('./results/ms_srn.csv', 'w', newline='', encoding='utf-8') as f:
	# 	writer = csv.writer(f)
	# 	writer.writerow(['id', 'amd', 'erm', 'macular_edema'])
	# 	for i in range(len(image_lists)):
	# 		writer.writerow([image_lists[i], outputs_list[i][0], outputs_list[i][1], outputs_list[i][2]])

	return True


if __name__ == '__main__':
	main('resnet')
