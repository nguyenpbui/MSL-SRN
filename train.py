import os
import numpy as np
import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torchvision import models
from torchsummary import summary
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

from src.helper_functions.helper_functions import OCTDataset, evaluation_cal_multilabel
from src.models.octnet import OCTNet
from src.models.resnet import *
from src.models.resoctnet import *
from src.models.mobilenetv2 import *
from src.models.vggnet import *
from src.models.multi_scale_cnn import *
from efficientnet_pytorch import EfficientNet

import timm
from randaugment import RandAugment
from sklearn.metrics import multilabel_confusion_matrix, classification_report, accuracy_score


parser = argparse.ArgumentParser(description='PyTorch OCT Training')
parser.add_argument('--data', help='path to dataset', default='../multi-label-classifier/data/OCT/total_images')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
					metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.4, type=float,
					metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
					metavar='N', help='print frequency (default: 64)')


def main(backbone):
	args = parser.parse_args()
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Setup model
	print('Creating model...')
	# model = OCTNet(num_classes=3)
	# model = ResOCTNet(num_classes=3)
	# model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)
	# model = resnet50(pretrained=True, num_classes=3)
	# model = models.mobilenet_v2(num_classes=3)
	# model = mobilenetv2(num_classes=3)
	# model = vgg11(num_classes=3)
	model = MultiScaleCNN(num_classes=3)
	
	# model = models.alexnet(pretrained=True)
	# num_ftrs = model.classifier[6].in_features
	# model.classifier[6] = nn.Linear(num_ftrs, 3)

	# model = models.inception_v3(pretrained=True)
	# num_ftrs = model.AuxLogits.fc.in_features
	# model.AuxLogits.fc = nn.Linear(num_ftrs, 3)
	# num_ftrs = model.fc.in_features
	# model.fc = nn.Linear(num_ftrs, 3)
	# model = timm.create_model('resnet50', pretrained=True, num_classes=3)

	model = model.cuda()
	summary(model, (3, 224, 224))

	print('Done !!!\n')

	# COCO Data loading
	instances_path_test = os.path.join('./data', 'test_6_80_20_updated.csv')
	instances_path_train = os.path.join('./data', 'train_6_80_20.csv')
	data_path = args.data
	
	if backbone in ['resnet', 'effnet', 'inception']:
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
	else:
		mean = [0.5]
		std = [0.5]
	
	print(mean, std)
	valid_transform = transforms.Compose([
								transforms.ToPILImage(),
								transforms.Resize((args.image_size, args.image_size)),
								transforms.ToTensor(),
								transforms.Normalize(mean, std)
								])
	
	val_dataset = OCTDataset(data_path, instances_path_test, valid_transform)
	
	train_transform = transforms.Compose([
									transforms.ToPILImage(),
									transforms.Resize((args.image_size, args.image_size)),
									# transforms.RandomHorizontalFlip(),
									transforms.RandomRotation(15),
									transforms.ToTensor(),
									transforms.Normalize(mean, std)
								  ])

	train_dataset = OCTDataset(data_path, instances_path_train, train_transform)

	print("len(train_dataset)): ", len(train_dataset))
	print("len(val_dataset)): ", len(val_dataset))

	# Pytorch Data loader
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=False)

	# Actual Training
	train_multi_label(model, train_loader, val_loader, args.lr)


def train_multi_label(model, train_loader, val_loader, lr):

	# set optimizer
	Epochs = 200
	# weight_decay = 1e-4
	# class_weights = torch.Tensor([1, 1, 2])
	criterion = nn.MultiLabelSoftMarginLoss()
	# criterion = AsymmetricLoss()
	# parameters = add_weight_decay(model, weight_decay)
	# optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
	optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
	steps_per_epoch = len(train_loader)
	# scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=80, pct_start=0.2)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
		optimizer, T_0=len(train_loader), eta_min=1e-6, T_mult=2
	)

	highest_acc = 0
	trainInfoList = []
	# scaler = GradScaler()
	for epoch in range(Epochs):
		for i, (inputData, target) in enumerate(train_loader):
			inputData = inputData.cuda()
			target = target.cuda()

			optimizer.zero_grad()

			with autocast():  # mixed precision
				output = model(inputData)#.float()  # sigmoid will be done in loss !
			loss = criterion(output, target)

			model.zero_grad()
			loss.backward()
			optimizer.step()
			
			if i % 100 == 0:
				trainInfoList.append([epoch, i, loss.item()])
				print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.5f}'
					  .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
							  scheduler.get_last_lr()[0], loss.item()))

		# try:
		# 	torch.save(model.state_dict(), os.path.join(
		# 		'./models/10/', 'resresnet-rotate-lr003-autocast-{}.ckpt'.format(epoch)))
		# except:
		# 	pass

		model.eval()
		acc_reg = validate_multi(val_loader, model)
		model.train()
		scheduler.step()
		if acc_reg > highest_acc:
			highest_acc = acc_reg
			torch.save(model.state_dict(), os.path.join('./models/8020_1/', 'resnet50.ckpt'))
			print('Saved checkpoint !!!')
		# torch.save(model.state_dict(), os.path.join('./models/8020_42/', 'vgg-rotate-lr003-{}.ckpt'.format(epoch)))
		# print('Saved checkpoint per epoch !!!')
		print('current_Acc = {:.2f}, highest_Acc = {:.2f}\n'.format(acc_reg*100, highest_acc*100))


def validate_multi(val_loader, model):
	print("starting validation")
	sigmoid = torch.nn.Sigmoid()
	preds_reg = []
	targets = []
	for i, (input, target) in enumerate(val_loader):
		target = target
		input = input.cuda()

		with torch.no_grad():
			with autocast():
				output_reg = sigmoid(model(input)) #.cpu()
		# for mAP calculation
		preds_reg.append(torch.where(output_reg > 0.4, 1, 0).cpu().detach())
		targets.append(target.cpu().detach())

	acc, sen, spe = evaluation_cal_multilabel(torch.cat(targets).numpy(), torch.cat(preds_reg).numpy())
	print("Accuracy: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}".format(acc*100, sen*100, spe*100))
	return acc


if __name__ == '__main__':
	seed = 1
	print('Seed: ', seed)
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	main('resnet')
