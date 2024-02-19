import torch.nn as nn
import torch.nn.functional as F

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x) 

		return x

class DepthWiseSepFCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(DepthWiseSepFCDiscriminator, self).__init__()

		self.conv1_d = nn.Conv2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1,groups=num_classes)
		self.conv1_p = nn.Conv2d(num_classes, ndf, kernel_size=1, padding=1)

		self.conv2_d = nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1,groups=ndf)
		self.conv2_p = nn.Conv2d(ndf, ndf*2, kernel_size=1, padding=1)

		self.conv3_d = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=2, padding=1,groups=ndf*2)
		self.conv3_p = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, padding=1)
		
		self.conv4_d = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1,groups=ndf*4)
		self.conv4_p = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, padding=1)

		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1_d(x)
		x = self.leaky_relu(x)
		x=self.conv1_p(x)
		x = self.leaky_relu(x)

		x = self.conv2_d(x)
		x = self.leaky_relu(x)
		x=self.conv2_p(x)
		x = self.leaky_relu(x)

		x = self.conv3_d(x)
		x = self.leaky_relu(x)
		x=self.conv3_p(x)
		x = self.leaky_relu(x)

		x = self.conv4_d(x)
		x = self.leaky_relu(x)
		x=self.conv4_p(x)
		x = self.leaky_relu(x)
		
		x = self.classifier(x)
		return x

class DepthWiseSepBNFCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(DepthWiseSepBNFCDiscriminator, self).__init__()

		self.conv1_d = nn.Conv2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1,groups=num_classes)
		self.bn1_d=nn.BatchNorm2d(num_classes)
		self.conv1_p = nn.Conv2d(num_classes, ndf, kernel_size=1, padding=1)
		self.bn1_p=nn.BatchNorm2d(ndf)

		self.conv2_d = nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1,groups=ndf)
		self.bn2_d=nn.BatchNorm2d(ndf)
		self.conv2_p = nn.Conv2d(ndf, ndf*2, kernel_size=1, padding=1)
		self.bn2_p=nn.BatchNorm2d(ndf*2)

		self.conv3_d = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=2, padding=1,groups=ndf*2)
		self.bn3_d=nn.BatchNorm2d(ndf*2)
		self.conv3_p = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, padding=1)
		self.bn3_p=nn.BatchNorm2d(ndf*4)
		
		self.conv4_d = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1,groups=ndf*4)
		self.bn4_d=nn.BatchNorm2d(ndf*4)
		self.conv4_p = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, padding=1)
		self.bn4_p=nn.BatchNorm2d(ndf*8)

		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1_d(x)
		x=self.bn1_d(x)
		x = self.leaky_relu(x)
		x=self.conv1_p(x)
		x=self.bn1_p(x)
		x = self.leaky_relu(x)

		x = self.conv2_d(x)
		x=self.bn2_d(x)
		x = self.leaky_relu(x)
		x=self.conv2_p(x)
		x=self.bn2_p(x)
		x = self.leaky_relu(x)

		x = self.conv3_d(x)
		x=self.bn3_d(x)
		x = self.leaky_relu(x)
		x=self.conv3_p(x)
		x=self.bn3_p(x)
		x = self.leaky_relu(x)

		x = self.conv4_d(x)
		x=self.bn4_d(x)
		x = self.leaky_relu(x)
		x=self.conv4_p(x)
		x=self.bn4_p(x)
		x = self.leaky_relu(x)
		
		x = self.classifier(x)
		return x
