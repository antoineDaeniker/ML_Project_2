import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import constants



class FocalLoss(nn.Module):
	def __init__(self, alpha=0.005, gamma=2):
		super(FocalLoss, self).__init__()
		# alpha is proportion of positive instances
		# gamma is relaxation parameter
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets):
		BCE_loss = F.cross_entropy(inputs, targets.squeeze(1), reduce=None)
		p_t = torch.exp(-BCE_loss)

		# if target = 1, use (1 - alpha), otherwise alpha 
		alpha_tensor = (1 - self.alpha) * targets + self.alpha * (1 - targets)
		f_loss = alpha_tensor * (1 - p_t) ** self.gamma * BCE_loss
		return f_loss.mean()



class SE_Block(nn.Module):
    def __init__(self, in_channels, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        weight = self.squeeze(x).view(batch_size, channels)
        weight = self.excitation(weight).view(batch_size, channels, 1, 1)
        return x * weight.expand(x.shape)



class Block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(out_channels),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(out_channels)
		)

	def forward(self, x):
		"""(convolution => [BN] => ReLU) * 2"""
		return self.double_conv(x)
    


class Encoder(nn.Module):
	def __init__(self, channels=(3, 16, 32, 64), attention=True):
		super().__init__()
		self.enc_blocks = nn.ModuleList([Block(channels[i], channels[i + 1])
									     for i in range(len(channels) - 1)])
		if(attention):
			self.attention_blocks = nn.ModuleList([SE_Block(channels[i + 1])
										     	   for i in range(len(channels) - 1)])
		else:
			self.attention_blocks = nn.ModuleList([nn.Identity()
										     	   for i in range(len(channels) - 1)])

		self.pool = nn.MaxPool2d(kernel_size=2)

	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		block_outputs = []

		# loop through the encoder blocks
		for block, attention_block in zip(self.enc_blocks, self.attention_blocks):
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			x = attention_block(x)
			block_outputs.append(x)
			x = self.pool(x)

		# return the list containing the intermediate outputs
		return block_outputs



class Decoder(nn.Module):
	def __init__(self, channels=(64, 32, 16), attention=True, dropout_p=0.1):
		super().__init__()
		self.channels = channels
		self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
								      for i in range(len(channels) - 1)])
		self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i + 1])
									  for i in range(len(channels) - 1)])
		if(attention):
			self.attention_blocks = nn.ModuleList([SE_Block(channels[i + 1])
										     	   for i in range(len(channels) - 1)])
		else:
			self.attention_blocks = nn.ModuleList([nn.Identity()
										     	   for i in range(len(channels) - 1)])
		self.dropout = nn.Dropout(p=dropout_p)

	def forward(self, x, enc_features):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)

			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			enc_feature = self.crop(enc_features[i], x)
			x = torch.cat([x, enc_feature], dim=1)
			x = self.dropout(x)
			x = self.dec_blocks[i](x)
			x = self.attention_blocks[i](x)

		# return the final decoder output
		return x

	def crop(self, enc_features, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		enc_features = transforms.CenterCrop([H, W])(enc_features)
		
		# return the cropped features
		return enc_features



class UNet(nn.Module):
	def __init__(self, config, gamma=2):
		super().__init__()
		self.encoder = Encoder(config["encoder_channels"], config["attention"])
		self.decoder = Decoder(config["decoder_channels"], config["attention"], config["dropout"])

		self.head = nn.Conv2d(config["decoder_channels"][-1], constants.NUM_CLASSES, kernel_size=1)
		self.retain_dim = config["retain_dim"]
		self.out_size = (constants.INPUT_IMAGE_WIDTH, constants.INPUT_IMAGE_HEIGHT)

		self.loss_function = FocalLoss(gamma=gamma)

	def forward(self, x):
		# grab the features from the encoder
		enc_features = self.encoder(x)

		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		dec_features = self.decoder(enc_features[::-1][0],
									enc_features[::-1][1:])

		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		pred_segmentation_mask = self.head(dec_features)

		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retain_dim:
			pred_segmentation_mask = F.interpolate(pred_segmentation_mask, self.out_size)
			
		# return the segmentation map
		return pred_segmentation_mask

	def loss(self, pred, label):
		return self.loss_function(pred, label)