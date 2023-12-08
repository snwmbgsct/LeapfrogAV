import torch
import torch.nn as nn
from models.layers import MLP,social_transformer,spatial_transformer,st_encoder,VectorMapEncoder



class LEDInitializer(nn.Module):
	def __init__(self, t_h: int=10, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=40):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions. num of agents

		'''
		super(LEDInitializer, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2, 1, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, x, mask=None):
		'''
		x: batch size, t_p, 6
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256
		
		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		# B, 256

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale


class LEDInitializer_SpatialEnc(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=3, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions. num of agents

		'''
		super(LEDInitializer_SpatialEnc, self).__init__()
		self.n = k_pred
		self.d_f = d_f
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		#MDF
		self._lane_len = 50
		self._lane_feature = 7
		self._crosswalk_len = 30
		self._crosswalk_feature = 3

		self.social_encoder = social_transformer(t_h*d_h)
		self.spatial_encoder = spatial_transformer(256)

		#MDF
		self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len) #?dim: 7, 50
		self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len) #?dim: 30, 3

		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(256*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*3, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*3, 1, hid_feat=(256, 128), activation=nn.ReLU())

	
	def forward(self, data, x, mask=None):
		'''
		x: batch size, t_p, 6
		'''
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask) # social_embed: torch.Size([44, 1, 256])  x.shape:torch.Size([44, 10, 6]) mask:torch.Size([44, 44])
		social_embed = social_embed.squeeze(1)
		# B, 256

		# vector maps #MDF
		map_lanes = data['map_lanes'].cuda()
		map_crosswalks = data['map_crosswalks'].cuda()

		# map encoding #MDF
		encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes) # torch.Size([4, 15, 256])
		encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks) #dim: torch.Size([4, 15, 256])
		map_info = torch.cat([encoded_map_lanes, encoded_map_crosswalks], dim=1) #torch.Size([4, 215, 256])
		map_mask = torch.cat([lanes_mask, crosswalks_mask], dim=1)
		spatial_embed = self.spatial_encoder(map_info, map_mask) # torch.Size([4, 215, 256])


		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		# B, 256

		mean_total = torch.cat((ego_mean_embed, social_embed, spatial_embed), dim=-1) #dim: torch.Size([44, 256]), torch.Size([44, 256]), torch.Size([44, 215, 256])
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, self.d_f)

		scale_total = torch.cat((ego_scale_embed, social_embed, spatial_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, self.d_f)

		return guess_var, guess_mean, guess_scale