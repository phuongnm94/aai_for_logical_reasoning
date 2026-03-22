import argparse
import glob
import logging
import sys, inspect
import time
from datetime import datetime, timedelta
from typing import Optional
import json
import os
import pandas as pd
import torch
from torch import nn 
from lightning import seed_everything
import gc

from transformers.masking_utils import sdpa_mask, eager_mask, causal_mask_function, AttentionMaskInterface
from transformers.modeling_utils import AttentionInterface
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.integrations.eager_paged import eager_paged_attention_forward
from transformers.integrations.flex_attention import repeat_kv

from custom_attn import generate_focusing_rule_masked_positions, \
	generate_constraint_rule_masked_positions, generate_mixed_focusing_and_constraint_masked_attn, no_masked_attn, \
	generate_focusing_rule_inc_attn_masked_positions, generate_constraint_focal_rule_masked_positions, \
	dump_attn_viz
from evaluate import evaluate_json, evaluate_simple_lg, evaluate_simple_lg_gsm8k, evaluate_simple_lg_logicdeduction, evaluate_simple_lg_mask_attn
from infer_llm_base import llm_infer
import pandas as pd 

logger = logging.getLogger(__name__)

first_print = True 
global_infor = {}


def shift_left(matrix, step=1):
	result = torch.zeros_like(matrix).fill_(False)
	result[:, :-step] = matrix[:, step:]
	return result
def shift_right(matrix, step=1):
	result = torch.zeros_like(matrix).fill_(False)
	result[:, step:] = matrix[:, :-step]
	return result
def shift_top(matrix, step=1):
	result = torch.zeros_like(matrix).fill_(False)
	result[:-step, :] = matrix[step:, :]
	return result
def shift_down(matrix, step=1):
	result = torch.zeros_like(matrix).fill_(False)
	result[step:, :] = matrix[:-step, :]
	return result

def value_to_quantile(df_vals, value):
	q_value = (df_vals <= value).sum() / len(df_vals)
	return q_value

def compute_attn_rate_pattern(layer_index, attn_w):
	global g_args
	all_data = {'head_idx':[], 'diagonal_rate': [],'vertical_rate': [],'horizontal_rate': [], 'strong_attn_rate': [], 'center_attn_rate': []}
	for head_idx in range( attn_w[layer_index].shape[1]):
		# print(masked)
		x =   attn_w[layer_index][0,head_idx] 
		# masked = x >  0.04
		if g_args.strong_att_const is None:
			threshold_value = pd.DataFrame(x.flatten()).quantile(g_args.strong_att_percentile)[0].item()
		else:
			threshold_value = g_args.strong_att_const
		masked = x > threshold_value

		total_point = torch.sum (masked)
		diagonal_rate =  torch.sum (masked & shift_top(shift_left(masked))) / total_point
		vertical_rate = torch.sum(masked &  (shift_top(masked)))/ total_point
		horizontal_rate =   torch.sum (masked &  (shift_left(masked))) / total_point 
		all_data['diagonal_rate'].append(diagonal_rate.item())
		all_data['vertical_rate'].append(vertical_rate.item())
		all_data['horizontal_rate'].append(horizontal_rate.item())
		all_data['head_idx'].append(head_idx)
		total_pair = masked.shape[0]*(masked.shape[0]+1) / 2 
		all_data['strong_attn_rate'].append(total_point / total_pair)
		
		count_not_center = 0 
		for i in range(masked.shape[0]):
			if masked[i, i] or masked[0, i]:
				count_not_center += 1
		all_data['center_attn_rate'].append((total_point- count_not_center)/ total_pair)

	df = pd.DataFrame(all_data)
	return df  

def filter_high_diagonal_attention(attn_w):
	head_coordinates = []
	head_count = len(attn_w)* attn_w[0].shape[1]
	for layer_index in range(len(attn_w)):
		df_check = compute_attn_rate_pattern(layer_index, attn_w)
		percentile_heads = value_to_quantile(df_check["diagonal_rate"], g_args.diagonal_rate_threshold)
		print(f"Layer {layer_index}: Percentile(> {g_args.diagonal_rate_threshold}) = {percentile_heads} => {1-percentile_heads} % total heads" )
		head_indexes = df_check[df_check["diagonal_rate"] > g_args.diagonal_rate_threshold]['head_idx'].to_list()
		for h_idx in head_indexes:
			head_coordinates.append((layer_index, h_idx))

	print(f"Total reweighting head {len(head_coordinates)}/{head_count} = {len(head_coordinates)/head_count} % total heads" )
	return set(head_coordinates)

def filter_diagonal_att_percentile(attn_w):
	head_coordinates = []

	for layer_index in range(len(attn_w)):
		df_check = compute_attn_rate_pattern(layer_index, attn_w)
		if g_args.diagonal_att_percentile_max_val is None:
			threshold_value = pd.DataFrame(df_check["diagonal_rate"]).quantile(g_args.diagonal_att_percentile).iloc[0].item()
			# print(value_to_quantile(df_check["diagonal_rate"], 0.3))
			head_indexes = df_check[df_check["diagonal_rate"] > threshold_value]['head_idx'].to_list()
		else:
			df_diagonal_rate = pd.DataFrame(df_check["diagonal_rate"])
			threshold_min_value = df_diagonal_rate.quantile(g_args.diagonal_att_percentile).iloc[0].item()
			threshold_max_value = df_diagonal_rate.quantile(g_args.diagonal_att_percentile_max_val).iloc[0].item()
			head_indexes = df_check[(df_check["diagonal_rate"] > threshold_min_value) & (df_check["diagonal_rate"] < threshold_max_value)] ['head_idx'].to_list()

		for h_idx in head_indexes:
			head_coordinates.append((layer_index, h_idx))

	return set(head_coordinates)

def filter_high_verticle_low_others_attention(attn_w, threshold=0.6):
	head_coordinates = []
	head_count = len(attn_w)* attn_w[0].shape[1]
	for layer_index in range(len(attn_w)):
		df_check = compute_attn_rate_pattern(layer_index, attn_w)
		head_indexes = df_check[(df_check["vertical_rate"] > threshold) & (df_check["horizontal_rate"] < 0.3) & (df_check["diagonal_rate"] < 0.3)]['head_idx'].to_list()
		print(f"Layer {layer_index}:  {len(head_indexes) / attn_w[0].shape[1]} % total heads" )
		for h_idx in head_indexes:
			head_coordinates.append((layer_index, h_idx))

	print(f"Total reweighting head {len(head_coordinates)}/{head_count} = {len(head_coordinates)/head_count} % total heads" )
	return set(head_coordinates)

def filter_high_center_attention(attn_w, threshold=0.6):
	head_coordinates = []
	for layer_index in range(len(attn_w)):
		df_check = compute_attn_rate_pattern(layer_index, attn_w)
		head_indexes = df_check[(df_check["horizontal_rate"] > 0.2) & (df_check["diagonal_rate"] > 0.2) & \
						  (df_check["center_attn_rate"] /df_check["strong_attn_rate"]  > 0.8)]['head_idx'].to_list()
		for h_idx in head_indexes:
			head_coordinates.append((layer_index, h_idx))

	return set(head_coordinates)


def filter_gather_information_attention(attn_w, threshold=0.6):
	head_coordinates = []
	for layer_index in range(len(attn_w)):
		df_check = compute_attn_rate_pattern(layer_index, attn_w)
		head_indexes = df_check[(df_check["horizontal_rate"] > 0.2) & (df_check["diagonal_rate"] > 0.2)]['head_idx'].to_list()
		for h_idx in head_indexes:
			head_coordinates.append((layer_index, h_idx))

	return set(head_coordinates)

def my_new_sdpa(
			module,
			query_states,
			key_states,
			value_states,
			attention_mask_,
			*args,
			**kwargs
			):

	if g_args.dump_attn_viz:
		sdpa_func = eager_paged_attention_forward
	else:
		sdpa_func = sdpa_attention_forward

	attn_output, x = sdpa_func(module,
			query_states,
			key_states,
			value_states,
			attention_mask=attention_mask_, # get_attn_mask(query_states),
			*args,
			**kwargs)
	if g_args.dump_attn_viz:
		global global_infor
		if 'attn_score' not in global_infor:
			global_infor['attn_score'] = x
	return attn_output, x 
	
 
def my_new_sdpa_true_false_attn_mask(
		batch_size: int,  # required arg
			cache_position: torch.Tensor,  # required arg
			kv_length: int,  # required arg
			kv_offset: int = 0,  # required arg
			mask_function = causal_mask_function,  # required arg
			attention_mask: Optional[torch.Tensor] = None,  # required arg
	*args, **kwargs):

	spda_mask_mt = sdpa_mask(batch_size,  cache_position,  kv_length,   
			kv_offset,  mask_function,  attention_mask,  *args, **kwargs)
	
	# [[ by phuongnm ]]
	global final_masked_infor
	delete_value = False
	if spda_mask_mt.shape[2]!= 1:
		info_seq_lens = torch.sum(attention_mask, dim=1)
		for i_batch in range(batch_size):

			off_set = kv_length - info_seq_lens[i_batch] 
			# spda_mask_mt[i_batch, :, 0:off_set,0:off_set]= delete_value # left padding positions

			for mask_point in final_masked_infor:
				spda_mask_mt[i_batch, :,mask_point[0][0]+off_set:mask_point[0][1]+off_set,
								mask_point[1][0]+off_set:mask_point[1][1]+off_set]= delete_value #  attention_mask[0][0][0][5]
			
			global first_print
			if first_print:
				print(f"masked = {len(final_masked_infor)} spans")
				first_print = False
	# ........

	return spda_mask_mt
 
def my_new_sdpa_value_mask(
		batch_size: int,  # required arg
			cache_position: torch.Tensor,  # required arg
			kv_length: int,  # required arg
			kv_offset: int = 0,  # required arg
			mask_function = causal_mask_function,  # required arg
			attention_mask: Optional[torch.Tensor] = None,  # required arg
	*args, **kwargs):

	spda_mask_mt = sdpa_mask(batch_size,  cache_position,  kv_length,   
			kv_offset,  mask_function,  attention_mask,  *args, **kwargs)
	
	# [[ by phuongnm ]]
	global final_masked_infor
	global final_masked_inc_attn
	global inc_attn_score
	info_seq_lens = torch.sum(attention_mask, dim=1)
	delete_value = float("-inf")
	if spda_mask_mt.shape[2]!= 1:
		spda_mask_mt_values = (~spda_mask_mt).bfloat16()
		spda_mask_mt_values.masked_fill_(~spda_mask_mt, delete_value)
		for i_batch in range(batch_size):

			off_set = kv_length - info_seq_lens[i_batch] 
			# spda_mask_mt[i_batch, :, 0:off_set,0:off_set]= delete_value # left padding positions

			for mask_point in final_masked_infor:
				spda_mask_mt_values[i_batch, :,mask_point[0][0]+off_set:mask_point[0][1]+off_set,
								mask_point[1][0]+off_set:mask_point[1][1]+off_set]= delete_value #  attention_mask[0][0][0][5]
			
			for mask_point in final_masked_inc_attn:
				spda_mask_mt_values[i_batch, :,mask_point[0][0]+off_set:mask_point[0][1]+off_set,
								mask_point[1][0]+off_set:mask_point[1][1]+off_set]= inc_attn_score #  attention_mask[0][0][0][5]
			
			global first_print
			if first_print:
				print(f"maksked = {len(final_masked_infor)} spans: final_masked_infor")
				print(f"maksked = {len(final_masked_inc_attn)} spans: final_masked_inc_attn")
				first_print = False
		return spda_mask_mt_values
	# ........

	return spda_mask_mt

def __customize_attention_mask(query_states, key_states, attention_mask_, **kwargs):
	# [[ by phuongnm ]]
	global final_masked_infor
	global final_masked_inc_attn
	global inc_attn_score
	global global_infor
	delete_value = float("-inf")
	batch_size = query_states.shape[0]
	if attention_mask_.shape[2]!= 1:
		layer_index = kwargs['layer_index']
		num_attention_heads = kwargs['num_attention_heads']

		info_seq_lens = torch.sum(attention_mask_[:, 0, -1, :]==0, dim=1)

		attention_mask_ = attention_mask_.repeat(1, num_attention_heads, 1, 1)   
		# spda_mask_mt_values.masked_fill_(~spda_mask_mt, delete_value)

		if f"__customize_attention_mask_l{layer_index}" in global_infor:
			# reload attention from cached 
			customized_mask = global_infor[f"__customize_attention_mask_l{layer_index}"].to(attention_mask_.device)
			for i_batch in range(batch_size):
				offset_paired_with_sample = customized_mask.shape[1] - info_seq_lens[i_batch] 
				off_set_padding = key_states.shape[2] - info_seq_lens[i_batch] 
				if offset_paired_with_sample == 0:
					attention_mask_[i_batch,:, off_set_padding:,
					 							off_set_padding: ] = customized_mask
				elif offset_paired_with_sample > 0:
					attention_mask_[i_batch,:, off_set_padding:,
					 							off_set_padding: ] = customized_mask[:, :info_seq_lens[i_batch],
																						:info_seq_lens[i_batch]]
				elif offset_paired_with_sample < 0:
					attention_mask_[i_batch,:, off_set_padding:customized_mask.shape[1]+off_set_padding,
					 							off_set_padding: customized_mask.shape[1]+off_set_padding] = customized_mask
			del customized_mask
			return attention_mask_
		
		# get patterned heads
		filter_heads = set()
		if g_args.filter_head_attn_pattern is not None:
			filter_heads = filter_heads.union(global_infor.get(g_args.filter_head_attn_pattern, set()))

		for head_idx in range(num_attention_heads):
			head_value = g_args.coef_inc_attn_score * global_infor['median_scaled_dot_values'][layer_index][head_idx] + g_args.bias_inc_attn_score
			
			if g_args.filter_head_attn_pattern is not None:
				# skip this head
				if (layer_index, head_idx) not in filter_heads:
					continue

			for i_batch in range(batch_size):

				off_set = key_states.shape[2] - info_seq_lens[i_batch] 
				# spda_mask_mt[i_batch, :, 0:off_set,0:off_set]= delete_value # left padding positions
	
				for mask_point in final_masked_infor:
					attention_mask_[i_batch, head_idx,mask_point[0][0]+off_set:mask_point[0][1]+off_set,
									mask_point[1][0]+off_set:mask_point[1][1]+off_set]= delete_value #  attention_mask[0][0][0][5]
				
				for mask_point in final_masked_inc_attn:
					attention_mask_[i_batch, head_idx, mask_point[0][0]+off_set:mask_point[0][1]+off_set,
									mask_point[1][0]+off_set:mask_point[1][1]+off_set]=  head_value # inc_attn_score #  attention_mask[0][0][0][5]
				
				global first_print
				if first_print:
					print(f"maksked = {len(final_masked_infor)} spans: final_masked_infor")
					print(f"maksked = {len(final_masked_inc_attn)} spans: final_masked_inc_attn")
					first_print = False

		if not g_args.no_cache_attention_mask and f"__customize_attention_mask_l{layer_index}" not in global_infor:
			# cache value of longest samples (no padding)
			i_batch_cache = [idx for idx, e in enumerate(info_seq_lens) if e == torch.max(info_seq_lens)][0]
			global_infor[f"__customize_attention_mask_l{layer_index}"] = attention_mask_[i_batch_cache].cpu()
	# ........

	return attention_mask_

def adaptive_sdpa(
			module,
			query_states,
			key_states,
			value_states,
			attention_mask_,
			*args,
			**kwargs
			):

	if g_args.dump_attn_viz:
		sdpa_func = eager_paged_attention_forward
	else:
		sdpa_func = sdpa_attention_forward


	module_name_infor = global_infor['module_to_name'].get(module, None) 
	all_module_infor = module_name_infor.split(".")
	layer_index = int(all_module_infor[all_module_infor.index("layers")+1])
	num_attention_heads = global_infor['model_config'].num_attention_heads
	attention_mask_ = __customize_attention_mask(query_states, key_states, attention_mask_, 
											  layer_index=layer_index, num_attention_heads=num_attention_heads, **kwargs)

	attn_output, x = sdpa_func(module,
			query_states,
			key_states,
			value_states,
			attention_mask=attention_mask_, # get_attn_mask(query_states),
			*args,
			**kwargs)
	if g_args.dump_attn_viz:
		if 'attn_score' not in global_infor:
			global_infor['attn_score'] = x
	return attn_output, x 
	
 
def adaptive_sdpa_mask(
		batch_size: int,  # required arg
			cache_position: torch.Tensor,  # required arg
			kv_length: int,  # required arg
			kv_offset: int = 0,  # required arg
			mask_function = causal_mask_function,  # required arg
			attention_mask: Optional[torch.Tensor] = None,  # required arg
	*args, **kwargs):

	spda_mask_mt = sdpa_mask(batch_size,  cache_position,  kv_length,   
			kv_offset,  mask_function,  attention_mask,  *args, **kwargs)
	
	# [[ by phuongnm ]]
	delete_value = float("-inf")
	if spda_mask_mt.shape[2]!= 1:
		spda_mask_mt_values = (~spda_mask_mt).bfloat16()
		spda_mask_mt_values.masked_fill_(~spda_mask_mt, delete_value)
		return spda_mask_mt_values
	# ........

	return spda_mask_mt

def monitor_eager(
		module: nn.Module,
		query: torch.Tensor,
		key: torch.Tensor,
		value: torch.Tensor,
		attention_mask: Optional[torch.Tensor],
		scaling: float,
		dropout: float = 0.0,
		**kwargs,
	):
	cache = kwargs.pop("cache", None)
	if cache is not None:
		key, value = cache.update(key, value, module.layer_idx, **kwargs)

	key_states = repeat_kv(key, module.num_key_value_groups)
	value_states = repeat_kv(value, module.num_key_value_groups)

	attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
	
	# phuongnm : log values 
	flatten_attn_weights = attn_weights.squeeze().reshape(attn_weights.shape[1], -1)
	global_infor['max_scaled_dot_values'].append(torch.max(flatten_attn_weights, dim=1)[0])
	global_infor['min_scaled_dot_values'].append(torch.min(flatten_attn_weights, dim=1)[0])
	global_infor['avg_scaled_dot_values'].append(torch.mean(flatten_attn_weights, dim=1))
	global_infor['median_scaled_dot_values'].append(torch.median(flatten_attn_weights, dim=1)[0])
	#

	if attention_mask is not None:
		causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
		attn_weights = attn_weights + causal_mask

	attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
	attn_output = torch.matmul(attn_weights, value_states)
	attn_output = attn_output.transpose(1, 2).contiguous()

	# phuongnm : log values 
	global_infor['attn_weights'].append(attn_weights)
	#

	return attn_output, attn_weights

	
 
def monitor_eager_mask(
		batch_size: int,  # required arg
			cache_position: torch.Tensor,  # required arg
			kv_length: int,  # required arg
			kv_offset: int = 0,  # required arg
			mask_function = causal_mask_function,  # required arg
			attention_mask: Optional[torch.Tensor] = None,  # required arg
	*args, **kwargs):
 
	eager_mask_mt = eager_mask(batch_size,  cache_position,  kv_length,   
			kv_offset,  mask_function,  attention_mask,  *args, **kwargs)
	return eager_mask_mt

def set_up_attention_mask(args):
	if args.logical_masked_func=="no_masked_attn":
		# default setting 
		AttentionInterface.register("my_new_sdpa", sdpa_attention_forward)
		AttentionMaskInterface.register("my_new_sdpa", sdpa_mask)
	else:
		if args.apply_dynamic_attn_pattern:
			AttentionInterface.register("my_new_sdpa", adaptive_sdpa)
			AttentionMaskInterface.register("my_new_sdpa", adaptive_sdpa_mask)

			AttentionInterface.register("monitor_eager", monitor_eager)
			AttentionMaskInterface.register("monitor_eager", monitor_eager_mask)
		else:
			AttentionInterface.register("my_new_sdpa", my_new_sdpa)

			if args.use_true_false_mask:
				print(f"!!!![Warning] using  my_new_sdpa_true_false_attn_mask with masked True/False")
				AttentionMaskInterface.register("my_new_sdpa", my_new_sdpa_true_false_attn_mask)
			else:
				print(f"!!!![Warning] using  my_new_sdpa_value_mask with constant values = {args.inc_attn_score}")
				AttentionMaskInterface.register("my_new_sdpa", my_new_sdpa_value_mask)


	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process ...')
	parser.add_argument('--prompting_data_path',type=str, help='prompting_data_path', default='data/ProofWriter/test.logiccotkb_prompting.json')
	parser.add_argument('--model_name',type=str, help='model_name', default="Qwen/Qwen3-14B") 
	parser.add_argument('--max_new_tokens',type=int, help='max_new_tokens', default=500) 
	parser.add_argument('--batch_size',type=int, help='batch_size', default=4) 
	parser.add_argument('--eos_tok_str',type=str, help='eos_tok_str', default="-------") 
	parser.add_argument('--use_8bit_quantization',action="store_true", help='use_8bit_quantization', default=False)
	parser.add_argument('--logical_masked_func',type=str, help='logical_masked_func', default=generate_focusing_rule_masked_positions.__name__) 
	parser.add_argument('--override_llm_class',action="store_true", help='override_llm_class', default=False) 
	parser.add_argument('--inc_attn_score',type=float, help='inc_attn_score', default=None) 
	parser.add_argument('--coef_inc_attn_score',type=float, help='coef_inc_attn_score', default=1) 
	parser.add_argument('--bias_inc_attn_score',type=float, help='bias_inc_attn_score', default=0) 
	parser.add_argument('--use_true_false_mask',action="store_true", help='override_llm_class', default=False) 
	parser.add_argument('--output_id',type=str, help='output_id', default="result") 
	parser.add_argument('--dump_attn_viz', action="store_true", help='dump_attn_score', default=False)
	parser.add_argument('--apply_dynamic_attn_pattern', action="store_true", help='apply_dynamic_attn_pattern', default=False)
	parser.add_argument('--no_cache_attention_mask', action="store_true", help='use_cache_attention_mask', default=False)
	parser.add_argument('--strong_att_const', type=float, help='strong_att_percentile', default=None)
	parser.add_argument('--strong_att_percentile', type=float, help='strong_att_percentile', default=0.95)
	parser.add_argument('--diagonal_att_percentile', type=float, help='diagonal_att_percentile', default=0.8)
	parser.add_argument('--diagonal_att_percentile_max_val', type=float, help='diagonal_att_percentile_max_val', default=None)
	parser.add_argument('--diagonal_rate_threshold', type=float, help='diagonal_att_percentile', default=0.3)
	parser.add_argument('--scan_results', nargs='+',  help='scan path pattern of results', default=None)
	parser.add_argument('--dump_selected_head', action="store_true", help='dump_selected_head', default=False) 

	parser.add_argument('--filter_head_attn_pattern', type=str,
					 help='filter head in func: [filter_high_diagonal_attention, filter_high_verticle_low_others_attention]', 
					 default=None)
	parser.add_argument('--get_emb_vector', action="store_true", help='filter_head', default=False)
	seed_everything(42) 

	args, unknown = parser.parse_known_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if "\\n" in args.eos_tok_str:
		args.eos_tok_str = args.eos_tok_str.replace("\\n", "\n")
	if args.filter_head_attn_pattern is not None or args.apply_dynamic_attn_pattern: 
		args.override_llm_class = True
	print(args)

	# ========================================
	# scan all the results 
	if args.scan_results is not None:
		path_file_out = f"{'/'.join(args.scan_results[0].split('/')[:-1])}/all_result.csv"
		def_values = dict([(k, v) for k, v in args.__dict__.items()])
		all_results = []
		for pattern_path in args.scan_results:
			for r_file in glob.glob(pattern_path):
				r_performance= {}
				all_infor = json.load(open(r_file))
				if isinstance(all_infor, dict) and "config" in all_infor and 'result' in all_infor:
					r_configs = all_infor['config']
					r_performance['time'] = datetime.fromtimestamp(int(os.path.getmtime(r_file))).strftime("%Y/%m/%d, %H:%M:%S")
					r_performance.update(all_infor['result'])
					r_performance.update(def_values)
					r_performance.update(r_configs)
					r_performance["acc"] = r_performance['acc']*100
					r_performance["strong_att_percentile"] = r_performance['strong_att_percentile']*100
					r_performance['path_file'] = r_file
					all_results.append(r_performance)
				else:
					continue

		df_result = pd.DataFrame(all_results)
		print(f"- Write file: {path_file_out}")
		df_result.to_csv(path_file_out, sep=',')
		exit(0) 

	# ========================================
	# check configs 
	global inc_attn_score
	inc_attn_score = args.inc_attn_score

	global g_args
	g_args = args
	set_up_attention_mask(args)


	model_name =  args.model_name #  "Qwen/Qwen3-14B"   # "Qwen/Qwen3-8B"  #  # 'meta-llama/Llama-3.1-8B-Instruct'  'meta-llama/Meta-Llama-3-8B-Instruct'  
	logical_masked_func = {
		generate_focusing_rule_masked_positions.__name__: generate_focusing_rule_masked_positions,
		generate_constraint_rule_masked_positions.__name__: generate_constraint_rule_masked_positions,
		generate_mixed_focusing_and_constraint_masked_attn.__name__: generate_mixed_focusing_and_constraint_masked_attn,
		generate_focusing_rule_inc_attn_masked_positions.__name__: generate_focusing_rule_inc_attn_masked_positions,
		generate_constraint_focal_rule_masked_positions.__name__: generate_constraint_focal_rule_masked_positions,
		filter_diagonal_att_percentile.__name__: filter_diagonal_att_percentile,
		no_masked_attn.__name__: no_masked_attn
	}[args.logical_masked_func]
	
	# ========================================
	# Load data and exit if the result exist!!
	prompting_data_path = args.prompting_data_path
	path_file_result = prompting_data_path.replace(
		".json", f'.{model_name.split("/")[-1]}.ntok-{args.max_new_tokens}.{logical_masked_func.__name__}.quant8bit-{args.use_8bit_quantization}'+\
		f'.AA-{args.apply_dynamic_attn_pattern}.H-{args.filter_head_attn_pattern}.cache-{not args.no_cache_attention_mask}'+\
		f'.percent-{args.strong_att_percentile}.{args.output_id}.diagon-{g_args.diagonal_att_percentile}-{g_args.diagonal_att_percentile_max_val}.json')
	if args.strong_att_const is not None:
		path_file_result = path_file_result.replace(f"percent-{args.strong_att_percentile}.", f"percent-cst-{args.strong_att_const}.")
	
	if args.filter_head_attn_pattern == filter_high_diagonal_attention.__name__:
		path_file_result = path_file_result.replace(f".diagon-{g_args.diagonal_att_percentile}-{g_args.diagonal_att_percentile_max_val}.", f".diagon-rate-c-{args.diagonal_rate_threshold}.")
	if args.coef_inc_attn_score != 1:
		path_file_result = path_file_result.replace(f".json", f".coefA-{args.coef_inc_attn_score}.json")
	if args.bias_inc_attn_score != 0:
		path_file_result = path_file_result.replace(f".json", f".biasA-{args.bias_inc_attn_score}.json")
	
	# shorten path name 
	path_file_result = path_file_result.replace(f".logiccotkb_prompting.", f".").replace(f".ntok-2000", f"") \
			.replace(f".generate_focusing_rule_inc_attn_masked_positions", f".AAI")

	print("path_file_result = ", path_file_result)
	eval_func = evaluate_simple_lg_logicdeduction if ("logicdeduction" in path_file_result) else \
            evaluate_simple_lg_gsm8k if ("gsm8k" in path_file_result.lower()) else \
            evaluate_simple_lg_mask_attn if ( 'AAI' in path_file_result ) else \
            evaluate_simple_lg if 'logiccot' in path_file_result else \
            evaluate_simple_lg_mask_attn
	
	if os.path.exists(path_file_result) or os.path.exists(path_file_result.replace("-None.json", ".json")):
		print(f"File {path_file_result} already exists, skip inference")
		eval_data, acc = eval_func(path_file_result)
		exit(0) 
		
	
	raw_data = json.load(open(prompting_data_path))
	all_prompt_content= [e['prompting'] for e in raw_data]
	print(all_prompt_content[0])
	
	if args.apply_dynamic_attn_pattern:
		# ========================================
		# RUN compute attention analysis for 
		# hard fewshot samples 

		global_infor['max_scaled_dot_values'] = []
		global_infor['min_scaled_dot_values'] = []
		global_infor['avg_scaled_dot_values'] = []
		global_infor['median_scaled_dot_values'] = []
		global_infor['attn_weights'] = []
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		hard_fewshot_prompt = "-------".join(all_prompt_content[0].split("-------")[:-1])
		hard_sample_inputs = tokenizer(hard_fewshot_prompt, return_tensors="pt")

		# Run forward with attention outputs
		with torch.no_grad():
			model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="monitor_eager")
			outputs = model(**hard_sample_inputs)
		
		print(len(global_infor['attn_weights']),"layers, ", global_infor['attn_weights'][0].shape)  
		# Run group attention weight 
		# torch.save(global_infor['attn_weights'], open('attn_w', 'wb'))
		# print("saved")

		del model
		del tokenizer
		torch.cuda.empty_cache()
		gc.collect()

	if args.filter_head_attn_pattern is not None:
		# ========================================
		# RUN filter heads based on 
		# attention pattern 
		map_filter_head_attn_pattern ={
			filter_high_diagonal_attention.__name__: filter_high_diagonal_attention,
			filter_high_verticle_low_others_attention.__name__: filter_high_verticle_low_others_attention,
			filter_gather_information_attention.__name__: filter_gather_information_attention,
			filter_high_center_attention.__name__: filter_high_center_attention,
			filter_diagonal_att_percentile.__name__: filter_diagonal_att_percentile
		}
		global_infor[g_args.filter_head_attn_pattern] = map_filter_head_attn_pattern[args.filter_head_attn_pattern](attn_w=global_infor['attn_weights'])
		print(len(global_infor[g_args.filter_head_attn_pattern]), g_args.filter_head_attn_pattern, list(global_infor[g_args.filter_head_attn_pattern])[:50])

	if args.dump_selected_head:
		if os.path.exists('selected_head_stats.json'):
			stats_head = json.load(open('selected_head_stats.json', 'rt'))
		else:
			stats_head = []
		stats_head.append({
			"config" : dict([(k, v)for k, v in args.__dict__.items()] + [("path_file_result", path_file_result)]),
			"head_index": list(global_infor[g_args.filter_head_attn_pattern])
		})
		json.dump(stats_head, open("selected_head_stats.json", "wt"), indent=1) 
		exit(0)
	
	# ========================================
	# Load Model and Tokenizer 
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.padding_side = 'left'
	tokenizer.pad_token = tokenizer.eos_token

	tensor_data_type = torch.bfloat16   
	if "gpt-oss" not in  model_name:
		# bnb_config = BitsAndBytesConfig(
		# 	load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
		# )
		bnb_config = BitsAndBytesConfig(
			load_in_8bit=True
		)

		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			device_map="auto",
			torch_dtype=tensor_data_type,
			low_cpu_mem_usage=True,
			attn_implementation="my_new_sdpa",
			quantization_config = bnb_config if args.use_8bit_quantization else None,
		) 
	else:
		from transformers import Mxfp4Config
		# bnb_config = Mxfp4Config(
		# 	load_in_4bit=True
		# )
		bnb_config = Mxfp4Config(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

		model = AutoModelForCausalLM.from_pretrained(
			"openai/gpt-oss-20b",
			torch_dtype="bfloat16",  # or torch.float16
			# attn_implementation="my_new_sdpa",
			device_map="auto"
		)
	model.eval() 
	
	# ========================================
	# RUN generate logical attention masked 
	global final_masked_infor
	global final_masked_inc_attn
	final_masked_infor = []
	final_masked_inc_attn = []
	if args.logical_masked_func != no_masked_attn.__name__:
		# compute masked focus 
		prompt = all_prompt_content[0]
		all_shot_infor, answer_infor, final_masked_infor = logical_masked_func(prompt, tokenizer)
		if isinstance(final_masked_infor, tuple) and len(final_masked_infor) == 2:
			final_masked_infor, final_masked_inc_attn = final_masked_infor[0], final_masked_infor[1]
	
	# saving 
	# - map from module to its name to get information 
	# - model config 
	module_to_name = {m: n for n, m in model.named_modules()}
	global_infor['module_to_name'] = module_to_name
	global_infor['model_config'] = model.config 

	# (( 
	# dump attention if it is required
	if args.dump_attn_viz:
		all_prompt_content = all_prompt_content[:2]
		args.max_new_tokens = 1
		global_infor['prompt_content'] = all_prompt_content[0]
		global_infor['inputs'] = tokenizer(all_prompt_content, return_offsets_mapping=True)
	# ))
		
	# ========================================
	# RUN inference
	start_time = time.time()
	all_output_text = llm_infer(model, tokenizer, all_prompt_content, 
							 max_new_tokens=args.max_new_tokens, 
							 eos_tok_str=args.eos_tok_str,
							 batch_size=args.batch_size,
							 get_emb_vector=args.get_emb_vector)
	processing_time = str(timedelta(seconds=int(time.time()-start_time)))

	# (( 
	# dump attention if it is required
	if args.get_emb_vector:
		path_out = path_file_result.replace(".json", ".tensor") 
		print(f"- Save tensor data to {path_out}")
		with open(path_out, "wb") as f:
			torch.save({"raw_data": raw_data, "all_vector": torch.cat(all_output_text, dim=0)}, f)
		sys.exit() 
	# ))

	# (( 
	# dump attention if it is required
	if args.dump_attn_viz:
		print_tokens = [tokenizer.convert_ids_to_tokens(e) for e in global_infor['inputs'].input_ids]

		s_idx = -900 # -900 # -900 #cot #  -550 # -900 # 645  # len(tokens) - 100
		e_idx = -395 #-395 # -375 #cot # -268 # -395 # len(tokens)- 51
		head_ids=[0,3]
		dump_attn_viz(print_tokens[0], attn_weight_data=global_infor['attn_score'][0][head_ids[0]:head_ids[1], :,:], 
				 output_id=f"masked_attn{s_idx}-{e_idx}_h{head_ids[0]}-{head_ids[1]}_e2e", 
				 s_idx=s_idx, e_idx=e_idx, combine_words=5, pic_size=16)

		sys.exit() 
	# ))

	# ========================================
	# RUN evaluation and save results 
	results = []
	for prompt_content, output_text, raw_data_e in zip(all_prompt_content, all_output_text, raw_data):
		results.append({
				"id": raw_data_e['id'],
				'label': raw_data_e['label'],
				"prompt": prompt_content,
				"generated_output": output_text,
				"correct": None,
				"err_format": None,
			}) 

	with open(path_file_result, "w", encoding="utf-8") as f:
		saved_results = {
				"result": {'processing_time': processing_time},
				"config": dict([(k, v)for k, v in args.__dict__.items()] + [("path_file_result", path_file_result)]), 
				"detail_prediction": results, 
				"code_version": os.environ.get('CODE_VERSION')}
		json.dump(saved_results, f, ensure_ascii=False, indent=1) 
		

	data_eval, acc = eval_func(path_file_result)
	with open(path_file_result, "w", encoding="utf-8") as f:
		data_eval["result"] = {"acc": acc, 'processing_time': processing_time}
		json.dump(data_eval, f, ensure_ascii=False, indent=1)
