import argparse
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModel, AutoTokenizer,LlamaForCausalLM, AutoModelForCausalLM
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
import re
import pandas as pd
import json 
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import random, os
import numpy as np
import torch
from lightning import seed_everything

from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.integrations.flex_attention import flex_attention_forward, repeat_kv
import torch

import re

from evaluate import evaluate_json, evaluate_simple_lg, evaluate_simple_lg_logicdeduction
from accelerate import Accelerator

class StopOnsString(StoppingCriteria):
	def __init__(self, tokenizer, eos_tok_str):
		self.tokenizer = tokenizer
		self.buffer = ""
		self.eos_tok_str = eos_tok_str

	def __call__(self, input_ids, scores, **kwargs):
		texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
		return torch.tensor([text.endswith(self.eos_tok_str) for text in texts], device=input_ids.device)


# ===========================
# 1. Define Dataset (raw texts only)
# ===========================
class TextDataset(Dataset):
	def __init__(self, texts):
		self.texts = texts

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		return {"text": self.texts[idx]}

def llm_infer(model, tokenizer_, prompt_texts, eos_tok_str=None, max_new_tokens=600, batch_size=4, get_emb_vector=False):
	# Initialize Accelerator
	accelerator = Accelerator()

 
	# Data collator → pads dynamically to longest in batch
	data_collator = DataCollatorWithPadding(tokenizer_, padding="longest", return_tensors="pt")

	# Custom collate function: tokenize inside collator
	def collate_fn(batch):
		return data_collator([tokenizer_(item["text"]) for item in batch])
	
	dataset = TextDataset(prompt_texts)
	prompting_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

	# Prepare model and dataloader with Accelerator
	model, prompting_loader = accelerator.prepare(model, prompting_loader)

	all_outputs = []
	all_inputs = []
	
	# found eos token id
	eos_tok_str = eos_tok_str if eos_tok_str is not None else '-------'
	eos_tok_id = tokenizer_(eos_tok_str).input_ids[-1]

	tmp_inputs = tokenizer_(prompt_texts[0], return_offsets_mapping=True)
	input_ids = tmp_inputs.input_ids 
	offsets = tmp_inputs.offset_mapping 
	for idx in range(len(input_ids)):
		start, end = offsets[idx][0], offsets[idx][1]
		if eos_tok_str in prompt_texts[0][start:end]:
			print("- found eos token id in prompt_texts = ", input_ids[idx])
			eos_tok_id = input_ids[idx]
			break 
	print("- eos token id = ", eos_tok_id)
			
	model.eval()
	with torch.no_grad():
		for i_batch, batch in enumerate(tqdm(prompting_loader)): 
		# for batch in dataloader:
			if get_emb_vector:
				batch = {k: v.to(accelerator.device) for k, v in batch.items()} 
				outputs = model(**batch, output_hidden_states=True)
				layer_h_states = [h[:,-1,:].cpu().float() for h in outputs.hidden_states]
				layer_h_states = torch.cat(layer_h_states, dim=-1)
				all_outputs.append(layer_h_states.cpu().float())
			else:
				batch = {k: v.to(accelerator.device) for k, v in batch.items()}
				gen_kwargs = {'max_new_tokens': max_new_tokens, 
							'do_sample': False, 
							'eos_token_id': eos_tok_id, 
							'pad_token_id': tokenizer_.pad_token_id,
							'temperature': None, 
							'top_p': None, 
							'top_k': None, 
							"stopping_criteria": StoppingCriteriaList([StopOnsString(tokenizer_, eos_tok_str)])
							}
				input_length = batch['input_ids'].shape[1]
				outputs = model.generate(**batch, **gen_kwargs)
				raw_out = tokenizer_.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
				all_outputs.extend(raw_out)
		# if i_batch == 4:
		#	 json.dump(list(zip(prompt_texts[:len(all_outputs)], all_inputs, all_outputs)), open('tmp_out.json', "wt"), ensure_ascii=False, indent=2)

	# json.dump(list(zip(prompt_texts[:len(all_outputs)], all_inputs, all_outputs)), open('tmp_out.json', "wt"), ensure_ascii=False, indent=2)
	return all_outputs 

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Process ...')
	parser.add_argument('--prompting_data_path',type=str, help='prompting_data_path', default='data/ProofWriter/test.logiccotkb_prompting.json')
	parser.add_argument('--model_name',type=str, help='model_name', default="Qwen/Qwen3-14B") 
	parser.add_argument('--max_new_tokens',type=int, help='max_new_tokens', default=500) 
	parser.add_argument('--eos_tok_str',type=str, help='eos_tok_str', default="-------") 
	seed_everything(42) 

	args, unknown = parser.parse_known_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(args)
	
	prompting_data_path = args.prompting_data_path
	model_name =  args.model_name #  "Qwen/Qwen3-14B"   # "Qwen/Qwen3-8B"  #  # 'meta-llama/Llama-3.1-8B-Instruct'  'meta-llama/Meta-Llama-3-8B-Instruct'  
	
	path_file_result = prompting_data_path.replace(".json", f'.{model_name.split("/")[-1]}.ntok-{args.max_new_tokens}.result.json')
	print("path_file_result = ", path_file_result)
	eval_func = evaluate_simple_lg_logicdeduction if ("logicdeduction" in path_file_result and 'logiccot' in path_file_result ) else \
			evaluate_simple_lg if 'logiccot' in path_file_result else \
			evaluate_json
	
	if os.path.exists(path_file_result):
		print(f"File {path_file_result} already exists, skip inference")
		eval_data, acc = eval_func(path_file_result)
		exit(0) 
		
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.padding_side = 'left'
	tokenizer.pad_token = tokenizer.eos_token
	
	raw_data = json.load(open(prompting_data_path)) 
	all_prompt_content= [e['prompting'] for e in raw_data]
	
	global final_maskedsked_infor
	print(all_prompt_content[0])
 
	tensor_data_type = torch.bfloat16   
	# bnb_config_4bit = BitsAndBytesConfig(
	#	 load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=tensor_data_type
	# )
	bnb_config = BitsAndBytesConfig(
		load_in_8bit=True
	)
	

	if "32B" in model_name or "14B" in model_name:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			device_map="auto",
			low_cpu_mem_usage=True,
			quantization_config = bnb_config
		)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			device_map="auto",
			low_cpu_mem_usage=True,
		)
	
	model.eval() 
	all_output_text = llm_infer(model, tokenizer, all_prompt_content, max_new_tokens=args.max_new_tokens, eos_tok_str=args.eos_tok_str)

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
		json.dump(results, f, ensure_ascii=False, indent=1)
	with open(path_file_result[:-5]+".config.json", "w", encoding="utf-8") as f:
		json.dump(dict([(k, v)for k, v in args.__dict__.items()] + [("path_file_result", path_file_result)]), f, ensure_ascii=False, indent=1)
		

	data_eval, acc = eval_func(path_file_result)
	with open(path_file_result, "w", encoding="utf-8") as f:
		json.dump(data_eval, f, ensure_ascii=False, indent=1)
