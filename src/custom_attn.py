import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def mapping_offset_to_idx(start_offset_, end_offset_, tokenizer_offsets):
	start_tok_idx = None
	end_tok_idx = None
	for tok_idx, (s, e) in enumerate(tokenizer_offsets):
		if start_tok_idx is None and start_offset_ < s:
			start_tok_idx = tok_idx - 1
		if  end_tok_idx is None and  end_offset_ < e:
			end_tok_idx = tok_idx+1
	return [start_tok_idx, end_tok_idx] 

def mapping_offset_to_idx2(start_offset_, end_offset_, tokenizer_offsets):
	start_tok_idx = None
	end_tok_idx = None
	for tok_idx, (s, e) in enumerate(tokenizer_offsets):
		if start_tok_idx is None and start_offset_ < s:
			start_tok_idx = tok_idx - 1
		if  end_tok_idx is None and  end_offset_ <= e:
			end_tok_idx = tok_idx+1
	return [start_tok_idx, end_tok_idx]

def generate_focusing_rule_masked_positions(prompt_content, tokenizer, fewshot_bounding="-------"):
	few_shot_contents = prompt_content.split(fewshot_bounding)
	inputs = tokenizer(prompt_content, return_tensors="pt", return_offsets_mapping=True)
	# input_ids = inputs.input_ids[0]
	# offsets = inputs.offset_mapping[0]

	# for idx in range(len(inputs.input_ids[0])):
	# 	start, end = inputs['offset_mapping'][0][idx][0], inputs['offset_mapping'][0][idx][1]
	# 	print(idx,  prompt_content[start:end])
  
	def process_one_shot(one_shot_content, whole_prompt_content, tokenizer_offsets, base_offset=0, sample_id=None):

		all_lines = one_shot_content.strip().split("\n")
		rules = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		answer_inference_steps = []
		for iline, e in enumerate(all_lines):
			if e.startswith("# (Answer):"):
				answer_inference_steps = all_lines[iline:]
				break 
		# answer_inference_steps = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Answer):") or e.startswith("=> F(")]
		rules_for_answer_token_idx = []
		if len(answer_inference_steps) > 0:
			answer_inference_content = "\n".join(answer_inference_steps)
			answer_inference_steps = answer_inference_content.split("=>")
			answer_inference_offset = base_offset+whole_prompt_content[base_offset:].index( answer_inference_content)
			for m in re.compile(r"Rule\d+").finditer(answer_inference_content):
				rule_id_start_idx = m.start() + answer_inference_offset
				rule_id_end_idx = m.end() + answer_inference_offset

				rules_for_answer_token_idx.append([mapping_offset_to_idx(rule_id_start_idx, rule_id_end_idx, tokenizer_offsets), m.group(), sample_id])
		

		rule_ids = [ (sample_id, e.split(":")[0][3:-1]) for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		rule_offsets = []
		rule_tok_idx = []
		for rule_content in rules:
			rule_offset = base_offset + whole_prompt_content[base_offset:].index(rule_content)
			start_offset = rule_offset
			end_offset = start_offset + len(rule_content)
			rule_offsets.append([start_offset, end_offset])

			rule_tok_idx.append(mapping_offset_to_idx(start_offset, end_offset, tokenizer_offsets))
		rules = [rule_content.strip() for rule_content in rules]
		return list(zip(rule_ids, zip(rule_offsets, rules, rule_tok_idx))), rules_for_answer_token_idx
	
	all_shot_infor = []
	answer_infor = []
	final_masked = []
	for i_shot, shot in enumerate(few_shot_contents):
		rule_descriptions, rules_for_answer_infor = process_one_shot(shot, 
									prompt_content, 
									tokenizer_offsets=inputs.offset_mapping[0],
									base_offset=len(f"{fewshot_bounding}".join(few_shot_contents[:i_shot])) + (len(fewshot_bounding) if i_shot > 0 else 0),
									sample_id=i_shot
									)
		all_shot_infor += rule_descriptions
		answer_infor += rules_for_answer_infor

		for ans in rules_for_answer_infor:
			for rule_check in rule_descriptions:
				selected_rule_id = (ans[2], ans[1])
				if rule_check[0] != selected_rule_id:
					final_masked.append([ans[0], rule_check[1][2]])


	return all_shot_infor, answer_infor, final_masked
 
def generate_focusing_rule_inc_attn_masked_positions(prompt_content, tokenizer, fewshot_bounding="-------"):
	few_shot_contents = prompt_content.split(fewshot_bounding)
	inputs = tokenizer(prompt_content, return_tensors="pt", return_offsets_mapping=True)
	# input_ids = inputs.input_ids[0]
	# offsets = inputs.offset_mapping[0]

	# for idx in range(len(inputs.input_ids[0])):
	# 	start, end = inputs['offset_mapping'][0][idx][0], inputs['offset_mapping'][0][idx][1]
	# 	print(idx,  prompt_content[start:end])
  
	def process_one_shot(one_shot_content, whole_prompt_content, tokenizer_offsets, base_offset=0, sample_id=None):

		all_lines = one_shot_content.strip().split("\n")
		rules = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		answer_inference_steps = []
		for iline, e in enumerate(all_lines):
			if e.startswith("# (Answer):"):
				answer_inference_steps = all_lines[iline:]
				break 
		# answer_inference_steps = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Answer):") or e.startswith("=> F(")]
		rules_for_answer_token_idx = []
		if len(answer_inference_steps) > 0:
			answer_inference_content = "\n".join(answer_inference_steps)
			answer_inference_steps = answer_inference_content.split("=>")
			answer_inference_offset = base_offset+whole_prompt_content[base_offset:].index( answer_inference_content)
			for m in re.compile(r"Rule\d+").finditer(answer_inference_content):
				rule_id_start_idx = m.start() + answer_inference_offset
				rule_id_end_idx = m.end() + answer_inference_offset

				rules_for_answer_token_idx.append([mapping_offset_to_idx(rule_id_start_idx, rule_id_end_idx, tokenizer_offsets), m.group(), sample_id])
		

		rule_ids = [ (sample_id, e.split(":")[0][3:-1]) for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		rule_offsets = []
		rule_tok_idx = []
		for rule_content in rules:
			rule_offset = base_offset + whole_prompt_content[base_offset:].index(rule_content)
			start_offset = rule_offset
			end_offset = start_offset + len(rule_content)
			rule_offsets.append([start_offset, end_offset])

			rule_tok_idx.append(mapping_offset_to_idx(start_offset, end_offset, tokenizer_offsets))
		rules = [rule_content.strip() for rule_content in rules]
		return list(zip(rule_ids, zip(rule_offsets, rules, rule_tok_idx))), rules_for_answer_token_idx
	
	all_shot_infor = []
	answer_infor = []
	final_masked = []
	final_masked_inc_attn = []
	for i_shot, shot in enumerate(few_shot_contents):
		rule_descriptions, rules_for_answer_infor = process_one_shot(shot, 
									prompt_content, 
									tokenizer_offsets=inputs.offset_mapping[0],
									base_offset=len(f"{fewshot_bounding}".join(few_shot_contents[:i_shot])) + (len(fewshot_bounding) if i_shot > 0 else 0),
									sample_id=i_shot
									)
		all_shot_infor += rule_descriptions
		answer_infor += rules_for_answer_infor

		for ans in rules_for_answer_infor:
			for rule_check in rule_descriptions:
				selected_rule_id = (ans[2], ans[1])
				if rule_check[0] != selected_rule_id:
					final_masked.append([ans[0], rule_check[1][2]])
				else:
					final_masked_inc_attn.append([ans[0], rule_check[1][2]])

	return all_shot_infor, answer_infor, (final_masked, final_masked_inc_attn)

def generate_constraint_focal_rule_masked_positions(prompt_content, tokenizer, fewshot_bounding="-------"):
	few_shot_contents = prompt_content.split(fewshot_bounding)
	inputs = tokenizer(prompt_content, return_tensors="pt", return_offsets_mapping=True) 
 
	# for idx in range(len(inputs.input_ids[0])):
	# 	start, end = inputs['offset_mapping'][0][idx][0], inputs['offset_mapping'][0][idx][1]
	# 	print(idx,  prompt_content[start:end])
	def process_one_shot(one_shot_content, whole_prompt_content, tokenizer_offsets, base_offset=0, sample_id=None):
		all_lines = one_shot_content.strip().split("\n")
		rules = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		# answer_inference_steps = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Answer):")]
		answer_inference_steps = []
		for iline, e in enumerate(all_lines):
			if e.startswith("# (Answer):"):
				answer_inference_steps = all_lines[iline:]
				break 

		rules_for_answer_token_idx = []
		if len(answer_inference_steps) > 0:
			answer_inference_content = "\n".join(answer_inference_steps) # answer_inference_steps[0]
			answer_inference_steps = answer_inference_content.split("=>")
			answer_inference_offset = base_offset+whole_prompt_content[base_offset:].index( answer_inference_content)
			for infer_func in re.compile(r"(F\(.*\))").finditer(answer_inference_content):
				rule_id_start_idx = infer_func.start(1) + answer_inference_offset
				rule_id_end_idx = infer_func.end(1) + answer_inference_offset

				for rule in re.compile(r"Rule\d+").finditer(infer_func.group()):
					rules_for_answer_token_idx.append([mapping_offset_to_idx(rule_id_start_idx, rule_id_end_idx, tokenizer_offsets), rule.group(), sample_id])
		

		rule_ids = [ (sample_id, e.split(":")[0][3:-1]) for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		rule_offsets = []
		rule_tok_idx = []
		for rule_content in rules:
			rule_offset = base_offset + whole_prompt_content[base_offset:].index(rule_content)
			start_offset = rule_offset
			end_offset = start_offset + len(rule_content)
			rule_offsets.append([start_offset, end_offset])

			rule_tok_idx.append(mapping_offset_to_idx(start_offset, end_offset, tokenizer_offsets))
					
					
		rules = [rule_content.strip() for rule_content in rules]
		

	
		return list(zip(rule_ids, zip(rule_offsets, rules, rule_tok_idx))), rules_for_answer_token_idx
	
	all_shot_infor = []
	answer_infor = []
	final_masked = []
	final_masked_inc_attn = []
	for i_shot, shot in enumerate(few_shot_contents):
		rule_descriptions, rules_for_answer_infor = process_one_shot(shot, 
									prompt_content, 
									tokenizer_offsets=inputs.offset_mapping[0],
									base_offset=len(f"{fewshot_bounding}".join(few_shot_contents[:i_shot])) + (len(fewshot_bounding) if i_shot > 0 else 0),
									sample_id=i_shot
									)
		all_shot_infor += rule_descriptions
		answer_infor += rules_for_answer_infor

		reasoning_paths = set([(e[0][0], e[0][1], e[1], e[2]) for e in rules_for_answer_infor])
		reasoning_positions = set([(e[0][0], e[0][1]) for e in rules_for_answer_infor])
		for resoning_position in reasoning_positions:
			for rule_check in rule_descriptions:
				if (resoning_position[0], resoning_position[1], rule_check[0][1], rule_check[0][0]) not in reasoning_paths:
					final_masked.append([resoning_position, rule_check[1][2]])
				else:
					final_masked_inc_attn.append([resoning_position, rule_check[1][2]])

	return all_shot_infor, answer_infor,  (final_masked, final_masked_inc_attn)

def generate_constraint_rule_masked_positions(prompt_content, tokenizer, fewshot_bounding="-------"):
	few_shot_contents = prompt_content.split(fewshot_bounding)
	inputs = tokenizer(prompt_content, return_tensors="pt", return_offsets_mapping=True) 
 
	# for idx in range(len(inputs.input_ids[0])):
	# 	start, end = inputs['offset_mapping'][0][idx][0], inputs['offset_mapping'][0][idx][1]
	# 	print(idx,  prompt_content[start:end])
	def process_one_shot(one_shot_content, whole_prompt_content, tokenizer_offsets, base_offset=0, sample_id=None):
		all_lines = one_shot_content.strip().split("\n")
		rules = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		answer_inference_steps = [ e for iline, e in enumerate(all_lines) if e.startswith("# (Answer):")]
		rules_for_answer_token_idx = []
		if len(answer_inference_steps) > 0:
			answer_inference_content = answer_inference_steps[0]
			answer_inference_steps = answer_inference_content.split("=>")
			answer_inference_offset = base_offset+whole_prompt_content[base_offset:].index( answer_inference_content)
			for infer_func in re.compile(r"F\([^=]*(Output\d*=`?)").finditer(answer_inference_content):
				rule_id_start_idx = infer_func.start(1) + answer_inference_offset
				rule_id_end_idx = infer_func.end(1) + answer_inference_offset

				for rule in re.compile(r"Rule\d+").finditer(infer_func.group()):
					rules_for_answer_token_idx.append([mapping_offset_to_idx(rule_id_start_idx, rule_id_end_idx, tokenizer_offsets), rule.group(), sample_id])
		

		rule_ids = [ (sample_id, e.split(":")[0][3:-1]) for iline, e in enumerate(all_lines) if e.startswith("# (Rule")]
		rule_offsets = []
		rule_tok_idx = []
		for rule_content in rules:
			rule_offset = base_offset + whole_prompt_content[base_offset:].index(rule_content)
			start_offset = rule_offset
			end_offset = start_offset + len(rule_content)
			rule_offsets.append([start_offset, end_offset])

			rule_tok_idx.append(mapping_offset_to_idx(start_offset, end_offset, tokenizer_offsets))
					
					
		rules = [rule_content.strip() for rule_content in rules]
		

	
		return list(zip(rule_ids, zip(rule_offsets, rules, rule_tok_idx))), rules_for_answer_token_idx
	
	all_shot_infor = []
	answer_infor = []
	final_masked = []
	for i_shot, shot in enumerate(few_shot_contents):
		rule_descriptions, rules_for_answer_infor = process_one_shot(shot, 
									prompt_content, 
									tokenizer_offsets=inputs.offset_mapping[0],
									base_offset=len(f"{fewshot_bounding}".join(few_shot_contents[:i_shot])) + (len(fewshot_bounding) if i_shot > 0 else 0),
									sample_id=i_shot
									)
		all_shot_infor += rule_descriptions
		answer_infor += rules_for_answer_infor

		reasoning_paths = set([(e[0][0], e[0][1], e[1], e[2]) for e in rules_for_answer_infor])
		reasoning_positions = set([(e[0][0], e[0][1]) for e in rules_for_answer_infor])
		for resoning_position in reasoning_positions:
			for rule_check in rule_descriptions:
				if (resoning_position[0], resoning_position[1], rule_check[0][1], rule_check[0][0]) not in reasoning_paths:
					final_masked.append([resoning_position, rule_check[1][2]])



	return all_shot_infor, answer_infor, final_masked

def generate_mixed_focusing_and_constraint_masked_attn(prompt_content, tokenizer, fewshot_bounding="-------"):
    all_shot_infor, answer_infor, final_masked = generate_constraint_rule_masked_positions(prompt_content, tokenizer, fewshot_bounding)
    _, _, final_masked2 = generate_focusing_rule_masked_positions(prompt_content, tokenizer, fewshot_bounding)
    return all_shot_infor, answer_infor, final_masked + final_masked2

def no_masked_attn(*args, **kwargs):
	pass 

		
def dump_attn_viz(tokens_, attn_weight_data, output_id="c", s_idx=0, e_idx=50, combine_words=10, pic_size=4):

    tokens_ = tokens_ [s_idx:e_idx]
    attn_weight_data = attn_weight_data.to(torch.float)
    attn_layer = attn_weight_data[:, s_idx:e_idx, s_idx:e_idx]   # shape: (num_heads, seq_len, seq_len)

    # s_idx = 1200-12
    def group_token(tokens_, n_combine_words):
        new_tokens = []
        for i in range(0, len(tokens_)):
            if i%n_combine_words == 4:
                new_tokens.append("".join(tokens_[i-n_combine_words: i]).replace("Ġ", " "))# .replace("Ċ", "\\n"))
            else:
                new_tokens.append("")
        return new_tokens

    viz_tokens = group_token(tokens_, combine_words)
 
    num_heads = attn_layer.shape[0]

    # --- heatmap for all heads ---
    cols = 2  #  
    rows = (num_heads + cols - 1) // cols  #  

    fig, axes = plt.subplots(rows, cols, figsize=(pic_size*cols, pic_size*rows))

    for h in range(num_heads):
        ax = axes[h // cols, h % cols]
        sns.heatmap(attn_layer[h].detach().cpu().numpy(),
                    xticklabels=viz_tokens,
                    yticklabels=viz_tokens,
                    cmap="Blues",
                      cbar=False, square=True, ax=ax)
        ax.set_title(f"Head {h}")
        ax.set_xticklabels(viz_tokens, rotation=90)
        ax.set_yticklabels(viz_tokens, rotation=0)

    # hidden subplot 
    for h in range(num_heads, rows*cols):
        fig.delaxes(axes[h // cols, h % cols])

    plt.suptitle(f"Attention Heatmaps: "+output_id, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"./src/analyze/{output_id}_attn_{len(viz_tokens)}.pdf")
    # plt.show()
    plt.clf()

