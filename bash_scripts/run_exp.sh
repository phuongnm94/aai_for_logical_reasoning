#!/usr/bin/bash
#
#         Job Script for VPCC , JAIST
#                                    2018.2.25 

#PBS -N lgCot
#PBS -j oe 
#PBS -q GPU-1
#PBS -o data/ProofWriter/logs/
#PBS -e data/ProofWriter/logs/

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

cd ~/aai_for_logical_reasoning/ 
conda activate ./env_aai_lr/ 

CODE_VERSION=$(git diff src/*.py bash_scripts/*.sh)
export CODE_VERSION

### MASKED_FUNC
# no_masked_attn                                        # for baseline model (Symbolic-Aided CoT)
# generate_focusing_rule_masked_positions               # for AAI^{-HeadsSel.}
# generate_focusing_rule_inc_attn_masked_positions      # for AAI
 

DATA_PATH="./data/ProofWriter/test.logiccotkb_prompting.json"
 
 

MODEL_NAME="Qwen/Qwen3-32B"  # or "allenai/OLMo-2-0325-32B-Instruct" | "microsoft/phi-4"
 
# default setting - baseline system 
MASKED_FUNC="no_masked_attn"  
python ./src/infer_llm.py    \
    --model_name $MODEL_NAME  --prompting_data_path $DATA_PATH --max_new_tokens 2000 \
    --logical_masked_func $MASKED_FUNC --batch_size 4 

# AAI main setting - filter diagonal pattern heads
MASKED_FUNC="generate_focusing_rule_inc_attn_masked_positions"  
FILTER_HEAD="filter_high_diagonal_attention"
python ./src/infer_llm.py    \
    --model_name $MODEL_NAME  --prompting_data_path $DATA_PATH --max_new_tokens 2000 \
    --logical_masked_func $MASKED_FUNC --batch_size 4 \
    --apply_dynamic_attn_pattern --strong_att_const 0.04 --filter_head_attn_pattern $FILTER_HEAD 

# AAI^{agg.} setting - filter diagonal pattern heads
MASKED_FUNC="generate_focusing_rule_inc_attn_masked_positions"  
FILTER_HEAD="filter_high_verticle_low_others_attention"
python ./src/infer_llm.py    \
    --model_name $MODEL_NAME  --prompting_data_path $DATA_PATH --max_new_tokens 2000 \
    --logical_masked_func $MASKED_FUNC --batch_size 4 \
    --apply_dynamic_attn_pattern --strong_att_const 0.04 --filter_head_attn_pattern $FILTER_HEAD 



## ------------------------ 
## Note: use --eos_tok_str parameter config for  "allenai/OLMo-2-0325-32B-Instruct" model
## for early-stopping
## ------------------------ 
# MODEL_NAME="allenai/OLMo-2-0325-32B-Instruct"
# python ./src/infer_llm.py    \
#     --model_name $MODEL_NAME  --prompting_data_path $DATA_PATH --max_new_tokens 2000 \
#     --logical_masked_func $MASKED_FUNC --batch_size 4 \
#     --apply_dynamic_attn_pattern --strong_att_const 0.04 --filter_head_attn_pattern $FILTER_HEAD \
#     --eos_tok_str "\\n\\n" 

wait

