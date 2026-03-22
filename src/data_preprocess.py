
import nltk
import json
import random, os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from infer_llm_base import llm_infer
from datasets import load_dataset

def load_data_proverstyle(data_path):
    """
    Load FOLIO data from a JSON file.
    """
    all_raw_data = json.load(open(data_path))
    for e in tqdm(all_raw_data):
        e['all_rules'] = nltk.sent_tokenize(e['context']) # this gives us a list of sentences
        e['label'] = {"A":"True", "B":"False", "C": "Uncertain"}[e['answer']]
        e['main_question'] = e['question']
    return all_raw_data

def load_FOLIO_data(data_path):
    return load_data_proverstyle(data_path)

def load_original_FOLIO_data(data_path):
    with open(data_path, "r") as f:
        all_raw_data = [json.loads(l.strip()) for l in f.readlines()]
        
    for e in tqdm(all_raw_data):
        e['all_rules'] = nltk.sent_tokenize(e['premises']) # this gives us a list of sentences
        e['label'] = e['label']
        e['main_question'] = e['conclusion'].strip()
        e['id'] = e['example_id']
        
        # fake data for matching the format
        e['question'] = "Based on the above information, is the following statement true, false, or uncertain? "+e['conclusion']
        e['context'] = e['premises']
        e['options'] =[ "A) True", "B) False", "C) Uncertain"]
        e['answer'] = "A" if e['label'] == "True" else "B" if e['label'] == "False" else "C"
    return all_raw_data
 

def cot_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="CoT"):
    return mix_data_with_hard_prompting(_data, _tokenizer, path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring=keystring)

def mix_data_with_hard_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="CoT"):
    """
    Generate prompts for logic reasoning tasks.
    """
    if isinstance(_tokenizer, str):
        _tokenizer = AutoTokenizer.from_pretrained(_tokenizer)
        
    all_prompt_content = []
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = 'left'
    token_text = _tokenizer.decode([_tokenizer.eos_token_id])

    default_promt = json.load(open(path_prompting_cot_demonstration, "r"))[keystring]  
    for i, e in enumerate(_data):
        prompt_content = default_promt.replace("[[EOSTOKEN]]",token_text).replace("[[CONTEXT]]",e["context"]) .replace("[[QUESTION]]",e["question"]) .replace("[[OPTIONS]]","\n".join(e["options"])) 
        e['prompting'] = prompt_content
    return _data

def standard_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None):
    return cot_prompting(_data, _tokenizer,path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring="Direct")

def logiccotkb_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None):
    return logiccot_prompting(_data, _tokenizer,path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring="LogicCoTKB")

def logiccot_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="LogicCoT"):

    for e in tqdm(_data):
        if e['label'] not in "ABCDEFGH":
            e['label'] = {"A":"True", "B":"False", "C": "Uncertain"}[e['answer']]
        e['main_question'] = e['question'] 

    for i, e in enumerate(_data):
        all_rules = nltk.sent_tokenize(e['context']) # this gives us a list of sentences 
        rule_prompting = "\n".join([f"# (Rule{j+1}): {rule}" for j, rule in enumerate(all_rules)])
        e['context'] = rule_prompting
        e['question'] = e['question'] 
        
        e['options'] = e.get('options', [])
    return mix_data_with_hard_prompting(_data, _tokenizer, path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring=keystring)

def logiccotreverse_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="LogicCoTReverse"):
    return logiccot_prompting(_data, _tokenizer, path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring=keystring)

def logiccot_reverse_resoning_prompting(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="ReverseLogicCoT"):
    return logiccot_prompting(_data, _tokenizer, path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring=keystring)

def logiccot_reverse_resoning_promptingV2(_data, _tokenizer,path_prompting_cot_demonstration=None, keystring="ReverseLogicCoTv2"):
    return logiccot_prompting(_data, _tokenizer, path_prompting_cot_demonstration=path_prompting_cot_demonstration, keystring=keystring)
 

def logic_cot_OD_prompting(_data, _tokenizer, path_prompting_cot_demonstration=None):
    """
    Generate prompts for logic reasoning tasks.
    """
    if isinstance(_tokenizer, str):
        _tokenizer = AutoTokenizer.from_pretrained(_tokenizer)
        
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = 'left'
    token_text = _tokenizer.decode([_tokenizer.eos_token_id])

    default_promt = f"""### Let us define F as a function that determines whether a statement is True, False, or Uncertain, based on a given list of facts and rules. Using these facts and rules, provide a reasoning path that leads to one of the values: `True`, `False`, or `Uncertain`.
{token_text}
### Example1: Given list of facts and rules: 
# (Rule1): If K then P; 
# (Rule2): Q is true; 
# (Rule3): If Q then K. 
# (Question): truth value of P? 
# (Answer): F(Rule2) => F(Rule2, Rule3, Output1=`K`) => F(Output1, Rule1, Output2=`P`) => F(Question) = True. 
{token_text}
### Example2: Given list of facts and rules:  
# (Rule1): If the restaurant is listed in Yelp’s recommendations, then the restaurant does not receive many negative reviews. 
# (Rule2): All restaurants with a rating greater than 9 are listed in Yelp’s recommendations. 
# (Rule3): Some restaurants that do not provide take-out service receive many negative reviews. 
# (Rule4): All restaurants that are popular among local residents have ratings greater than 9. 
# (Rule5): Subway has a rating greater than 9 or is popular among local residents.
# (Question): Subway provides take-out service and does not receive many negative reviews.
# (Answer): F(Rule5) =>  F(Rule5, Rule4, Output1=`Subway has a ratings greater than 9`) => F(Output1, Rule2, Output2=`Subway is listed in Yelp’s recommendations`) => F(Output2, Rule1, Output3=`Subway does not receive many negative reviews`) => F(Output3, Rule3, Output4 = `Subway provides take-out service`) => F(Question) = Uncertain. 
{token_text}
### Example3: Given list of facts and rules:"""
    for i, e in enumerate(_data):
        rules = e['all_rules']
        rule_prompting = "\n".join([f"# (Rule{j+1}): {rule}" for j, rule in enumerate(rules)])
        question_prompting = f"# (Question): {e['main_question']}" 
        answer_prompting = "# (Answer):"
        prompt_content = f"""{default_promt}\n{rule_prompting}\n{question_prompting}\n{answer_prompting}"""
        e['prompting'] = prompt_content
    return _data

def gen_proofwriter_promting(data_path="./data/ProofWriter/test.json", 
                             model_name="meta-llama/Llama-3.1-8B-Instruct", 
                             prompting_function = logiccot_reverse_resoning_prompting, file_name_id=None):
    _data = load_data_proverstyle(data_path)
    file_name_id = file_name_id+"." if file_name_id is not None else ""
    
    prompting_data = prompting_function(_data, model_name, 
                                        path_prompting_cot_demonstration="./data/icl_examples/ProofWriter.json")
    
    json.dump(prompting_data, 
              open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}.{file_name_id}json"), "wt"), 
              ensure_ascii=False, 
              indent=1)


def gen_folio_promting(data_path="./data/FOLIO/folio_test.jsonl", 
                             model_name="meta-llama/Llama-3.1-8B-Instruct", 
                             prompting_function = logiccot_reverse_resoning_prompting, file_name_id=None):
    _data = load_original_FOLIO_data(data_path)
    file_name_id = file_name_id+"." if file_name_id is not None else ""
    
    prompting_data = prompting_function(_data, model_name, 
                                        path_prompting_cot_demonstration="./data/icl_examples/FOLIO.json")
    
    json.dump(prompting_data, 
              open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}.{file_name_id}json"), "wt"), 
              ensure_ascii=False, 
              indent=1)
    

def gen_gsm8k_promting(data_path=None, model_name=None, prompting_function = None, file_name_id=None, keystring=None):

    print('\nLoading dataset gsm8k...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset) 
    print('gsm8k test size:', datasize) 

    def extract_ground_truth(text):
        return text.split('####')[-1].strip()
    
    # Define a stopping condition for generation 
    _data = []
    for i in tqdm(range(datasize), desc='Data processing'):
        e = {}
        example = dataset[i]
        input_text = example['question']
        reasoning_answer = example['answer']
        gold_answer = extract_ground_truth(example['answer'])
        sentences = nltk.sent_tokenize(input_text) # this gives us a list of sentences

        # fake data structure for matching the format
        e['id'] = f"gsm8k_test_{i}"
        e['label'] = gold_answer
        e['question'] = sentences[-1]
        e['main_question'] = sentences[-1]
        e['context'] = input_text
        e['answer'] = gold_answer
        e['reasoning_answer'] = reasoning_answer
        _data.append(e)

    file_name_id = file_name_id+"." if file_name_id is not None else ""
     

    default_promt = json.load(open("./data/icl_examples/GSM8k.json", "r"))[keystring]  
    for i, e in enumerate(_data):
        prompt_content = default_promt.replace("[[CONTEXT]]",e["context"])  
        e['prompting'] = prompt_content

    if keystring == "LogicCoTKB":
        file_out_name = f"./data/GSM8k/gsm8k_test.logiccotkb_prompting.{file_name_id}json"
    elif keystring == "CoT":
        file_out_name = f"./data/GSM8k/gsm8k_test.cot_prompting.{file_name_id}json"
    
    print("- Writing file: ", file_out_name)
    json.dump(_data, 
              open(file_out_name, "wt"), 
              ensure_ascii=False, 
              indent=1)
    
def gen_folio_dev_promting(data_path="./data/FOLIO/dev.json", 
                             model_name="meta-llama/Llama-3.1-8B-Instruct", 
                             prompting_function = logiccot_reverse_resoning_prompting, file_name_id=None):
    _data = load_data_proverstyle(data_path)
     
    file_name_id = file_name_id+"." if file_name_id is not None else ""
    
    prompting_data = prompting_function(_data, model_name, 
                                        path_prompting_cot_demonstration="./data/icl_examples/FOLIO.json")
    
    json.dump(prompting_data, 
              open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}.{file_name_id}json"), "wt"), 
              ensure_ascii=False, 
              indent=1)
def gen_ld_promting(data_path="./data/logicdeduction/logicdeduction.json", 
                             model_name="meta-llama/Llama-3.1-8B-Instruct", 
                             prompting_function = logiccot_reverse_resoning_prompting, file_name_id=None):
    
    _data = json.load(open(data_path))
    for e in tqdm(_data):
        e['all_rules'] = nltk.sent_tokenize(e['context']) # this gives us a list of sentences
        e['label'] = e['answer']
        e['main_question'] = e['question']

     
    file_name_id = file_name_id+"." if file_name_id is not None else ""
    
    prompting_data = prompting_function(_data, model_name, 
                                        path_prompting_cot_demonstration="./data/icl_examples/logicdeduction.json")
    
    json.dump(prompting_data, 
              open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}.{file_name_id}json"), "wt"), 
              ensure_ascii=False, 
              indent=1)
    
def gen_prontoqa_promting(data_path="./data/prontoQA/dev.json", 
                             model_name="meta-llama/Llama-3.1-8B-Instruct", 
                             prompting_function = logiccotkb_prompting, file_name_id=None):
    _data = load_data_proverstyle(data_path)
    file_name_id = file_name_id+"." if file_name_id is not None else ""
    
    prompting_data = prompting_function(_data, model_name, 
                                        path_prompting_cot_demonstration="./data/icl_examples/ProntoQA.json")
    
    json.dump(prompting_data, 
              open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}.{file_name_id}json"), "wt"), 
              ensure_ascii=False, 
              indent=1)
# def gen_folio_promting(data_path="./data/ProofWriter/test.json", 
#                              model_name="meta-llama/Llama-3.1-8B-Instruct"):
#     _data = load_data_proverstyle(data_path)
#     prompting_function = logiccot_prompting
    
#     prompting_data = logiccot_prompting(_data, model_name, 
#                                         path_prompting_cot_demonstration="./data/icl_examples/ProofWriter.json")
    
#     json.dump(prompting_data, 
#               open(data_path.replace(".jsonl" if data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}--{model_name.split('/')[-1]}.json"), "wt"), 
#               ensure_ascii=False, 
#               indent=1)
# def old_exp():
#     # Load FOLIO data
#     model_name =  "meta-llama/Llama-3.1-8B-Instruct"
#     folio_data_path = "./libs/ProverGen/logic_data/FOLIO/dev.json"
    
#     folio_data_path = "./data/ProofWriter/test.json"
#     folio_data = load_data_proverstyle(folio_data_path)
    
#     # folio_data_path = "./data/FOLIO/folio_validation.jsonl"
#     # folio_data = load_original_FOLIO_data(folio_data_path)
#     # prompting_function = logic_cot_prompting
    
#     folio_data_path = "./data/ProofWriter/test.json"
#     folio_data = load_data_proverstyle(folio_data_path)
#     prompting_function = logiccot_reverse_resoning_prompting
    
#     prompting_data = prompting_function(folio_data, model_name, path_prompting_cot_demonstration="./data/icl_examples/ProofWriter.json")
    
#     json.dump(prompting_data, 
#               open(folio_data_path.replace(".jsonl" if folio_data_path.endswith("jsonl") else ".json", f".{prompting_function.__name__}--{model_name.split('/')[-1]}.json"), "wt"), 
#               ensure_ascii=False, 
#               indent=1)
#     pass
    
if __name__=="__main__":
     
    gen_gsm8k_promting(prompting_function = logiccot_prompting, keystring="CoT")
    # gen_ld_promting(model_name="Qwen/Qwen3-8B", prompting_function = logiccot_prompting, 
    #                     #    data_path="./data/logicdeduction/logicdeduction.json",
    #                        file_name_id="sim")
    # gen_proofwriter_promting(data_path="./data/ProofWriter/test.json", 
    #                          model_name="meta-llama/Llama-3.1-8B-Instruct", 
    #                          prompting_function = logiccotkb_prompting, file_name_id=None)
    # logiccot_reverse_resoning_prompting(model_name="Qwen/Qwen3-8B")
    # gen_folio_promting(model_name="Qwen/Qwen3-8B")
    