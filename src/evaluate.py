import json
import os
import re

def evaluate_QA(result_file, QA_results=None):
    # this evaluation code is borrowed from repo of 
    # "Chengwen Qi et. al, 2025. Large language models meet symbolic provers for logical reasoning evaluation. ICLR 2025
    # https://github.com/opendatalab/ProverGen
    # 
    answer_cnt = [0]*7
    goal_answer_cnt =  [0]*7
    correct_propotion =  [0]*7
    
    if QA_results is None:
        with open(result_file, 'r') as f:
            QA_results = json.load(f)

    correct_cnt = 0
    total_count = 0
    error_parse_cnt = 0
    for sample in QA_results:
        gold_answer = sample['label']
        try:
            if sample['model_answer'].endswith('"answer": "A"\n}') or sample['model_answer'].startswith('{\n  "answer": "A"\n}'):
                sample['model_answer'] = '{\n  "answer": "A"\n}'
            elif sample['model_answer'].endswith('"answer": "B"\n}') or sample['model_answer'].startswith('{\n  "answer": "B"\n}'):
                sample['model_answer'] = '{\n  "answer": "B"\n}'
            elif sample['model_answer'].endswith('"answer": "C"\n}') or sample['model_answer'].startswith('{\n  "answer": "C"\n}'):
                # answer": "C"\n}
                sample['model_answer'] = '{\n  "answer": "C"\n}'
                
            prediction = eval(sample['model_answer'].replace('```json', '').replace('```', '').split("\n\n")[0])['answer']
            
            list_answer = list("ABCDEFG")
            if prediction == gold_answer:
                correct_cnt += 1
                
                if prediction == "A":
                    correct_propotion[0] += 1
                elif prediction == "B":
                    correct_propotion[1] += 1
                elif prediction == "C":
                    correct_propotion[2] += 1
                elif prediction in list_answer:
                    correct_propotion[list_answer.index(prediction)] += 1
            
            print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
            
            if prediction == "A":
                answer_cnt[0] += 1
            elif prediction == "B":
                answer_cnt[1] += 1
            elif prediction == "C":
                answer_cnt[2] += 1
            elif prediction in list_answer:
                answer_cnt[list_answer.index(prediction)] += 1
                
            if gold_answer == "A":
                goal_answer_cnt[0] += 1
            elif gold_answer == "B":
                goal_answer_cnt[1] += 1
            elif gold_answer == "C":
                goal_answer_cnt[2] += 1
            elif prediction in list_answer:
                goal_answer_cnt[list_answer.index(prediction)] += 1
        except:
            error_parse_cnt += 1

        total_count += 1
    
    avg_acc = correct_cnt / total_count
    print(f"Acc: {avg_acc}")
    print(f"Correct: {correct_cnt}. Total: {total_count}")
    print(f"Parse Error: {error_parse_cnt}")
    
    print(f"Goal Answer: {goal_answer_cnt}")
    print(f"Model Answer: {answer_cnt}")
    print(f"Correct: {correct_propotion}")
    return avg_acc


def evaluate_json(path_file, detail_print=False):
    pred_data = json.load(open(path_file))
    if 'detail_prediction' in pred_data:
        data = pred_data['detail_prediction']
    else:
        data = pred_data
    for e in data:
        pred = None
        if e['label'] not in ['A', 'B', "C", "D", "E", "F", "G"]:
            e['label'] = "A" if e['label'] == "True" else "B" if e['label'] == "False" else "C"
        try:
            e['model_answer'] = json.loads(e['generated_output'].split("}")[0]+"}") 
            if detail_print:
                print(json.dumps(e['model_answer'], indent=2))
            pred = e['model_answer']['answer']
        except:
            e['model_answer'] = e['generated_output']
            print(e['generated_output'])
            print("+++"*10)
            e['correct'] = False
        e['model_answer'] = json.dumps(e['model_answer'])
        e['correct'] = (pred == e['label'] and e['label']!= None)

    return pred_data, evaluate_QA(result_file=None, QA_results=data)
    
def evaluate_simple_lg(path_file, detail_print=False):
    pred_data = json.load(open(path_file))
    if 'detail_prediction' in pred_data:
        data = pred_data['detail_prediction']
    else:
        data = pred_data
    for e in data:
        label = e['label']
        if e['label'] not in ['A', 'B', "C", "D", "E", "F", "G"] and "gsm8k" not in path_file.lower():
            e['label'] = "A" if e['label'] == "True" else "B" if e['label'] == "False" else "C"
        try:
            output_text = e['generated_output']
            if "### Example" in output_text:
                output_text = output_text.split("### Example")[0] 
            output_text = output_text.split("-----")[0].strip()
            pred =  output_text.split("=")[-1].replace(".","").strip().lower()
            pred = "A" if pred == "true" else "B" if pred == "false" else "C"
        except:
            e['model_answer'] = {"answer":  output_text }
            print(e['generated_output'])
            print("+++"*10)
        if detail_print:
            print("+++"*10)
            print(e['id'])
            print(e['prompt'].split("----\n")[-1])
            print(output_text)
            print(f"Label: {label}")
        e['model_answer'] = json.dumps({"reasoning":output_text, "answer": pred})
        e['correct'] = (pred == e['label'])
    return pred_data, evaluate_QA(result_file=None, QA_results=data)
    
def evaluate_simple_lg_logicdeduction(path_file, detail_print=False):
    pred_data = json.load(open(path_file))
    if "gsm8k" in path_file.lower():
        raw_data = json.load(open('/home/ach17589xr/fuzzy_lg_llm/data/GSM8k/gsm8k_test.logiccotkb_prompting.json'))
        label_map = dict([(x['id'], x['label']) for x in raw_data])

    if 'detail_prediction' in pred_data:
        data = pred_data['detail_prediction']
    else:
        data = pred_data
    for e in data:
        label = e['label']
        
        if "gsm8k" in path_file.lower():
            label = label_map[e['id']]
            e['label'] = label

        if e['label'] not in ['A', 'B', "C", "D", "E", "F", "G"] and "gsm8k" not in path_file.lower():
            e['label'] = "A" if e['label'] == "True" else "B" if e['label'] == "False" else "C"
        try:
            output_text = e['generated_output']
            output_text = output_text.split("-----")[0].strip()
            if "### Example" in output_text:
                output_text = output_text.split("### Example")[0].strip()
            if "\n\n" in output_text:
                output_text = output_text.split("\n\n")[0].strip()
            last_line = output_text.split("\n")[-1].strip()
            if "{" in last_line and "}" in last_line:
                pred = ""
                try:
                    pred = json.loads(last_line)['answer']
                except:
                    pass 
            else:
                pred =  output_text.split("\n")[-1].split(" ")[-1].replace(".","").strip()
            if pred not in ['A', 'B', "C", "D", "E", "F", "G"]:
                print(e['generated_output'])
                print("-----?????")
        except:
            e['model_answer'] = {"answer":  output_text }
            print(e['generated_output'])
            print("+++"*10)
        if detail_print:
            print("+++"*10)
            print(e['id'])
            print(e['prompt'].split("----\n")[-1])
            print(output_text)
            print(f"Label: {label}")
        e['model_answer'] = json.dumps({"reasoning":output_text, "answer": pred})
        e['correct'] = (pred == e['label'])
    return pred_data, evaluate_QA(result_file=None, QA_results=data)



def evaluate_simple_lg_gsm8k(path_file, detail_print=False):
    pred_data = json.load(open(path_file))
    raw_data = json.load(open('/home/ach17589xr/fuzzy_lg_llm/data/GSM8k/gsm8k_test.logiccotkb_prompting.json'))
    label_map = dict([(x['id'], x['label']) for x in raw_data])

    def check_num(a, b):
        if b == "1,875":
            print()
        try:
            a = re.sub(r'[^0-9]', '', str(a))
            b = re.sub(r'[^0-9]', '', str(b))
            return float(a) == float(b)
        except:
            return False
    def extract_predicted_answer(text):
        regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
        regexes_to_ignore =[
            ",",
            "\\$",
            "(?s).*#### ",
            "\\.$"
        ]
        match = re.findall(regex_pattern, text)
        if match:
            match = match[-1]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            text = match.strip()

            for regex in regexes_to_ignore:
                text = re.sub(regex, "", text)
            return text
        else:
            return None

    def extract_ground_truth(text):
        return text.split('####')[-1].strip()
    
    if 'detail_prediction' in pred_data:
        data = pred_data['detail_prediction']
    else:
        data = pred_data
    count_true = 0
    for e in data:
        label = e['label']
        
        label = label_map[e['id']]
        e['label'] = label

        try:
            output_text = e['generated_output']
            output_text = output_text.split("-----")[0].strip()
            if "### Example" in output_text:
                output_text = output_text.split("### Example")[0].strip()
            if "\n\n" in output_text:
                output_text = output_text.split("\n\n")[0].strip()
            last_line = output_text.split("\n")[-1].strip()

            if "{" in last_line and "}" in last_line:
                pred = ""
                try:
                    pred = json.loads(last_line)['answer']
                except:
                    pass 
            else:
                last_line =  output_text.split("\n")[-1]
                pred = extract_predicted_answer(last_line)
        except:
            e['model_answer'] = {"answer":  output_text }
            print(e['generated_output'])
            print("+++"*10)
        if detail_print:
            print("+++"*10)
            print(e['id'])
            print(e['prompt'].split("----\n")[-1])
            print(output_text)
            print(f"Label: {label}")
        e['model_answer'] = json.dumps({"reasoning":output_text, "answer": pred})
        e['correct'] = (pred == e['label'] or check_num(pred, e['label']))
        if e['correct']:
            count_true+=1
    acc = count_true / len(data)
    return pred_data, acc

def evaluate_simple_lg_mask_attn(path_file, detail_print=False):
    pred_data = json.load(open(path_file))
    if 'detail_prediction' in pred_data:
        data = pred_data['detail_prediction']
    else:
        data = pred_data
    for e in data:
        label = e['label']
        if e['label'] not in ['A', 'B', "C", "D", "E", "F", "G"] and "gsm8k" not in path_file.lower():
            e['label'] = "A" if e['label'] == "True" else "B" if e['label'] == "False" else "C"
        try:
            output_text = e['generated_output']
            output_text = output_text.split("-----")[0].strip()
            if "### Example" in output_text:
                output_text = output_text.split("### Example")[0].strip()
            
            if "\n=> Validate(" in output_text:
                output_lines = []
                for l in output_text.split("\n"):
                    output_lines.append(l)
                    if l.startswith("=> Validate"):
                        break
                output_text = "\n".join(output_lines)
                
            last_line = output_text.split("\n")[-1].strip()
            if "{" in last_line and "}" in last_line:
                pred = ""
                try:
                    pred = json.loads(last_line)['answer']
                except:
                    pass 
            else:
                pred =  output_text.split("\n")[-1].split(" ")[-1].replace(".","").strip().lower()
                pred = "A" if pred == "true" else "B" if pred == "false" else "C"
        except Exception as exception_:
            e['model_answer'] = {"answer":  output_text }
            print(exception_)
            print(e['generated_output'])
            print("+++"*10)
        if detail_print:
            print("+++"*10)
            print(e['id'])
            print(e['prompt'].split("----\n")[-1])
            print(output_text)
            print(f"Label: {label}")
        e['model_answer'] = json.dumps({"reasoning":output_text, "answer": pred})
        e['correct'] = (pred == e['label'])
    return pred_data, evaluate_QA(result_file=None, QA_results=data)
if __name__=="__main__":
    all_results = []
    path_file_result_template = './data/ProofWriter/test.cot_prompting.Qwen3-{}B.ntok-500.result.json'
    path_file_result_template = './data/ProofWriter/test.logiccotkb_prompting.Llama-2-{}-hf.ntok-500.result.json'
    path_file_result_template = './data/GSM8k/gsm8k_test.cot_prompting.sep.phi-4.no_masked_attn.quant8bit-False.AA-False.H-None.cache-True.percent-0.95.result.diagon-0.8-None.json'
    for mod_size in ["1.7", "4", "8", "14", "32"][:1]:
        path_file_result = path_file_result_template.format(mod_size)
        if not os.path.exists(path_file_result): 
            found = False
            if os.path.exists(path_file_result.replace("quant8bit-False", 'quant8bit-True')):
                path_file_result = path_file_result.replace("quant8bit-False", 'quant8bit-True')
                found = True
            if os.path.exists(path_file_result.replace("quant8bit-True", 'quant8bit-False')):
                path_file_result = path_file_result.replace("quant8bit-True", 'quant8bit-False')
                found = True
            if not found:
                continue 
        
        eval_func = evaluate_simple_lg_logicdeduction if ("logicdeduction" in path_file_result) else \
            evaluate_simple_lg_gsm8k if ("gsm8k" in path_file_result.lower()) else \
            evaluate_simple_lg_mask_attn if ( 'AAI' in path_file_result ) else \
            evaluate_simple_lg if 'logiccot' in path_file_result else \
            evaluate_simple_lg_mask_attn
        
        d, v = eval_func(path_file_result)
        if isinstance(d, list):
            d= {'result': {}, 'detail_prediction': d}
        d['result']['acc'] = v
        json.dump(d, open(path_file_result, 'wt'), indent=1)
        print(v)
        all_results.append((mod_size, v))
    print(all_results)
    print(eval_func)
    