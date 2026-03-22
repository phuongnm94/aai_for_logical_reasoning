
import json
import logging
import os
import pickle
import sys
from dotenv import dotenv_values
from sklearn.metrics import f1_score
import torch
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import traceback

from evaluate import evaluate_json, evaluate_simple_lg, evaluate_simple_lg_logicdeduction
   
class LLMBaseQuery:
    def __init__(self, args) -> None:
        self.args = args 
        note_str = "."+args.note+"." if args.note is not None else ""
        self.base_id = args.data_path.replace(".json", f".{args.model}.ntok-{args.max_new_tokens}{note_str}/")
        os.makedirs(self.base_id, exist_ok=True)
        
        self.run_id = f"{self.base_id}/result.json"
        json.dump(vars(args), open(self.run_id.replace(".json", ".config.json"), 'wt'), indent=2)
        logging.basicConfig(filename=self.run_id.replace(".json", ".log"), filemode='a', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        logging.info(self.args)
 
    
    def prepare_openai_query(self):
        
        with open(args.data_path) as f:
            query_data = json.load(f) 
            print(query_data[0])
        tracking_submission_file = f"{self.base_id}/tracking_submission.json"
        if not os.path.exists(tracking_submission_file):
            all_data = []
            for idx, sample in enumerate(query_data):
                sample_data = {
                    "custom_id": str(idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                            "model": self.args.model,
                            "messages": [
                                {"role": "system", "content": "You are an expert in logical reasoning tasks. Follow the examples provided below to solve the new problem:"},
                                {"role": "user", "content": sample['prompting']}
                            ],
                            "temperature": 0.1,
                            "max_tokens": self.args.max_new_tokens
                    }
                }
                all_data.append(json.dumps(sample_data))
            
            json.dump(query_data, open(f'{self.base_id}/query_data.json', 'wt'), indent=1, ensure_ascii=False)
            with  open(f'{self.base_id}/data_openai.jsonl', 'wt') as f:
                f.write("\n".join(all_data)) 
                
            from openai import OpenAI

            config = dotenv_values(".env_reasoning_llm")
            KEY=config['OPENAI_KEY'] 
            client = OpenAI(api_key=KEY)
                
            batch_input_file = client.files.create(
                file=open(f'{self.base_id}/data_openai.jsonl', "rb"),
                purpose="batch"
            )
            batch_obj = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"{self.base_id}"
                }
            )
            print(batch_obj)
            json.dump(
                dict([(str(k), str(v)) for k, v in batch_obj.__dict__.items()]), 
                open(tracking_submission_file, "wt"), 
                indent=1)
        else:
            if not os.path.exists(f'{self.base_id}/result_submission.json'):
                from openai import OpenAI

                config = dotenv_values(".env_reasoning_llm")
                KEY=config['OPENAI_KEY'] 
                client = OpenAI(api_key=KEY)
                submission_info = json.load(open(f'{self.base_id}/tracking_submission.json', 'rt'))
                batch_obj = client.batches.retrieve(submission_info['id'])
                print(batch_obj)
                if batch_obj.output_file_id is not None:
                    file_response = client.files.content(batch_obj.output_file_id)
                    with open(f'{self.base_id}/result_submission.json', 'wt') as f:
                        f.write(file_response.text)
                    print(f"Check result in: {self.base_id}/result_submission.json")

                if batch_obj.error_file_id is not None:
                    file_response = client.files.content(batch_obj.error_file_id)
                    with open(f'{self.base_id}/err_log_submission.log', 'wt') as f:
                        f.write(file_response.text)
            else:      
                print(f"Check result in: {self.base_id}/result_submission.json")
                
            self.conbine_openai_result()
                
    def conbine_openai_result(self):
        file_result  = f'{self.base_id}/result_submission.json'
        cleaned_result = f'{self.run_id}'
        if os.path.exists(file_result):
            print(f'- conbine_openai_result => {cleaned_result}')
            combined_results = []
            with open(f'{self.base_id}/query_data.json') as f:
                input_data = json.load(f)
            with open(f'{self.base_id}/result_submission.json') as f:
                shuffle_results = [json.loads(e.strip()) for e in f.readlines()]
                results = [None]*len(input_data)
                for e in shuffle_results:
                    results[int(e['custom_id'])] = e
            count_err = 0
            for idx_s, (input_info, result_openai) in enumerate(zip(input_data, results)): 
                if result_openai == None:
                    print(f"- Can not take result at sample: {idx_s}")
                    count_err += 1
                    continue
                else:
                    pred = result_openai['response']['body']['choices'][0]['message']['content']
                combined_results.append({
                    "id": input_info['id'],
                    'label': input_info['label'],
                    "prompt": input_info['prompting'],
                    "generated_output": pred,
                    "correct": None,
                    "err_format": None,
                }) 
               
            with open(cleaned_result, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, ensure_ascii=False, indent=1) 
            
            eval_func = evaluate_simple_lg_logicdeduction if ("logicdeduction" in cleaned_result and 'logiccot' in cleaned_result ) else \
                    evaluate_simple_lg if 'logiccot' in cleaned_result else \
                    evaluate_json
            data_eval, acc = eval_func(cleaned_result)
            with open(cleaned_result, "w", encoding="utf-8") as f:
                json.dump(data_eval, f, ensure_ascii=False, indent=1)

            print("- Count err = ", count_err, len(combined_results))
         

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', type=str, default="gpt4")
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--note',  type=str, default=None)
    
    args, unknown = parser.parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     
    # generate training data for LLM   
    querier:LLMBaseQuery = LLMBaseQuery(args)

    querier.prepare_openai_query( )