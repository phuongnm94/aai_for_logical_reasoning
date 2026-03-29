## Improving Chain-of-Thought for Logical Reasoning via Attention-Aware Intervention 
[![gitcgr](https://gitcgr.com/badge/phuongnm94/aai_for_logical_reasoning.svg)](https://gitcgr.com/phuongnm94/aai_for_logical_reasoning)

Modern logical reasoning with LLMs primarily relies on employing complex interactive frameworks that decompose the reasoning process into subtasks solved through carefully designed prompts or requiring external resources (e.g., symbolic solvers) to exploit their strong logical structures. While interactive approaches introduce additional overhead or depend on external components, which limit their scalability. In this work, we introduce a non-interactive, end-to-end framework for reasoning tasks, enabling reasoning to emerge within the model itself---improving generalization while preserving analyzability without any external resources. We show that introducing structural information into the few-shot prompt activates a subset of attention heads that patterns aligned with logical reasoning operators. Building on this insight, we propose Attention-Aware Intervention (AAI), an inference-time intervention method that reweights attention scores across selected heads identified by their logical patterns. AAI offers an efficient way to steer the model's reasoning toward leveraging prior knowledge through attention modulation. Extensive experiments show that AAI enhances logical reasoning performance across diverse benchmarks, and model architectures, while incurring negligible additional computational overhead.  

Full paper here: [Improving Chain-of-Thought for Logical Reasoning via Attention-Aware Intervention](https://aclanthology.org/2026.findings-eacl.152/)
 
 

##  Python ENV 
Init python environment 
```cmd
    conda create --prefix=./env_aai_lr/  python=3.9
    conda activate ./env_aai_lr/ 
    pip install -r requirements.txt
```
> Note: Please check the file `package_version.txt` to ensure the re-implementation is correct.
## Run 
1. Init environment follow the above step. 
2. Run  
    Run following command to train a new model. 
    ```bash 
    bash ./bash_scripts/run_exp.sh
    ```
    > **Note**: Please check this scripts to check the setting and choose which data and setting you want to run. 
    
 
## Citation 
This work:
```bibtex
@inproceedings{nguyen-etal-2026-improving,
    title = "Improving Chain-of-Thought for Logical Reasoning via Attention-Aware Intervention",
    author = "Nguyen, Phuong Minh  and
      Huu-Tien, Dang  and
      Inoue, Naoya",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.152/",
    pages = "2917--2941",
    ISBN = "979-8-89176-386-9"
}

``` 

Our previous work, which is strong related:
```bibtex
@inproceedings{nguyen-etal-2025-non,
    title = "Non-Interactive Symbolic-Aided Chain-of-Thought for Logical Reasoning",
    author = "Nguyen, Minh-Phuong  and
      Dang, Tien  and
      Inoue, Naoya",
    editor = "Huang, Chu-Ren  and
      Harada, Yasunari  and
      Kim, Jong-Bok  and
      Huyen, Nguyen T.M.  and
      Huong, Le Thanh  and
      Hien, Pham  and
      Chersoni, Emmanuele  and
      Nguyen, Le Minh  and
      Roxas, Rachel Edita O{\~n}ate  and
      Dita, Sherly",
    booktitle = "Proceedings of the 39th Pacific {A}sia Conference on Language, Information and Computation",
    month = dec,
    year = "2025",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.paclic-1.29/",
    pages = "329--340"
}
``` 
