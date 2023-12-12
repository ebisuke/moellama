
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from transformers import Trainer, TrainingArguments,DataCollatorForSeq2Seq,DataCollatorForLanguageModeling
from datasets import load_dataset,DownloadConfig
from datasets import Dataset,concatenate_datasets
import parse_noja_role as parse_noja
from azureml.core import Workspace, Experiment
from transformers.integrations import AzureMLCallback
from peft import get_peft_model,LoraConfig,PeftModelForCausalLM,prepare_model_for_kbit_training
import jsonlines,unicodecsv,datasets
import os,glob
from modeling_moe_llama import LlamaMoEForCausalLM
from dotenv import load_dotenv
load_dotenv()
model_name="./moellama_orig"
runname="moe-part2-f1"
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank == 0 or local_rank == -1:
    device=f"cuda:{local_rank}"
else:
    device="cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaMoEForCausalLM.from_pretrained(model_name,trust_remote_code=True)
model.gradient_checkpointing_enable()

model.config.use_cache = False
peftconfig=LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    target_modules=[
             'gate_proj','gate',"up_proj","down_proj"
        ]
)


peft_model=model
peft_model=get_peft_model(peft_model,peftconfig)
peft_model.print_trainable_parameters()
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False 



# # merge
dollyen=datasets.load_dataset("databricks/databricks-dolly-15k",split="train").shuffle(42)
dollyen=parse_noja.load_dollyen(dollyen)
dollyja=datasets.load_dataset("kunishou/databricks-dolly-15k-ja",split="train").shuffle(42)
dollyja=parse_noja.load_dollyja(dollyja)
alpacaja=datasets.load_dataset("shi3z/alpaca_cleaned_ja_json",split="train").shuffle(42)
alpacaja=parse_noja.load_dollyja(alpacaja)

noja=dollyen+dollyja+alpacaja
# shuffle
import random
random.shuffle(noja)
noja_winlose=noja[:len(noja)//10]
noja=noja[len(noja)//10:]


def generate_instruct_text_kai(insts:dict)->str:

    rets=[]
    
    for inst in insts:
            content=inst["content"] if "content" in inst else ""
            fncall=inst["fncall"] if "fncall" in inst else ""
            state=inst["state"] if "state" in inst else ""
            emb=inst["emb"] if "emb" in inst else ""
            fncall_def=inst["fncall_def"] if "fncall_def" in inst else ""
            req_fncall=""
            req_state=""
            req_fncall_def=""
            req_emb=""
            if fncall!="":
                req_fncall=' {withfunc}'
                fncall=f"[fn]{fncall}[/fn]"
            if emb!="":
                req_emb=' {withemb}'
                emb=f"[emb]{emb}[/emb]"
            if fncall_def!="":
                req_fncall_def=' {withfndef}'
                fncall_def=f"[fndef]{fncall}[/fndef]"
            if state!="":
                req_state=' {withstate}'
                state=f"[state]{state}[/state]"

            # replace 「」 with 『』
            # orca style
            if inst["role"]=="you":
                rets.append(f'<|assistant|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            elif inst["role"]=="their":
                rets.append(f'<|user|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            elif inst["role"]=="their_behind":
                rets.append(f'<|user_bg|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            elif inst["role"]=="you_behind":
                rets.append(f'<|assistant_bg|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            elif inst["role"]=="a":
                rets.append(f'<|another one|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            else:
                rets.append(f'<|{inst["role"]}|>{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}</s>')
            # if inst["role"]=="you":
            #     rets.append(f'ASSISTANT: {req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}')
            # elif inst["role"]=="their":
            #     rets.append(f'USER: {req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}')
            # else:
            #     rets.append(f'{inst["role"]}: {req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}')

            # # replace 「」 with 『』
            # if inst["role"]=="you":
            #     rets.append(f'<|im_start|>you\n{content}\n{fncall}{state}<|im_end|>')
            # elif inst["role"]=="their":
            #     rets.append(f'<|im_start|>target\n{content}\n{fncall}{state}<|im_end|>')
            # elif inst["role"]=="their_behind":
            #     rets.append(f'<|im_start|>target\'s background\n{content}\n{fncall}{state}<|im_end|>')
            # elif inst["role"]=="you_behind":
            #     rets.append(f'<|im_start|>your background\n{content}\n{fncall}{state}<|im_end|>')
            # elif inst["role"]=="a":
            #     rets.append(f'<|im_start|>another one\n{content}\n{fncall}{state}<|im_end|>')
            # else:
            #     rets.append(f'<|im_start|>{inst["role"]}\n{content}\n{fncall}{state}<|im_end|>')
    
            
    return {"text":""+("\n".join(rets[:-1])),"text_target":"".join(rets[-1:])}
# map
noja_map=list(map(generate_instruct_text_kai,noja))
noja_eval_map=list(map(generate_instruct_text_kai,noja_winlose))  
noja_map=Dataset.from_list(noja_map)
noja_eval_map=Dataset.from_list(noja_eval_map)

# tokenize dataset
def tokenize_function(examples):
    prompt=examples["text"]
    target = examples["text_target"] + tokenizer.eos_token
    input_ids_prompt, input_ids_target = tokenizer([prompt, target],truncation=True,max_length=2048).input_ids
    input_ids = input_ids_prompt + input_ids_target
    labels = input_ids.copy()
    labels[:len(input_ids_prompt)] = [-100] * len(input_ids_prompt)
    return {"input_ids": input_ids, "labels": labels}
# dataset=datasets.Dataset.from_list(noja_map)
# dataset_eval=datasets.Dataset.from_list(noja_eval_map)
dataset=noja_map
dataset_eval=noja_eval_map
tokenized_dataset=dataset.map(tokenize_function,batched=False,batch_size=1,num_proc=4)
tokenized_eval_dataset=dataset_eval.map(tokenize_function,batched=False,batch_size=1,num_proc=4)
ta=TrainingArguments(
    run_name=runname,
    report_to="azure_ml",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    output_dir="ftwip",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    auto_find_batch_size=False,
    gradient_checkpointing=True,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_ratio=0.075,
    lr_scheduler_type='cosine',
    remove_unused_columns=True,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=1000,
    fp16=True,
    logging_steps=2,
    optim="adafactor",
)
    
import os
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank == 0 or local_rank == -1:
    from azureml.core import Workspace
    ws = Workspace(
        subscription_id=os.getenv("AZUREML_SUBSCRIPTION_ID"),
        resource_group=os.getenv("AZUREML_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZUREML_WORKSPACE_NAME"),
    )
    from azureml.core import Experiment
    exp = Experiment(ws, runname)
    run = exp.start_logging(snapshot_directory=None)
else:
    run=None



trainer = Trainer(
    model=peft_model,
    args=ta,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    callbacks=[AzureMLCallback(run)],
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    ),
)
try:
    trainer.train()
finally:
    if run is not None:
        run.complete()
    #trainer.save_model()
    model=peft_model.merge_and_unload()
    model.save_pretrained(runname)
    tokenizer.save_pretrained(runname)

