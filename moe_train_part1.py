
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
runname="moe-part1-f1"
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
# load tsv
total=[]
for file in glob.glob("./o2chcorpus/*.tsv"):
    with open(file,"r",encoding="utf-8",errors="ignore") as f:
        # split
        for line in f:
            spr=line.split("\t")
            if len(spr)<2:
                continue
            # get head
            msg=spr[0]
            resp=spr[1]
            if len(msg)<80:
                continue
            # __BR__ to \n
            msg=msg.replace("__BR__","\n")
            resp=resp.replace("__BR__","\n")
            total.append({"text":f"<|user|>{msg}</s>\n<|assistant|>","text_target":resp+"</s>"})
print(len(total))
ds=Dataset.from_list(total)



dds=ds.train_test_split(test_size=0.025,shuffle=True)

noja_map=dds["train"]
noja_eval_map=dds["test"]

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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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

