
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
runname="moe-jacc-f1"
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
    r=256,
    lora_alpha=512,
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

def Tokenize(ex):
    return tokenizer(ex["text"],text_target=ex["text"], padding="max_length", truncation=True, max_length=2048)


ds1=load_dataset("mmnga/wikipedia-ja-20230720-100k",split="train")
ds3=load_dataset("izumi-lab/wikinews-ja-20230728",split="train")
dataset=concatenate_datasets([ds1,ds3]).shuffle(42).select(range(10000))

tokenized_dataset=dataset.map(Tokenize,batched=False,batch_size=1,num_proc=4)
tokenized_datasets=tokenized_dataset.train_test_split(test_size=0.02)
tokenized_dataset=tokenized_datasets["train"]
tokenized_eval_dataset=tokenized_datasets["test"]
ta=TrainingArguments(
    run_name=runname,
    report_to="azure_ml",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=100,
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
    save_steps=100,
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
    data_collator=DataCollatorForLanguageModeling(
        tokenizer, return_tensors="pt", mlm=False,
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

