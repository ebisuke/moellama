
import unicodecsv
import os,datasets
import sys,torch
def load_normal(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # prompt->instruction
    # completion->output
    
    ret=[]
    for row in csv:
        retrow=[]
        retrow.append({
            "role":"their",
            "content":row["prompt"]
        })
        retrow.append({
            "role":"you",
            "content":row["completion"]
        })
        ret.append(retrow)
    return ret
def load_dollyja(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # prompt->instruction
    # completion->output
    
    ret=[]
    for row in csv:
        retrow=[]
        concatenated=[]
        instruction=row["instruction"]
        if instruction is None:
            instruction=""
        input=row["input"]
        if input is None:
            input=""
        output=row["output"]
        if output is None:
            output=""
            
        retrow.append({
            "role":"their",
            "content":instruction+input
        })
        retrow.append({
            "role":"you",
            "content":output
        })
        ret.append(retrow)
    return ret
def load_dollyen(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # prompt->instruction
    # completion->output
    
    ret=[]
    for row in csv:
        retrow=[]
        concatenated=[]

        instruction=row["instruction"]
        if instruction is None:
            instruction=""
            
        context=row["context"]
        if context is None:
            context=""
        
        response=row["response"]
        if response is None:
            response=""
        
        retrow.append({
            "role":"their",
            "content":instruction+context
        })
        retrow.append({
            "role":"you",
            "content":response
        })
        ret.append(retrow)
    return ret
def load_winlose(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # prompt->instruction
    # winning_completion->output
    
    ret=[]
    for row in csv:
        retrow=[]
        retrow.append({
            "role":"their",
            "content":row["prompt"]
        })
        retrow.append({
            "role":"you",
            "content":row["winning_completion"]
        })
        ret.append(retrow)
    return ret
def load_conversation(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # conversation->(nested)
    # last_answer->output
    
    ret=[]
    for row in csv:
        retrow=[]
        # last_answer
        last_answer=row["last_answer"]

        spr=row["conversation"].split("\n")
        their=None
        you=None
        for line in spr:
            if line.startswith("t:"):
                retrow.append({
                    "role":"their",
                    "content":line[2:]
                })
                # skip
            elif their is not None and line.startswith("y:"):
                retrow.append({
                    "role":"you",
                    "content":line[2:]
                })
                # skip
        retrow.append({
                    "role":"you",
                    "content":last_answer
                })
        ret.append(retrow)
    return ret
def load_deepthink_conversation(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # conversation->(nested)
    # last_answer->output
    
    ret=[]
    for row in csv:
        retrow=[]
        # last_answer
        last_answer=row["last_answer"]
        if row["background"]!="":
            retrow.append({
                "role":"their_behind",
                "content":row["background"]
            })
        spr=row["conversation"].split("\n")
        their=None
        you=None
        for line in spr:
            if line.startswith("t:"):
                retrow.append({
                    "role":"their",
                    "content":line[2:]
                })
                # skip
            elif their is not None and line.startswith("y:"):
                retrow.append({
                    "role":"you",
                    "content":line[2:]
                })
                # skip
        retrow.append({
                    "role":"you",
                    "content":last_answer
                })
        ret.append(retrow)
    return ret
def load_precondition(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # conversation->(nested)
    # last_answer->output
    
    ret=[]
    for row in csv:
        retrow=[]
        # last_answer

        if row["precondition"]!="":
            retrow.append({
                "role":"you_behind",
                "content":row["precondition"]
            })
        spr=row["history"].split("\n")
        their=None
        you=None
        for line in spr:
            if line.startswith("t:"):
                retrow.append({
                    "role":"their",
                    "content":line[2:]
                })
                # skip
            elif their is not None and line.startswith("y:"):
                retrow.append({
                    "role":"you",
                    "content":line[2:]
                })
            elif their is not None and line.startswith("a:"):
                retrow.append({
                    "role":"a",
                    "content":line[2:]
                })
                # skip
        if row["prompt"] != "":
            retrow.append({
                        "role":"their",
                        "content":row["prompt"]+("\n"+row["input"] if row["input"]!="" else "")
                    })
        retrow.append({
                    "role":"you",
                    "content":row["response"]
                })
        ret.append(retrow)
    return ret
  
def load_instruction(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # instruction->instruction
    # input->input
    # response->output
    
    ret=[]
    for row in csv:
        retrow=[]
       
        retrow.append({
            "role":"their",
            "content":f"{row['instruction']}\n{row['input']}"
        })
        retrow.append({
            "role":"you",
            "content":row["response"]
        })
        ret.append(retrow)
    return ret
def load_repetitive(csv:list[dict])->list[list[dict]]:
    # (original)->(result)
    # prompt->instruction
    # history->input
    # response->output
    
    ret=[]
    for row in csv:
        retrow=[]
    
        retrow.append({
            "role":"their",
            "content":f"『{row['history'][2:]}』"+row["prompt"]
        })
        retrow.append({
            "role":"you",
            "content":row["response"]
        })
        ret.append(retrow)
    return ret
def load_fncall(csv:list[dict])->list[list[dict]]:
    # (original)->(result)

    
    ret=[]

    for row in csv:
        retrow=[]
        if "preprompt" in row:
            retrow.append({
                "role":"system",
                "fncall_def":row["preprompt"] if "preprompt" in row else None
            })
        if "prompt" in row:
            retrow.append({
                "role":"their",
                "content":row["prompt"]+(("\n"+row["input"]) if "input" in row else "")
            })
        if "intermediate_response" in row:
            retrow.append({
                "role":"you",
                "content":row["intermediate_response"],
                "fncall":row["intermediate_script"] if "intermediate_script" in row else None
            })
        if "intermediate_text" in row:
            retrow.append({
                "role":"their" if row["intermediate_text_talker"]=="t" else "system",
                "content":row["intermediate_text"]
            })
    
        retrow.append({
            "role":"you",
            "content":row["response_text"] if "response_text" in row else None,
            "fncall":row["response_script"] if "response_script" in row else None,
        })
        ret.append(retrow)
    return ret
def load_state(csv:list[dict])->list[list[dict]]:
    # (original)->(result)

    
    ret=[]

    for row in csv:
        retrow=[]

        retrow.append({
            "role":"you",
            "state":row["before_state"] if "before_state" in row else "",
        })
    
        retrow.append({
            "role":"their" if row["prompt_talker"]=="t" else "system",
            "content":row["prompt"]
        })
        retrow.append({
            "role":"you",
            "content":row["completion"],
            "state":row["after_state"] if "after_state" in row else "",
        })
        
        ret.append(retrow)
    return ret
def load_emb(csv:list[dict])->list[list[dict]]:
    # (original)->(result)

    
    ret=[]

    for row in csv:
        retrow=[]

        retrow.append({
            "role":"you",
            "emb":row["before_emb"] if "before_emb" in row else "",
        })
    
        retrow.append({
            "role":"their" if row["prompt_talker"]=="t" else "system",
            "content":row["prompt"]
        })
        retrow.append({
            "role":"you",
            "content":row["completion"],
            "emb":row["after_emb"] if "after_emb" in row else "",
        })
        
        ret.append(retrow)
    return ret
def load_openai(jsonl:list[dict])->list[list[dict]]:
    # (original)->(result)

    
    ret=[]

    for row in jsonl:
        retrow=[]
        for msg in row['messages']:
            if msg['role']=="user":
                retrow.append({
                    "role":"you",
                    "content":msg["content"]
                })
            elif msg['role']=="assistant":
                retrow.append({
                    "role":"their",
                    "content":msg["content"]
                })
        ret.append(retrow)
    return ret
def load_vip(rows:list[str])->list[list[dict]]:
    # (original)->(result)

    
    ret=[]

    for row in rows:
        # replace __BR__ with \n
        row=row.replace("__BR__","\n")
        spr=row.split("\t")
        # when length is even, first turn is user
        turn='you'
        if len(spr)%2==0:
            turn = 'their'
        retrow=[]
        for i in range(len(spr)):
            retrow.append({
                "role":turn,
                "content":spr[i]
            })
            if turn=='you':
                turn='their'
            else:
                turn='you'
        ret.append(retrow)
    return ret

def generate_instruct_text(insts:dict)->str:

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
                fncall=f"<|func|>\n{fncall}\n<|/func|>\n"
            if emb!="":
                req_emb=' {withemb}'
                emb=f"<|emb|>\n{emb}\n<|/emb|>\n"
            if fncall_def!="":
                req_fncall_def=' {withfndef}'
                fncall_def=f"<|fndef|>\n{fncall}\n<|/fndef|>\n"
            if state!="":
                req_state=' {withstate}'
                state=f"<|state|>\n{state}\n<|/state|>\n"

            # replace 「」 with 『』
            if inst["role"]=="you":
                rets.append(f'<|im_start|>you{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="their":
                rets.append(f'<|im_start|>target{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="their_behind":
                rets.append(f'<|im_start|>target\'s background{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="you_behind":
                rets.append(f'<|im_start|>your background{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="a":
                rets.append(f'<|im_start|>another one{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            else:
                rets.append(f'<|im_start|>{inst["role"]}{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')

    return {"text":"<|startoftext|>"+("\n".join(rets))+"<|endofgeneration|><|endoftext|>"}
def generate_test_text(insts:dict)->str:

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
                fncall=f"<|func|>\n{fncall}\n<|/func|>\n"
            if emb!="":
                req_emb=' {withemb}'
                emb=f"<|emb|>\n{emb}\n<|/emb|>\n"
            if fncall_def!="":
                req_fncall_def=' {withfndef}'
                fncall_def=f"<|fndef|>\n{fncall}\n<|/fndef|>\n"
            if state!="":
                req_state=' {withstate}'
                state=f"<|state|>\n{state}\n<|/state|>\n"

            # replace 「」 with 『』
            if inst["role"]=="you":
                rets.append(f'<|im_start|>you{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="their":
                rets.append(f'<|im_start|>target{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="their_behind":
                rets.append(f'<|im_start|>target\'s background{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="you_behind":
                rets.append(f'<|im_start|>your background{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            elif inst["role"]=="a":
                rets.append(f'<|im_start|>another one{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')
            else:
                rets.append(f'<|im_start|>{inst["role"]}{req_fncall}{req_fncall_def}{req_emb}{req_state}\n{content}\n{fncall_def}{fncall}{emb}{state}<|im_end|>')

    return {"text":"<|startoftext|>"+("\n".join(rets))+"<|endofgeneration|><|endoftext|>"}

def prepare_ds(csv_dir,tokenizer=None):
    
    # load nojaloli
    with open(os.path.join(csv_dir,"nojaloli.csv"),"rb") as f:
        noja_normal=unicodecsv.DictReader(f)
        noja_normal=load_normal(noja_normal)

    # load nojaloli_conversation
    with open(os.path.join(csv_dir,"nojaloli_conversation.csv"),"rb") as f:
        noja_conversation=unicodecsv.DictReader(f)
        noja_conversation=load_conversation(noja_conversation)

    # load nojaloli_longconversation
    with open(os.path.join(csv_dir,"nojaloli_longconversation.csv"),"rb") as f:
        noja_longconversation=unicodecsv.DictReader(f)
        noja_longconversation=load_conversation(noja_longconversation)
    with open(os.path.join(csv_dir,"nojaloli_control.csv"),"rb") as f:
        noja_control=unicodecsv.DictReader(f)
        noja_control=load_conversation(noja_control)
    with open(os.path.join(csv_dir,"nojaloli_deepthink.csv"),"rb") as f:
        noja_deepthink=unicodecsv.DictReader(f)
        noja_deepthink=load_deepthink_conversation(noja_deepthink)
    with open(os.path.join(csv_dir,"nojaloli_precondition.csv"),"rb") as f:
        noja_precondition=unicodecsv.DictReader(f)
        noja_precondition=load_precondition(noja_precondition)

    # load nojaloli_winlose
    with open(os.path.join(csv_dir,"nojaloli_winlose.csv"),"rb") as f:
        noja_winlose=unicodecsv.DictReader(f)
        noja_winlose=load_winlose(noja_winlose)

    # load nojaloli_instruct
    with open(os.path.join(csv_dir,"nojaloli_instruct.csv"),"rb") as f:
        noja_instruct=unicodecsv.DictReader(f)
        noja_instruct=load_instruction(noja_instruct)

    # load nojaloli_repetitive
    with open(os.path.join(csv_dir,"nojaloli_repetitive.csv"),"rb") as f:
        noja_repeat=unicodecsv.DictReader(f)
        noja_repeat=load_repetitive(noja_repeat)

    # load nojaloli_longsentence
    with open(os.path.join(csv_dir,"nojaloli_longsentence.csv"),"rb") as f:
        noja_longsentence=unicodecsv.DictReader(f)
        noja_longsentence=load_normal(noja_longsentence)
    # load nojaloli_state
    with open(os.path.join(csv_dir,"nojaloli_state.csv"),"rb") as f:
        noja_state=unicodecsv.DictReader(f)
        noja_state=load_state(noja_state)
    # load nojaloli_emb
    with open(os.path.join(csv_dir,"nojaloli_emb.csv"),"rb") as f:
        noja_emb=unicodecsv.DictReader(f)
        noja_emb=load_emb(noja_emb)
    # load nojaloli_fncall
    with open(os.path.join(csv_dir,"nojaloli_fncall.csv"),"rb") as f:
        noja_fncall=unicodecsv.DictReader(f)
        noja_fncall=load_fncall(noja_fncall)
    dollyen=datasets.load_dataset("databricks/databricks-dolly-15k",split="train")
    dollyen=dollyen.map(load_dollyen,batched=True,num_proc=4).shuffle(114514).select(range(1200))
    dollyja=datasets.load_dataset("kunishou/databricks-dolly-15k-ja",split="train")
    dollyja=dollyen.map(load_dollyja,batched=True,num_proc=4).shuffle(114514).select(range(1200))
    
    # merge
    noja=noja_normal+noja_conversation+noja_longconversation+noja_instruct+noja_repeat+ \
        noja_longsentence+noja_precondition+noja_deepthink+noja_control+noja_state+noja_emb+noja_fncall+dollyen+dollyja

    flattened=[]
    # flatten
    for row in noja:
            flattened.append(row)
    flattened_eval=[]
    for row in noja_winlose:
            flattened_eval.append(row)
    # generate text
    # map
    noja_map=list(map(generate_instruct_text,flattened))
    noja_eval_map=list(map(generate_instruct_text,flattened_eval))  

    cut_off=2048
    fillmask=False
    # tokenize dataset
    def tokenize_function(examples):
        # generate text
        withanswer=examples["text"]
        
        

        
        if not fillmask:
            # for numpy 
            full=tokenizer(withanswer,truncation=True,
                max_length=cut_off,
                padding=False)
            
            # if full["input_ids"][-1]!=tokenizer.eos_token_id:
            #     full["input_ids"].append(tokenizer.eos_token_id)
            #     full["attention_mask"].append(1)
            
            # # for torch
            # if full["input_ids"][-1]!=tokenizer.eos_token_id:
            #     full["input_ids"]=torch.cat([full["input_ids"],torch.tensor([tokenizer.eos_token_id])])
            #     full["attention_mask"]=torch.cat([full["attention_mask"],torch.tensor([1])])
            full["labels"]=full["input_ids"].copy()
        else:
            full=tokenizer(withanswer,truncation=True,
                max_length=cut_off,
                padding=False,return_tensors='pt')
            # mask
            inputs =full["input_ids"]

            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, 0.10)

            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)


            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            full["input_ids"]=inputs
            full["labels"]=labels
            # to list
            full["input_ids"]=full["input_ids"].tolist()
            full["labels"]=full["labels"].tolist()
        return full
    dataset=datasets.Dataset.from_list(noja_map)
    dataset_eval=datasets.Dataset.from_list(noja_eval_map)
    if tokenizer :
        tokenized_dataset=dataset.map(tokenize_function,batched=True,batch_size=1,num_proc=torch.get_num_threads())
        tokenized_eval_dataset=dataset_eval.map(tokenize_function,batched=True,batch_size=1,num_proc=torch.get_num_threads())
    else:
        tokenized_dataset=None
        tokenized_eval_dataset=None
    return dataset,dataset_eval,tokenized_dataset,tokenized_eval_dataset