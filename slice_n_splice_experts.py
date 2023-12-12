import torch
import torch.nn as nn
from configuration_moe_llama import LlamaMoEConfig
from modeling_moe_llama import LlamaMLP,LlamaMoEForCausalLM
from transformers import AutoTokenizer
import argparse
import tqdm
def slice(model,idx=0)->LlamaMoEForCausalLM:
    lconfig=LlamaMoEConfig(**model.config.__dict__)
    lconfig.num_experts=1
    lmodel=LlamaMoEForCausalLM(lconfig)
    sd=model.state_dict()
    # remove experts
    for k in list(sd.keys()):
        if ".gate." in k:
            del sd[k]
        if "experts" in k:
            del sd[k]
    lmodel.load_state_dict(sd,strict=False)
    for i in tqdm.tqdm(range(model.config.num_hidden_layers),unit="layer"):
        # load expert
        lmodel.model.layers[i].experts[0].load_state_dict(model.model.layers[i].experts[idx].state_dict())
        
        
    lmodel.config=lconfig
    return lmodel
    
def splice(model,num_tok_per_expert:int,add_exp,exp_idx=0)->LlamaMoEForCausalLM:
    moeconfig=LlamaMoEConfig(**model.config.__dict__)
    moeconfig.num_experts+=1
    print(f"num_experts={moeconfig.num_experts}")
    moeconfig.num_expert_per_tok=num_tok_per_expert
    if moeconfig.num_expert_per_tok>(moeconfig.num_experts-1):
        print("setting num_expert_per_tok to num_experts-1")
        moeconfig.num_expert_per_tok=moeconfig.num_experts-1
    moemodel=LlamaMoEForCausalLM(moeconfig)
    moemodel.config=moeconfig
    sd=model.state_dict()
    # remove experts
    for k in list(sd.keys()):
        if ".gate." in k:
            del sd[k]
        if "experts" in k:
            del sd[k]
    moemodel.load_state_dict(sd,strict=False)
    layer_id=0
    for layer in tqdm.tqdm(model.model.layers,unit="layer"):
        llayer=moemodel.model.layers[layer_id]
        alayer=add_exp.model.layers[layer_id]
        layer_id+=1
        for r in range(model.config.num_experts):
            llayer.experts[r].load_state_dict(layer.experts[r].state_dict())
        for r in range(model.config.num_experts,moeconfig.num_experts):
            llayer.experts[r].load_state_dict(alayer.experts[0].state_dict())
    moemodel.config=moeconfig
    return moemodel

if __name__ == "__main__":
    argp=argparse.ArgumentParser()
    argp.add_argument("action",type=str)
    argp.add_argument("model",type=str)
    argp.add_argument("output",type=str)
    argp.add_argument("--num-expert-per-tok",type=int,default=2)
    argp.add_argument("--expert",type=str)
    argp.add_argument("--expert-idx",type=int,default=0)
    args=argp.parse_args()
    if args.action=="slice":
        print("loading model")
        model=LlamaMoEForCausalLM.from_pretrained(args.model)
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        print("slicing")
        experts=slice(model,args.expert_idx)
        
        print("saving")
        experts.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
    elif args.action=="splice":
        print("loading model")
        config=LlamaMoEConfig.from_pretrained(args.model)
        model=LlamaMoEForCausalLM.from_pretrained(args.model)
        tokenizer=AutoTokenizer.from_pretrained(args.model)
        experts=LlamaMoEForCausalLM.from_pretrained(args.expert)
        
        print("splicing")
        lmodel=splice(model,args.num_expert_per_tok,experts,args.expert_idx)
        print("saving")
        lmodel.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    print("done")
        
