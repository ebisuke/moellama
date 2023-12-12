from transformers import LlamaForCausalLM,AutoTokenizer,LlamaConfig
import argparse
from configuration_moe_llama import LlamaMoEConfig
from modeling_moe_llama import LlamaMoEForCausalLM
argp=argparse.ArgumentParser()
argp.add_argument("model",type=str)
argp.add_argument("output",type=str)
argp.add_argument("--num-experts",type=int,default=2)
argp.add_argument("--num-expert-per-tok",type=int,default=2)
args=argp.parse_args()

print("loading model")
tokenizer=AutoTokenizer.from_pretrained(args.model)
config=LlamaConfig.from_pretrained(args.model)
moeconfig=LlamaMoEConfig(**config.__dict__,num_experts=args.num_experts,num_expert_per_tok=args.num_expert_per_tok)
model=LlamaForCausalLM.from_pretrained(args.model)
moemodel=LlamaMoEForCausalLM(moeconfig)

if moeconfig.num_experts<moeconfig.num_expert_per_tok:
    print("setting num_expert_per_tok to num_experts")
    moeconfig.num_expert_per_tok=moeconfig.num_experts
# copy dict
print("copying state dict")
moemodel.load_state_dict(model.state_dict(),strict=False)

# load to experts
print("loading to experts")

# load mlp layer
layer_id=0
for layer in model.model.layers:
    llayer=moemodel.model.layers[layer_id]
    layer_id+=1
    for r in range(moeconfig.num_experts):
        llayer.experts[r].load_state_dict(layer.mlp.state_dict())
del model

# save
print("saving")
moemodel.save_pretrained(args.output)
tokenizer.save_pretrained(args.output)

print("done")