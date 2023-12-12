from transformers import AutoTokenizer,pipeline,TextStreamer
from modeling_moe_llama import LlamaMoEForCausalLM
import argparse
import torch
argp=argparse.ArgumentParser()
argp.add_argument("model",type=str)
argp.add_argument("--device",type=str,default="cuda")
args=argp.parse_args()

model=LlamaMoEForCausalLM.from_pretrained(args.model).to(args.device)
tokenizer=AutoTokenizer.from_pretrained(args.model)

def say(text):
    print(f"----------------\n> {text}")
    input_ids=tokenizer.encode(text,return_tensors="pt")
    _=model.generate(input_ids.to(args.device),do_sample=True,temperature=0.7,max_new_tokens=128,pad_token_id=tokenizer.eos_token_id,
                     streamer=TextStreamer(tokenizer,skip_special_tokens=True,skip_prompt=True),repetition_penalty=1.2)

say("Hello. Nice to meet you. What your name")
say("I am")
say("織田信長は")
say("お前って特に")
say("<|user|>こんにちは。<|assistant|>")
say("<|user|>りんごとみかん、どちらが赤いですか。<|assistant|>")
say("<|user|>お茶は美味しいですか。<|assistant|>美味しいです。あなたはどう思いますか。</s><|user|>美味しいと思います。お菓子はいかがですか？\<|assistant|>")
say("<|user|>Hello. <|assistant|>")
say("<|user|>「基本的な学習をした」\n上記を英語にしてください。<|assistant|>")
say("<|user|>お前なんなんだよ！<|assistant|>")
