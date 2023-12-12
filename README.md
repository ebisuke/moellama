# 説明
`TinyLlama/TinyLlama-1.1B-Chat-v0.6`をMoEにして試したものです。
Geforce 2080Tiでとりあえず動くようにしてあります。
# やり方
```
tar xvf o2chcorpus.tar.xz 

python convert_to_moellama.py TinyLlama/TinyLlama-1.1B-Chat-v0.6 moellama_orig --num-experts 1
python moe_train_part1.py
python moe_train_part2.py
python slice_n_splice_experts.py splice moellama_orig/ --expert moe-part1-f1 moe-merged
python slice_n_splice_experts.py splice moe-merged --expert moe-part2-f1 moe-merged
python moe_train_calib.py
python test.py --model moe-finish-f1
```
# データセット
- [おーぷん2ちゃんねる対話コーパス](https://github.com/1never/open2ch-dialogue-corpus)
- databricks/databricks-dolly-15k
- kunishou/databricks-dolly-15k-ja
- shi3z/alpaca_cleaned_ja_json

# ベンチマーク
未実施
# 文献情報

@online{DatabricksBlog2023DollyV2,
    author    = {Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin},
    title     = {Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM},
    year      = {2023},
    url       = {https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm},
    urldate   = {2023-06-30}
}

@inproceedings{open2chdlc2019,
  title={おーぷん2ちゃんねる対話コーパスを用いた用例ベース対話システム},
  author={稲葉 通将},
  booktitle={第87回言語・音声理解と対話処理研究会(第10回対話システムシンポジウム), 人工知能学会研究会資料 SIG-SLUD-B902-33},
  pages={129--132},
  year={2019}
}
