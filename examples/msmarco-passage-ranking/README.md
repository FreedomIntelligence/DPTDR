
# Introduction
Here we provide the code for reproducing results for MS-MARCO using deep prompt learning of prompts' length as 128.

## Reproduction
Training: Each line of the the Train file is a training instance,
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
...
```
After preparing the training data, please run `run_s2.lr7e-3.nepochs10.psl128.prefix.sh` using [DPTDR RoBERTa-Large](https://drive.google.com/file/d/1FJWYQ6ycwRLVIc3mPZUP8H7owep5msb5/view?usp=sharing) for training a DPT retriever.
We also provide our trained models in [retriever_model_s2.lr7e-3.nepochs10.psl128.prefix](https://drive.google.com/file/d/1uUaHFwH92fqHq_O8q5lhSXPhSRqWNEYz/view?usp=sharing)

Inference/Encoding: Each line of the the encoding file is a piece of text to be encoded,
```
{text_id: ANY_TYPE, 'text': TEXT_TYPE}
...
```
`TEXT_TYPE` can be either raw string or pre-tokenized ids. `ANY_TYPE` means any type of objects in python.
After preparing the inference data of corpus and query, please run `sh encode_s2.lr7e-3.nepochs10.psl128.prefix.sh`


Search/evaluation:
Please run `sh search_s2.lr7e-3.nepochs10.psl128.prefix.sh; python score_to_marco.py dev.rank.retriever_model_s2.lr7e-3.nepochs10.psl128.prefix.tsv; python $PATH_TO_QRELS/qrels.dev.small.tsv dev.rank.retriever_model_s2.lr7e-3.nepochs10.psl128.prefix.tsv.marco`.

Then you shall get the exact results as follows:
```
#####################
1-RECALL @1000: 0.9886819484240688
MRR @10: 0.3905223541183433
QueriesRanked: 6980
#####################
```

