#!/bin/bash
for model in decapoda-research/llama-7b-hf decapoda-research/llama-13b-hf decapoda-research/llama-30b-hf
do
    CUDA_VISIBLE_DEVICES='4,5,6,7' python ./ppq/samples/LLM/llm_eval.py --model $model --task wikitext --method cosine >wiki_llama_cosine.out
    if [ $? -ne 0 ]; then
        echo "Error: Python command failed."
        break
    fi
done

for model in facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b
do
    CUDA_VISIBLE_DEVICES='4,5,6,7' python ./ppq/samples/LLM/llm_eval.py --model $model --task wikitext --method cosine >wiki_opt_cosine.out
    if [ $? -ne 0 ]; then
        echo "Error: Python command failed."
        break
    fi
done