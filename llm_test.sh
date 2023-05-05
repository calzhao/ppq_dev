#!/bin/bash
for model in decapoda-research/llama-7b-hf decapoda-research/llama-13b-hf decapoda-research/llama-30b-hf
do
    python ./ppq/samples/LLM/llm_eval.py --model $model --task wikitext --method snr --layerwise final >wiki_llama_cosine.out
    if [ $? -ne 0 ]; then
        echo "Error: Python command failed."
        break
    fi
done

for model in facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b
do
    python ./ppq/samples/LLM/llm_eval.py --model $model --task wikitext --method snr --layerwise final >wiki_opt_cosine.out
    if [ $? -ne 0 ]; then
        echo "Error: Python command failed."
        break
    fi
done