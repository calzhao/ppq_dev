#!/bin/bash
for model in decapoda-research/llama-7b-hf decapoda-research/llama-13b-hf decapoda-research/llama-30b-hf
do
    for task in wikitext lambada hellaswag piqa
    do
        python ./ppq/samples/LLM/llm_eval.py --model $model --task $task --method snr --layerwise final >full_llama_cosine.out
        if [ $? -ne 0 ]; then
            echo "Error: Python command $model $task failed."
        fi
    done
done

for model in facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/opt-30b
do
    for task in wikitext lambada hellaswag piqa
    do
        python ./ppq/samples/LLM/llm_eval.py --model $model --task $task --method snr --layerwise final >full_opt_cosine.out
        if [ $? -ne 0 ]; then
            echo "Error: Python command $model $task failed."
        fi
    done
done