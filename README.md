# TinyStories

## Files

- data/tinystories.py: Creates a cache of tinystories dataset and saves it to cache_dir defined in config.py.

- models/gpt2.py: Defines a Pytorch Lightning model based on gpt2 that will be trained from scratch.

- scripts/evaluate_gpt2_pretrained.py: evaluates pretrained gpt2 model from huggingface on an LLM.

- utils/llama_together_ai.py: evaluates a given story on Llama LLM using together.ai API.

- utils/tokenizer.py: imports gpt2 tokenizer and adds required special tokens.

## How to run

- python3 -m scripts.evaluate_gpt2_pretrained evaluates the model defined in the said file.

- python3 -m data.tinystories creates cache of the tinystories dataset

- python3 -m models.gpt2 to check the model on an example input and example generation(gibberish for now).
