#!/bin/bash

python3 -m scripts.krct_prompt_eval 128 8 4
python3 -m scripts.krct_prompt_eval 128 12 4
python3 -m scripts.krct_prompt_eval 256 8 4
python3 -m scripts.krct_prompt_eval 256 12 4
python3 -m scripts.krct_prompt_eval 512 8 4
python3 -m scripts.krct_prompt_eval 512 12 4

python3 -m scripts.krct_prompt_eval 256 8 8
python3 -m scripts.krct_prompt_eval 256 8 2