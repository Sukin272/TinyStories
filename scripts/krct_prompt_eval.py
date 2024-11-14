import json
import random
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from models.gpt2 import GPT2
from utils.llama_together_ai import evaluate_story,evaluate_prompt
from utils.tokenizer import gpt_neo_tokenizer, gpt2_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, tokenizer, num_stories=10, num_repeats=2):
    with open("data/reasoning_prompts.json", "r") as f:
        prompts = json.load(f)

    # stories = random.sample(stories, num_stories)
    num_stories = len(prompts)
    factual_knowledge = 0

    for orig_story in tqdm(prompts):
        for _ in range(num_repeats):
            st = "Given is a prompt. You should Complete it as you find best."
            ll = len(st)
            story = st + orig_story
            story = story.split()

            while 1:
                temp_story = story[:]
                temp_story = " ".join(temp_story)
                story = temp_story
                break

            input_ids = tokenizer.encode(story, return_tensors="pt").to(device)
            output = model.model.generate(
                input_ids,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            story_for_prompt = story[ll:] + " ***" + output_text[len(story) :]

            # print("story:" , story)
            # print("output_text:", output_text)
            # print("story_for_prompt:", story_for_prompt)
            # return

            eval_msg = evaluate_prompt(story_for_prompt,1)
            time.sleep(10)
            evals = json.loads(eval_msg)

            factual_knowledge += int(evals["reasoning ability"])

    factual_knowledge = factual_knowledge / (num_stories * num_repeats)

    return {
        "reasoning_ability": factual_knowledge,
    }


def main():
    tokenizer = gpt_neo_tokenizer()

    # model = GPT2(tokenizer)
    # checkpoint=torch.load("checkpoints/gpt2_128_12.ckpt")
    # model.load_state_dict(checkpoint["state_dict"])
    # model=model.to(device)

    model = GPT2.load_from_checkpoint(
        "models/gpt2_128_12.ckpt", tokenizer=tokenizer
    ).to(device)

    # prompt = "Once upon a time there was"

    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # output = model.generate(input_ids, max_length = 1000, num_beams=1)
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)
    ret = evaluate_model(model, tokenizer)
    print(ret)
    json.dump(ret, open("data/reasoning ability/gpt2_128_12_fact.json", "w"), indent=4)


if __name__ == "__main__":
    main()
