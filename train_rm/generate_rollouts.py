import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from tqdm import tqdm 
import pandas as pd

def load_generator():
    # load generator model
    generator = AutoModelForCausalLM.from_pretrained("../finetuned_student_model/")
    generator_tokenizer = AutoTokenizer.from_pretrained("../finetuned_student_model/")

    # set eos and pad
    generator_tokenizer.eos_token = "<|endoftext|>"
    generator_tokenizer.eos_token_id = 0

    # add special pad token
    generator_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # set max length to 200
    generator.config.max_length = 200

    # cast to fp16
    generator = generator.half()

    return generator, generator_tokenizer

def generate_rollouts():
    # load the dataset
    df = pd.read_csv("../datasets/product_names.csv")
    # convert to list & map
    def correct_string(string):
        return "Product name: " + string +"\nProduct review: "
    prompts = list(map(correct_string, df["product"].tolist()))

    print("Loading models...")
    gen, gen_tokenizer = load_generator()
    # set pad token
    gen.config.pad_token_id = gen_tokenizer.eos_token_id
    
    # move to cuda and set to eval mode
    gen = gen.cuda()
    gen.eval()

    # generate rollouts. for every item in prompts, generate N samples using top-p, top-k
    mbs = 10
    N = 1
    rollouts = []
    print("Generating rollouts...")
    # get the id for \n
    newline_id = gen_tokenizer.encode("\n")[0]

    for prompt_idx in tqdm(range(0, len(prompts), mbs)):
        batch = prompts[prompt_idx:prompt_idx+mbs]
        batch = gen_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(gen.device) for k, v in batch.items()}
        for _ in range(N):
            model_out = gen.generate(**batch,
            early_stopping=True, forced_eos_token_id=0, bad_words_ids=[[13443]])
            # decode 
            model_out = gen_tokenizer.batch_decode(model_out, skip_special_tokens=True)

            # append
            rollouts.extend(model_out)
    
    #save rollouts
    with open("rollouts.txt", "w") as f:
        f.write("\n\n".join(rollouts))


if __name__ == "__main__":
    generate_rollouts()