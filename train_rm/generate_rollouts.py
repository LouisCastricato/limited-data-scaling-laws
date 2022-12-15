import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from tqdm import tqdm 
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

def load_generator():
    # load generator model
    generator = AutoModelForCausalLM.from_pretrained("../finetuned_student_model/")
    generator_tokenizer = AutoTokenizer.from_pretrained("../finetuned_student_model/")

    # set eos and pad
    generator_tokenizer.eos_token = "<|endoftext|>"
    generator_tokenizer.eos_token_id = 0

    generator_tokenizer.pad_token = "<|padding|>"
    generator_tokenizer.pad_token_id = 1

    # cast to fp16
    #generator = generator.half()

    return generator, generator_tokenizer

def filter_outputs(string):
    # filters the outputs of a rollout on a set of conditions
    # first we need to parse out the review
    try:
        review = string.split("Product review: ")[1]
    except:
        # if we can't parse out the review, return False
        return False
    # 1. length
    if len(review) < 10:
        return False
    # 2. contains a word that is atleast 3 characters
    if not any([len(word) >= 3 for word in review.split()]):
        return False
    # 3. contains english letters
    if not any([word.isalpha() for word in review.split()]):
        return False

    return True

class RolloutDataset(Dataset):
    def __init__(self, dir, tokenizer):
        df = pd.read_csv(dir)
        # convert to list & map
        def correct_string(string):
            return "Product name: " + string +" Product review: "
        self.prompts = list(map(correct_string, df["product"].tolist()))
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        out = self.tokenizer(self.prompts[idx], return_tensors="pt", padding=False, truncation=False)
        # remove the last token
        out["input_ids"] = out["input_ids"][:,:-1].squeeze()
        out["attention_mask"] = out["attention_mask"][:,:-1].squeeze()
        return out

def generate_rollouts():
    mbs = 1
    N = 1


    print("Loading models...")
    gen, gen_tokenizer = load_generator()
    
    # move to cuda and set to eval mode
    gen = gen.cuda()
    gen.eval()

    # load the dataset
    dataset = RolloutDataset("../datasets/product_names.csv", gen_tokenizer)
    collator = DataCollatorWithPadding(return_tensors="pt", padding=True, tokenizer=gen_tokenizer)
    dataloader = DataLoader(dataset, batch_size=mbs, collate_fn=collator)

    # generate rollouts. for every item in prompts, generate N samples using top-p, top-k

    rollouts = []
    print("Generating rollouts...")
    # get the id for \n
    newline_id = gen_tokenizer.encode("\n")[0]

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(gen.device) for k, v in batch.items()}
        for _ in range(N):
            model_out = gen.generate(**batch, eos_token_id=newline_id, pad_token_id=1,
            early_stopping=True, repetition_penalty=1.2, max_new_tokens=50, no_repeat_ngram_size=2, length_penalty=0.9)
            # decode 
            model_out = gen_tokenizer.batch_decode(model_out, skip_special_tokens=True)
            # filter
            model_out = list(filter(filter_outputs, model_out))
            # append
            rollouts.extend(model_out)
    
    #save rollouts
    with open("rollouts.txt", "w") as f:
        f.write("\n\n".join(rollouts))


if __name__ == "__main__":
    generate_rollouts()
