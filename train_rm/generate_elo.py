import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm 
import pandas as pd

import sys
sys.path.append("..")
from critic_models import GPTSentimentELOCritic, T5SentimentELOCritic
from ppo_utils import elo_schedule

def load_elo_model():
    # load critic model
    elo_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").cuda()
    elo_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    prompt_dir = "../datasets/prompts_reprocessed.csv"
    suffix = "positive"
    return T5SentimentELOCritic(elo_model, elo_tokenizer, prompt_dir, suffix=suffix)

def normalize_elos(string_elos):
    """
    Normalize the elos to be in the range [-1, 1]
    args:
        string_elos: a list of tuples (string, elo)
    returns:
        a list of (string, normalized_elo)
    """
    elos = torch.tensor([elo for _, elo in string_elos])
    elos = (elos - torch.mean(elos)) / torch.std(elos)
    elos = elos.tolist()
    
    return [(string, elo) for (string, _), elo in zip(string_elos, elos)]

def generate_elo():
    # load the dataset, which is a txt named rollouts.txt. It is line delimited
    dataset = open("rollouts.txt", "r").readlines()
    # filter out all entries == "\n"
    dataset = [line for line in dataset if line != "\n"]
    product_review = [line.split(" Product review: ")[1] for line in dataset]
    product_names = [line.split("Product name: ")[1].split(" Product review: ")[0] for line in dataset]

    # load the elo model
    elo_model = load_elo_model()

    # curry to make a static function
    def match_function(prior, player1, player2):
        return elo_model.match_function(prior, player1, player2)

    # split the data into chunks of 50
    review_chunks = [product_review[i:i+50] for i in range(0, len(product_review), 50)]
    name_chunks =  [product_names[i:i+50] for i in range(0, len(product_names), 50)]

    finished_chunks = []
    for name, review in tqdm(zip(name_chunks, review_chunks), total=len(review_chunks)):
        # compute Elo
        elo_out = elo_schedule(None, review, match_function, tournament_size=5, samples=10, mbs=40)
        elo_out = elo_out[-1].items()

        items = normalize_elos(elo_out)
        # items is a list of tuples (review, score). convert it to (name, review, score)
        items = [(name, review, score) for (review, score), name in zip(items, name)]
        finished_chunks += items
    # save finsihed chunks, a list of tuple (string, float) to a csv
    df = pd.DataFrame(finished_chunks, columns=["name", "review", "normalized elo"])
    df.to_csv("rewards.csv", index=False)

if __name__ == "__main__":
    generate_elo()
    