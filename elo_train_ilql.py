# this file uses trlx and PPO to apply RLHF to LM
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

import trlx
from trlx.data.configs import TRLConfig

import pandas as pd
from typing import Dict, List
import yaml

def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]

class ReviewRewardDataset(Dataset):
    def __init__(self, data_dir):
        # datadir is a csv with columns ["name", "review", "normalized elo"]
        self.data = pd.read_csv(data_dir)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # get the product name
        name = self.data["name"].iloc[idx]
        # get the review
        review = self.data["review"].iloc[idx]
        return  "Product name: " + name + " Product review: " + review, self.data["normalized elo"].iloc[idx]

def load_validation_set():
    # load finetuning_examples.csv, which is comma delimited
    df = pd.read_csv("datasets/finetuning_examples.csv", sep=",")


    def collate(string):
        split_string = string.split('\n')
        name = split_string[0]
        review = '\n'.join(split_string[1:])

        # append EOT after
        output_string = "Product name: " + name + " Product review: " + review + "\n"
        return  output_string

    # collate over the entire dataset
    dataset = []
    for i in range(len(df)):
        dataset.append(collate(df.iloc[i]['text']))

    # take the last 5% of the dataset for validation
    return dataset[int(len(dataset)*0.95):]


default_config = yaml.safe_load(open("ilql_config.yml"))
def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    # training data
    dataset = ReviewRewardDataset("train_rm/rewards.csv")

    # validation
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,
    )

    def metric_fn(samples: List[str]) -> Dict[str, List[float]]:
        # samples will come in as "Product name: <name> Product review: <review>", we want only the review
        samples = [sample.split("Product review:")[1] for sample in samples]
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return {"sentiments": sentiments}

    # get text and rewards out of dataset
    text, rewards = zip(*dataset)

    trlx.train(
        "finetuned_student_model/",
        dataset=(text, rewards),
        eval_prompts=load_validation_set(),
        metric_fn=metric_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
    