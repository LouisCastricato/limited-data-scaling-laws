# this file uses trlx and PPO to apply RLHF to LM
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

import trlx
from trlx.data.configs import TRLConfig

from critic_models import GPTSentimentELOCritic, T5SentimentELOCritic
from ppo_utils import elo_schedule

import pandas as pd
from typing import List

if __name__ == "__main__":
    def correct_string(string):
        return "Product name: " + string +"\nProduct review: "

    # load the dataset
    df = pd.read_csv("datasets/product_names.csv")
    # convert to list & map
    prompts = list(map(correct_string, df["product"].tolist()))
   
    # splits 
    eval_prompts = prompts[-int(len(prompts)*0.05):]
    prompts = prompts[:int(len(prompts)*0.95)]

    # load critic model
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").cuda()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    prompt_dir = "datasets/prompts_reprocessed.csv"
    suffix = "positive"
    critic_model = T5SentimentELOCritic(model, tokenizer, prompt_dir, suffix=suffix)

    # curry to make a static function
    def match_function(prior, player1, player2):
        return critic_model.match_function(prior, player1, player2)

    # initialize the reward_fn
    def reward_fn(samples : List[str]) -> List[float]:
        """
        samples: list of strings for the samples
        prior: string for the prior
        Returns a list of rewards for each sample.
        """
        # for each sample, take the text after "Product review: "
        samples = [sample.split("Product review:")[1] for sample in samples]

        # get the match function, No prior
        rewards = torch.tensor(list(elo_schedule(None, samples, match_function)[-1].values()))

        # normalize the scores using std and mean
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

        # return the rewards
        return rewards.tolist()

    # laod TRLConfig
    config = TRLConfig.load_yaml("ppo_config.yml")
    model = trlx.train(
        "finetuned_student_model/",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config
    )

