# this file uses trlx and PPO to apply RLHF to LM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import trlx

from ppo_utils import ELOCriticModel
from typing import Any, List

class SentimentELOCritic(ELOCriticModel):
    def __init__(self, model, tokenizer, prompt_dir):
        super().__init__(model_name, tokenizer, device)
        self.model = model_name
        self.tokenizer = tokenizer

        # load dataframe from prompt_dir
        self.df = pd.read_csv(prompt_dir, sep=",")

        # choices for multiple choice questions
        self.option_tokens = [self.tokenizer(["A", "B"])['input_ids']]

    def get_prompt(self, k=3) -> str:
        """
        k: number of examples to use in the prompt.
        Gets a random prompt from the dataframe. Construct a prompt from k examples. Appends a blank example at the end.
        """
        # get the prompt from the dataframe
        raise NotImplementedError
    
    def match_function(player1 : List[Any], player2 : List[Any]) -> List[int]:
        bs = len(player1)

        # get our input prompts
        input_prompt = [self.get_prompt() for _ in range(bs)]
        # tokenize input prompts
        input_prompt = self.tokenizer(input_prompt, padding=True, truncation=True, reutrn_tensors="pt")
        # for the input prompt, find the length per prompt by summing the attention mask
        input_prompt_len = torch.sum(input_prompt['attention_mask'], dim=1)

        raise NotImplementedError


if __name__ == "__main__":
    pass