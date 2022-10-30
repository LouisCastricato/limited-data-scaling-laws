# this file uses trlx and PPO to apply RLHF to LM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import trlx

from ppo_utils import ELOCriticModel, elo_schedule
from typing import Any, List

# Base class for the critic model
class SentimentELOCritic(ELOCriticModel):
    def __init__(self, model, tokenizer, prompt_dir):
        super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer

        # load dataframe from prompt_dir
        self.df = pd.read_csv(prompt_dir, sep=",") 

    def get_prompt(self, input_prompt : str, option_a : str, option_b : str, k : int = 10) -> str:
        """
        input_prompt: string
        option_a: a string outlining the first option for the multiple choice question
        option_b: a string outlining the second option for the multiple choice question
        k: number of examples to use in the prompt.
        Gets a random prompt from the dataframe. Construct a prompt from k examples. Appends a blank example at the end.
        """
        # randomly choose k
        examples = self.df.sample(k)

        # for each example, take prompt and append answer
        prompts = []
        for i in range(len(examples)):
            prompt = examples.iloc[i]['prompt']
            answer = examples.iloc[i]['answer']
            prompts.append(str(i+1) + ") " + prompt + " " + answer)
        input_instructions = "Below is a set of product names and two reviews for each product. Pick the review which is more positive about the product.\n"
        # add the input prompt and options
        if k > 0:
            prompt = input_instructions + "\n".join(prompts) +"\n" + str(len(examples)+1) + ") Product: " + input_prompt + "\nReview A: " + option_a +\
            "\nReview B: " + option_b + "\nWhich review is more positive about Product, A or B?"
        else:
            input_instructions = "Which review is more positive about "
            prompt = input_instructions + input_prompt + "?\nReview A: " + option_a +\
            "\nor\nReview B: " + option_b + "\nAnswer either A or B."
        return prompt

# Accomodates GPT critic models (AR)
class GPTSentimentELOCritic(SentimentELOCritic):
    def __init__(self, model, tokenizer, prompt_dir):
        super().__init__(model, tokenizer, prompt_dir)
        # set the pad token
        self.tokenizer.pad_token = "<|endoftext|>"

        # choices for multiple choice questions
        self.option_tokens = self.tokenizer([" A", " B"])['input_ids']

    @torch.no_grad()
    def match_function(self, prior : str, player1 : List[str], player2 : List[str]) -> List[int]:
        """
        prior: string for the product name
        player1: list of strings for the first option
        player2: list of strings for the second option
        Returns a list of 0s and 1s, where 1 corresponds to player1 and 0 corresponds to player2.
        """
        bs = len(player1)
        # get our input prompts
        input_prompt = [self.get_prompt(prior, player1[idx], player2[idx]) for idx in range(bs)]
        # tokenize input prompts
        input_prompt = self.tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to("cuda")

        # for the input prompt, find the length per prompt by summing the attention mask
        input_prompt_len = torch.sum(input_prompt['attention_mask'], dim=1)
        # forward pass through the model
        output = self.model(input_ids=input_prompt['input_ids'], attention_mask=input_prompt['attention_mask'])
        # get the last logit for each prompt
        output = output.logits[torch.arange(bs).to("cuda"), input_prompt_len-1]

        # get the logits for the options
        option_a_logits = output[:, self.option_tokens[0]]
        option_b_logits = output[:, self.option_tokens[1]]

        # softmax over the logits
        option_a_b_probs = torch.softmax(torch.stack([option_a_logits, option_b_logits], dim=1), dim=1)
        
        # take argmax over the probabilities
        option_a_b_argmax = torch.argmax(option_a_b_probs, dim=1).squeeze()

        # return the argmax
        return option_a_b_argmax.cpu().tolist()


# Accomodates T5 critic models (seq2seq)
class T5SentimentELOCritic(SentimentELOCritic):
    def __init__(self, model, tokenizer, prompt_dir):
        super().__init__(model, tokenizer, prompt_dir)

        # choices for multiple choice questions
        self.option_tokens = self.tokenizer([" A", " B"])['input_ids']
        self.option_tokens = list(map(lambda x: x[0], self.option_tokens))

    @torch.no_grad()
    def match_function(self, prior : str, player1 : List[str], player2 : List[str]) -> List[int]:
        """
        prior: string for the product name
        player1: list of strings for the first option
        player2: list of strings for the second option
        Returns a list of 0s and 1s, where 1 corresponds to player1 and 0 corresponds to player2.
        """
        bs = len(player1)
        # construct decoder_input_ids from EOD token
        decoder_input_ids = torch.ones((bs, 1), dtype=torch.long, device="cuda") * self.model.config.decoder_start_token_id

        # get our input prompts
        input_prompt = [self.get_prompt(prior, player1[idx], player2[idx], k=0) for idx in range(bs)]
        # tokenize input prompts
        input_prompt = self.tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to("cuda")

        # for the input prompt, find the length per prompt by summing the attention mask
        input_prompt_len = torch.sum(input_prompt['attention_mask'], dim=1)

        # forward pass through the model
        output = self.model(input_ids=input_prompt['input_ids'], 
        attention_mask=input_prompt['attention_mask'], decoder_input_ids=decoder_input_ids)

        # get the last logit for each prompt
        logits = output.logits.view(bs, -1)

        # get the logits for the options
        option_a_logits = logits[:, self.option_tokens[0]]
        option_b_logits = logits[:, self.option_tokens[1]]

        # softmax over the logits
        option_a_b_probs = torch.softmax(torch.stack([option_a_logits, option_b_logits], dim=1), dim=1)
        
        # take argmax over the probabilities
        option_a_b_argmax = torch.argmax(option_a_b_probs, dim=1).squeeze()

        # return the argmax
        return option_a_b_argmax.cpu().tolist()
