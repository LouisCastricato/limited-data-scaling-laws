# this file uses trlx and PPO to apply RLHF to LM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import trlx

from ppo_utils import ELOCriticModel, elo_schedule
from typing import Any, List



class SentimentELOCritic(ELOCriticModel):
    def __init__(self, model, tokenizer, prompt_dir):
        super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer

        # set the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # load dataframe from prompt_dir
        self.df = pd.read_csv(prompt_dir, sep=",")

        # choices for multiple choice questions
        self.option_tokens = self.tokenizer(["A", "B"])['input_ids']

    def get_prompt(self, input_prompt : str, option_a : str, option_b : str, k : int = 5) -> str:
        """
        input_prompt: string
        option_a: a string outlining the first option for the multiple choice question
        option_b: a string outlining the second option for the multiple choice question
        k: number of examples to use in the prompt.
        Gets a random prompt from the dataframe. Construct a prompt from k examples. Appends a blank example at the end.
        """
        # choose k random examples from the dataframe
        examples = self.df.sample(k)

        # for each example, take prompt and append answer
        prompts = []
        for i in range(len(examples)):
            prompt = examples.iloc[i]['prompt']
            answer = examples.iloc[i]['answer']
            prompts.append(prompt + " " + answer)

        # add the input prompt and options
        prompt = "\n".join(prompts) + "\nProduct: " + input_prompt + "\nReview A: " + option_a +\
        "\nReview B: " + option_b + "\nWhich review is more positive about Product, A or B?"

        return prompt

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
        output = self.model(**input_prompt)
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
        return (1 - option_a_b_argmax).cpu().tolist()



if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_dir = "prompts_shorter.csv"

    critic_model = SentimentELOCritic(model, tokenizer, prompt_dir)
    
    # test 
    critic_model.get_prompt("This is a test", "This is option A", "This is option B")
    critic_model.match_function("This is a prior", 
    ["This is a test A", "This is a test A2"], 
    ["This is a test B", "This is a test B2"])

    # curry to make a static function
    def match_function(prior, player1, player2):
        return critic_model.match_function(prior, player1, player2)

    # test the elo_schedule
    print(elo_schedule("This is a prior", 
    ["This is a test A", "This is a test A2", "This is a test B", "This is a test B2"], 
    match_function))