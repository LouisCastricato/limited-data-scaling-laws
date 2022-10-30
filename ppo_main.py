# this file uses trlx and PPO to apply RLHF to LM
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import trlx

from critic_models import GPTSentimentELOCritic, T5SentimentELOCritic
from ppo_utils import elo_schedule

if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    prompt_dir = "prompts_shorter.csv"

    critic_model = T5SentimentELOCritic(model, tokenizer, prompt_dir)
    
    # curry to make a static function
    def match_function(prior, player1, player2):
        return critic_model.match_function(prior, player1, player2)

    # test the elo_schedule
    elo_out = elo_schedule("Avatar the last airbender - Complete Series DVD", 
    ["This is quite possibly the best show ever. I'm happy with my purchase.", "I really didn't like this show, there were many issues. I want to return this.",\
     "Possibly the worst show I've ever seen.", "Best purchase ever! I love this show.", "I loved this show as a child, I think it still holds up."], 
    match_function, step_factor=2, tournament_size=3, samples=20)
    print(elo_out)

    def reward_fn(samples : List[str], **kwargs) -> List[float]:
        """
        samples: list of strings for the samples
        prior: string for the prior
        Returns a list of rewards for each sample.
        """
        # get the match function
        rewards = elo_schedule(kwargs['prior'], samples, match_function)[1]

        # normalize the scores. highest elo is 4000, lowest is 1000
        rewards = list(map(lambda x: x/4000.0, scores))

        # return the rewards
        return rewards

    #model = trlx.train(
    #    "finetuned_student_model/",
    #    reward_fn=reward_fn,
    #    prompts=["This is a prior"],
    #    eval_prompts=["This is a different prior"]
    #)
    