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

    def reward_fn(samples : List[str]) -> List[float]:
        """
        samples: list of strings for the samples
        prior: string for the prior
        Returns a list of rewards for each sample.
        """
        # get the match function, No prior
        rewards = elo_schedule(None, samples, match_function)[1]

        # normalize the scores using std and mean
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

        # return the rewards
        return rewards

    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]
    # 95% for train
    train_prompts = prompts[:int(len(prompts)*0.95)]
    # 5% for test
    test_prompts = prompts[int(len(prompts)*0.95):]

    model = trlx.train(
        "gpt2-large",
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=test_prompts
    )
    