# this file uses trlx and PPO to apply RLHF to LM
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import trlx

from critic_models import GPTSentimentELOCritic, T5SentimentELOCritic
from ppo_utils import elo_schedule

if __name__ == "__main__":
    
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.3b-deduped", torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.3b-deduped")
    
    #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.float16).to("cuda")
    #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    prompt_dir = "prompts_reprocessed.csv"
    suffix = "positive"

    critic_model = GPTSentimentELOCritic(model, tokenizer, prompt_dir, suffix=suffix)

    # curry to make a static function
    def match_function(prior, player1, player2):
        return critic_model.match_function(prior, player1, player2)

    # load watchband.txt, line delimited
    continuations = open("watchband.txt", "r").read().splitlines()

    elo_out = elo_schedule(None, continuations, match_function, step_factor=0, tournament_size=5, samples=10, mbs=1)

    # elo_out is a list of dicts with keys "continuation" and "elo", take the last one and the top 5 elos and bottom 5
    filtered = []
    items = elo_out[-1].items()
    # sort items by value
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    # take the top 5 and bottom 5
    top_5 = sorted_items[:5]
    bottom_5 = sorted_items[-5:]
    
    top_5_to_graph = defaultdict(list)
    bottom_5_to_graph = defaultdict(list)
    other_graphs = defaultdict(list)
    top_bottom_keys = list()
    
    # get every top 5 graph
    for k,_ in top_5:
        for time_step in elo_out:
            top_5_to_graph[k].append(time_step[k])
            top_bottom_keys.append(k)

    # get every bottom 5
    for k,_ in bottom_5:
        for time_step in elo_out:
            bottom_5_to_graph[k].append(time_step[k])
            top_bottom_keys.append(k)
    
    # get the remaining keys
    difference_keys = set(elo_out[-1].keys()).difference(set(top_bottom_keys))

    # get these associated graphs
    for k in difference_keys:
        for time_step in elo_out:
            other_graphs[k].append(time_step[k])

    # plot the top 5 and bottom 5. top 5 is green, bottom 5 is red
    for k,v in top_5_to_graph.items():
        plt.plot(v, color="green")
    for k,v in bottom_5_to_graph.items():
        plt.plot(v, color="red")

    # plot the rest of the graphs, alpha = 0.5 and grey and dotted 
    for k,v in other_graphs.items():
        plt.plot(v, color="grey", alpha=0.5, linestyle="dotted")
    
    plt.xlabel("Tournament rounds")
    plt.ylabel("ELO")
    plt.title("Watchband Reviews, " + suffix + " sentiment")

    # set a legend, green is top 5, red is bottom 5, grey is the rest
    handles, labels = plt.gca().get_legend_handles_labels()
    top_5_line = plt.Line2D([0], [0], color="green", label="Top 5")
    bottom_5_line = plt.Line2D([0], [0], color="red", label="Bottom 5")
    other_line = plt.Line2D([0], [0], color="grey", alpha=0.5, linestyle="dotted", label="Other")
    plt.legend(handles=[top_5_line, bottom_5_line, other_line])

    # save to file
    plt.savefig("gpt_negative_sentiment.png")
    print("Top 5:")
    print(top_5)
    print("Bottom 5:")
    print(bottom_5)
