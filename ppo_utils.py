from numpy.random import randint
from typing import Any, Callable, List

def compute_elo(elo : float, win : int, k : int = 32) -> float:
    """
    elo: current elo score
    win: 1 if win, 0 if loss
    k: k factor
    """
    return elo + k * (win - 1 / (1 + 10 ** (elo / 400)))


def elo_schedule(players : List[Any], 
    match_function : Callable, 
    player_scores : List[float] = None, 
    samples : int = 100, 
    step_factor : int = 0) -> List[Any]:
    """
    players: list of players
    match_function: function that takes two players and returns a win (1) or loss (0)
    player_scores: list of scores for each player
    samples: number of matches to play per player
    step_factor: if 0, we choose the next player for a matchup, otherwise we choose a player up to step_factor away from the current player
    returns: a tuple of the players and their scores, sorted by score
    """
    
    # if this is the first time, set the initial scores to 1000
    if player_scores is None:
        player_scores = [1000] * len(players)

    # zip players and scores together
    players_and_scores = list(zip(players, player_scores))
    # sort by score
    players_and_scores.sort(key=lambda x: x[1], reverse=True)
    # unzip
    players, player_scores = zip(*players_and_scores)

    # base case
    if samples == 0:
        return players_and_scores

    # TODO(Louis) : Make this batched. Compute the match ups first, then microbatch over them using compute elo. Create a queue.

    # take every sequenic pair of players
    for i in range(len(players)//2):
        # get the first two players
        player1 = players[i*2]    

        # get the second player, which is either the next player or a random player
        rnd_int = randint(1, step_factor+1)
        player2 = players[min(i*2+rnd_int, len(players)-1)]

        # play a match between the two players
        match = match_function(player1, player2, samples)
        # update the player scores
        player_scores[i*2] = compute_elo(player1[1], match[0])
        player_scores[i*2+1] = compute_elo(player2[1], match[1])
    
    # recurse
    return elo_schedule(players, match_function, player_scores, samples - 1, step_factor)
    
# The critic model below is a language model that we'll prompt for a single set of logits.
class ELOCriticModel:
    def __init__(self, model, tokenizer):
        """
        model: A hugging face transformer model
        tokenizer: a tokenizer that takes a string and returns a list of ints
        """
        self.model = model
        self.tokenizer = tokenizer
        
    def match_function(self, player1, player2):
        raise NotImplementedError