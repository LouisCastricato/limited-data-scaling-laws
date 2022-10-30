from numpy.random import randint
from typing import Any, Callable, List

def compute_elo(elo : float, win : int, k : int = 32) -> float:
    """
    elo: current elo score
    win: 1 if win, 0 if loss
    k: k factor
    """
    return elo + k * (win - 1 / (1 + 10 ** (elo / 400)))


def elo_schedule(prior : Any,
    players : List[Any], 
    match_function : Callable, 
    player_scores : List[float] = None, 
    samples : int = 20, 
    step_factor : int = 0,
    mbs : int = 3,
    order : List[int] = None) -> List[Any]:
    """
    prior: prior distribution
    players: list of players
    match_function: function that takes two players and returns a win (1) or loss (0)
    player_scores: list of scores for each player
    samples: number of matches to play per player
    step_factor: if 0, we choose the next player for a matchup, otherwise we choose a player up to step_factor away from the current player
    mbs: number of matches to play at once
    order: list of indices for the order of the players
    returns: a tuple of the players and their scores
    """
    
    # if this is the first time, set the initial scores to 1000
    if player_scores is None:
        player_scores = [1000] * len(players)
        order = list(range(len(players)))

    # zip players and scores together
    players_and_scores = list(zip(players, player_scores, order))

    # sort by score
    players_and_scores.sort(key=lambda x: x[1], reverse=True)
    
    # unzip
    players_and_scores = list(map(list, zip(*players_and_scores)))
    players = players_and_scores[0]
    player_scores = players_and_scores[1]
    order = players_and_scores[2]
    
    # base case
    if samples == 0:
        # sort by order
        players_and_scores = list(zip(players, player_scores, order))
        players_and_scores.sort(key=lambda x: x[2])
        # extract just player and scores
        players_and_scores = list(map(list, zip(*players_and_scores)))
        players = players_and_scores[0]
        player_scores = players_and_scores[1]
        return [players, player_scores]

    # Compute the match ups first, then microbatch over them using compute elo.
    pairings = []
    idxs = []

    # take every sequenic pair of players
    for i in range(len(players)//2):
        # get the first two players
        player1 = players[i*2]    

        # get the second player, which is either the next player or a random player
        step = randint(1, step_factor+1) if step_factor > 0 else 1
        player2 = players[min(i*2+step, len(players)-1)]

        # queue up the match
        pairings.append((player1, player2))
        idxs.append((i*2, min(i*2+step, len(players)-1)))
    
    # play the matches, using mbs
    for i in range(0, len(pairings), mbs):

        # transpose so we get mbs_player1, mbs_player2
        mbs_player1 = [match[0] for match in pairings[i:i+mbs]]
        mbs_player2 = [match[1] for match in pairings[i:i+mbs]]

        # get the idxs to update the player scores
        mbs_idxs1 = [idx[0] for idx in idxs[i:i+mbs]]
        mbs_idxs2 = [idx[1] for idx in idxs[i:i+mbs]]

        # play the matches (This is the expensive part)
        mbs_results = match_function(prior, mbs_player1, mbs_player2)

        # update the scores
        for j in range(len(mbs_results)):
            player_scores[mbs_idxs1[j]] = compute_elo(player_scores[mbs_idxs1[j]], mbs_results[j])
            player_scores[mbs_idxs2[j]] = compute_elo(player_scores[mbs_idxs2[j]], 1 - mbs_results[j])

    # recurse
    return elo_schedule(prior, players, match_function, player_scores, samples - 1, step_factor, order=order)

# The critic model below is a language model that we'll prompt for a single set of logits.
class ELOCriticModel:
    def __init__(self, model, tokenizer):
        """
        model: A hugging face transformer model
        tokenizer: a tokenizer that takes a string and returns a list of ints
        """
        self.model = model
        self.tokenizer = tokenizer
        
    def match_function(self, priors, player1, player2):
        raise NotImplementedError