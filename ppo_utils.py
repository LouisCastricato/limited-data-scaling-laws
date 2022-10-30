from numpy.random import randint
from typing import Any, Callable, List

def expected(score_A, score_B):
    """
    Calculates the expected score of A in a match against B
    score_A: score of player A
    score_B: score of player B
    """
    return 1 / (1 + 10 ** ((score_B - score_A) / 400))

def compute_elo(old, expected, score, k=32):
    """
    Computes the new elo score for a player
    old: old elo score
    expected: expected score of the player
    score: actual score of the player
    k: k-factor
    """
    return old + k * (score - expected)



def elo_schedule(prior : Any,
    players : List[Any], 
    match_function : Callable, 
    player_scores : List[float] = None, 
    samples : int = 20, 
    step_factor : int = 0,
    tournament_size : int = 1,
    mbs : int = 4,
    order : List[int] = None) -> List[Any]:
    """
    prior: prior distribution
    players: list of players
    match_function: function that takes two players and returns a win (1) or loss (0)
    player_scores: list of scores for each player
    samples: number of matches to play per player
    step_factor: if 0, we choose the next player for a matchup, otherwise we choose a player up to step_factor away from the current player
    tournament_size: how many assignments per sample (lower bound)
    mbs: number of matches to play at once
    order: list of indices for the order of the players
    returns: a tuple of the players and their scores
    """
    
    # if this is the first time, set the initial scores to 1000 and initialize the order
    if player_scores is None:
        player_scores = [1000] * len(players)
        order = list(range(len(players)))

    # zip players and scores together
    players_and_scores = list(zip(players, player_scores, order))

    # base case
    if samples == 0:
        # Return to the original ordering
        players_and_scores.sort(key=lambda x: x[2])

        # extract just player and scores
        return list(map(list, zip(*players_and_scores)))[:2]

    # if we aren't in the base case, we have another sample to compute.

    # sort by score
    players_and_scores.sort(key=lambda x: x[1], reverse=True)
    
    # unzip
    players_and_scores = list(map(list, zip(*players_and_scores)))
    players = players_and_scores[0]
    player_scores = players_and_scores[1]
    order = players_and_scores[2]


    # Compute the match ups first, then microbatch over them using compute elo.
    pairings = []
    idxs = []
    wins = [0]*len(players)

    for _ in range(tournament_size):
        # get a bunch of pairs of players
        for i in range(len(players)-1):
            # get the first two players
            player1 = players[i]    

            # get the second player, which is either the next player or a random player
            step = randint(1, step_factor+1) if step_factor > 0 else 1
            player2 = players[min(i+step, len(players)-1)]

            # queue up the match
            if i != min(i+step, len(players)-1):
                pairings.append((player1, player2))
                idxs.append((i, min(i+step, len(players)-1)))
    
    # play the matches, using mbs
    for i in range(0, len(pairings), mbs):

        # transpose so we get mbs_player1, mbs_player2
        mbs_player1 = [match[0] for match in pairings[i:i+mbs]]
        mbs_player2 = [match[1] for match in pairings[i:i+mbs]]

        # get the idxs to update the player scores
        mbs_idxs1 = [idx[0] for idx in idxs[i:i+mbs]]
        mbs_idxs2 = [idx[1] for idx in idxs[i:i+mbs]]

        # play the matches (This is the expensive part). Do both ways to account for a tie.
        match_results_pos = match_function(prior, mbs_player1, mbs_player2)
        match_results_neg = match_function(prior, mbs_player2, mbs_player1)
        # average over both samples
        match_results = list(map(lambda x: (x[0] + x[1])/2, zip(match_results_pos, match_results_neg)))

        # record the results
        for idx, match in enumerate(match_results):
            wins[mbs_idxs1[idx]] += 1-match
            wins[mbs_idxs2[idx]] += match

    # update the scores
    for player_i in range(len(players)):
        # get all the matches that player_i played
        played_matches_loc = [i for i, idx in enumerate(idxs) if player_i in idx]
        played_matches_idxs = [idx for i, idx in enumerate(idxs) if player_i in idx]

        # for every played match, change it so that player_i is always idx 0
        played_matches_idxs = [(idx[0], idx[1]) if idx[0] == player_i else (idx[1], idx[0]) for idx in played_matches_idxs]

        # given the played matches, compute the expected ELO
        expected_elo = sum([expected(player_scores[match[0]], player_scores[match[1]]) for match in played_matches_idxs])

        # compute the new ELO
        player_scores[player_i] = compute_elo(player_scores[player_i], expected_elo, wins[player_i])

    # recurse
    return elo_schedule(prior, players, match_function, player_scores, samples - 1, step_factor, order=order, tournament_size=tournament_size)

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