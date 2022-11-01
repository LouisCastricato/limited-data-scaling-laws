import math
import numpy as np
from numpy.random import randint
from typing import Any, Callable, List

rng = np.random.RandomState(0)

def expected(score_A, score_B):
    """
    Calculates the expected score of A in a match against B
    score_A: score of player A
    score_B: score of player B
    """
    return 1 / (1 + 10 ** ((score_B - score_A) / 400))

def compute_elo(old, expected, score, k=16):
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
    mbs : int = 1,
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

    # base case
    if samples == 0:
        return players, player_scores

    wins = [0] * len(players)

    # Compute the match ups first, then microbatch over them using compute elo.
    pairs = []
    for _ in range(tournament_size):
        # since we can draw many samples, we can match randomly
        idxs = np.arange(len(players))
        rng.shuffle(idxs)

        # in uneven case, remove either first or last participant
        if len(players) & 1:
            idxs = np.delete(idxs, -rng.randint(2))

        pairs.append(idxs.reshape(-1, 2))

    pairs = np.vstack(pairs)
    players1 = [players[pair[0]] for pair in pairs]
    players2 = [players[pair[1]] for pair in pairs]

    # play the matches, using mbs
    results = []
    for i in range(math.ceil(len(pairs)/mbs)):
        batch_ixs = slice(i*mbs, (i+1)*mbs)
        results.extend(match_function(prior, players1[batch_ixs], players2[batch_ixs]))

    # record the results
    for result, (p1, p2) in zip(results, pairs):
        wins[p1] += result
        wins[p2] += 1 - result

    # update elo
    next_player_scores = np.zeros(len(players))
    for pix in range(len(players)):
        opponents = [set(pair).difference({pix}).pop() for pair in pairs if pix in pair]
        expected_score = sum(expected(player_scores[pix], player_scores[opp]) for opp in opponents)
        next_player_scores[pix] = compute_elo(player_scores[pix], expected_score, wins[pix])

    # recurse
    return elo_schedule(prior, players, match_function, next_player_scores, samples - 1, step_factor, tournament_size=tournament_size, mbs=mbs)

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

if __name__ == '__main__':
    # test elo_schedule
    players = np.arange(5)
    match_function = lambda prior, xs1, xs2: (np.array(xs1) > np.array(xs2)).astype(int)
    ranking = elo_schedule(None, players, match_function, mbs=3, tournament_size=10, samples=100)
    xs = sorted(zip(*ranking), key=lambda x: x[1], reverse=True)
    print('ELO for number comparisons:')
    for sample, rating in xs:
        print(f'[{rating:.0f}]', sample)
    assert all(players == np.argsort(ranking[1]))

