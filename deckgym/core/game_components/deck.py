# deckgym/core/game_components/deck.py

from deckgym.core.game_components.rank import Rank
from deckgym.core.game_components.suit import Suit
from deckgym.core.game_components.card import Card
from typing import List, Optional
import numpy as np # Using numpy for random operations for better control with seeding
from numpy.random import Generator


class Deck:
    """Represents a deck of playing cards."""
    def __init__(self, num_decks: int = 1, rng: Optional[Generator] = None):
        self._cards: List[Card] = []
        self.num_decks = num_decks
        # Use provided RNG or create a new default one
        self._rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        """Resets the deck to contain the initial number of shuffled cards."""
        self._cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    # For Blackjack, Ace is 11, Face cards are 10. Others are rank.value.
                    # This logic could be externalized if Card values are truly game-specific.
                    # For now, replicating the Blackjack card values here for compatibility.
                    if rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
                        self._cards.append(Card(suit, rank, value=10))
                    elif rank == Rank.ACE:
                        self._cards.append(Card(suit, rank, value=11))
                    else:
                        self._cards.append(Card(suit, rank, value=rank.value))
        self.shuffle() # Shuffle the deck immediately after resetting

    def shuffle(self):
        """Shuffles the deck using the internal random number generator."""
        self._rng.shuffle(self._cards)

    def deal(self, num_cards: int = 1) -> List[Card]:
        """
        Deals (removes and returns) the specified number of cards from the top of the deck.

        Args:
            num_cards (int): The number of cards to deal.

        Returns:
            List[Card]: A list of dealt cards.

        Raises:
            ValueError: If there are not enough cards in the deck to deal.
        """
        if len(self._cards) < num_cards:
            raise ValueError("Not enough cards in the deck to deal.")
        dealt_cards = [self._cards.pop() for _ in range(num_cards)]
        return dealt_cards

    def __len__(self):
        return len(self._cards)

