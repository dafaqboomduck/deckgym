# deckgym/core/game_components/deck.py

from deckgym.core.game_components.rank import Rank
from deckgym.core.game_components.suit import Suit
from deckgym.core.game_components.card import Card
from typing import List
import random




class Deck:
    """Represents a deck of playing cards."""
    def __init__(self, num_decks: int = 1):
        self._cards: List[Card] = []
        self.num_decks = num_decks
        self.reset()

    def reset(self):
        self._cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    self._cards.append(Card(suit, rank)) # Default value will be rank.value

    def shuffle(self):
        random.shuffle(self._cards)

    def deal(self, num_cards: int = 1) -> List[Card]:
        if len(self._cards) < num_cards:
            raise ValueError("Not enough cards in the deck to deal.")
        dealt_cards = [self._cards.pop() for _ in range(num_cards)]
        return dealt_cards

    def __len__(self):
        return len(self._cards)