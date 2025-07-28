# deckgym/core/game_components/card.py

from deckgym.core.game_components.rank import Rank
from deckgym.core.game_components.suit import Suit
from typing import Optional

class Card:
    """Represents a single playing card."""
    def __init__(self, suit: Suit, rank: Rank, value: Optional[int] = None):
        self.suit = suit
        self.rank = rank
        # `value` can be game-specific (e.g., 10 for face cards in Blackjack)
        # If None, a default (e.g., rank.value) can be used or calculated later.
        self._value = value

    @property
    def value(self) -> int:
        """Returns the game-specific value of the card."""
        if self._value is not None:
            return self._value
        return self.rank.value # Default to rank value if not specified

    def __str__(self):
        return f"{self.rank.name.capitalize()} of {self.suit.name.capitalize()}"

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name}, value={self._value})"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))