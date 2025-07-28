# deckgym/core/game_components/deck.py

from deckgym.core.game_components.card import Card
from typing import List, Optional

class Hand:
    """Represents a player's or dealer's hand."""
    def __init__(self, cards: Optional[List[Card]] = None):
        self._cards: List[Card] = cards if cards is not None else []

    def add_card(self, card: Card):
        self._cards.append(card)

    def add_cards(self, cards: List[Card]):
        self._cards.extend(cards)

    def clear(self):
        self._cards = []

    @property
    def cards(self) -> List[Card]:
        return self._cards

    # Game-specific hand value calculation should be implemented in the environment
    # or a utility function for that game. E.g., for Blackjack, a method
    # `calculate_blackjack_hand_value(hand)` would be used.

    def __str__(self):
        return ", ".join(str(card) for card in self._cards) if self._cards else "Empty Hand"

    def __repr__(self):
        return f"Hand({self._cards})"