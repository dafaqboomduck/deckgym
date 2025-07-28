# deckgym/envs/blackjack_env.py

import numpy as np
import logging
from typing import List, Tuple, Dict, Union, Any, Optional

# Import components from the new structure
from deckgym.core.game_components.card import Card
from deckgym.core.game_components.deck import Deck
from deckgym.core.game_components.hand import Hand
from deckgym.core.game_components.rank import Rank # For checking Ace rank

# Import the base environment and spaces
from deckgym.envs.base_env import BaseCardEnv
from deckgym.core.utils import spaces


# Configure logging for the blackjack environment
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # Set to INFO for general messages, DEBUG for detailed tracing


class PlayerHand(Hand):
    """
    Extends the base Hand class to include Blackjack-specific player hand state.

    Attributes:
        cards (List[Card]): The list of cards currently in the hand.
        stood (bool): Whether the player has chosen to stand on this hand.
        double_down (bool): Whether the player has doubled down on this hand.
        reward (float): The individual reward (positive, negative, or zero) assigned to this hand.
        is_split_ace (bool): Whether this hand originated from splitting Aces.
    """
    def __init__(self, cards: Optional[List[Card]] = None):
        super().__init__(cards)
        self.stood: bool = False
        self.double_down: bool = False
        self.reward: float = 0.0
        self.is_split_ace: bool = False

    def __str__(self) -> str:
        card_strs = [str(card) for card in self.cards]
        return f"Cards: {card_strs}, Stood: {self.stood}, DD: {self.double_down}, Reward: {self.reward:.2f}"


class BlackjackEnv(BaseCardEnv):
    """
    Blackjack Environment with additional rules and card counting,
    inheriting from BaseCardEnv.

    Observation Space:
    Tuple: (player_current_sum, dealer_card_showing, usable_ace[, running_count, true_count])
    - player_current_sum: Sum of player's current hand (int, 2–22).
    - dealer_card_showing: Value of dealer's visible card (int, 1–11, Ace=11).
    - usable_ace: Whether player has a usable ace (int, 0 or 1).
    - running_count: Current Hi-Lo running count (if count_cards=True)
    - true_count: Current Hi-Lo true count (running_count / decks_remaining) (if count_cards=True)

    Action Space:
    0: Stand
    1: Hit
    2: Double Down (if allowed)
    3: Split (if allowed)
    """

    # Define action constants as class attributes
    ACTION_STAND = 0
    ACTION_HIT = 1
    ACTION_DOUBLE_DOWN = 2
    ACTION_SPLIT = 3

    # Hi-Lo card counting values - now internal to BlackjackEnv as it's game-specific
    HI_LO_COUNT_VALUES = {
        Rank.TWO: 1, Rank.THREE: 1, Rank.FOUR: 1, Rank.FIVE: 1, Rank.SIX: 1,
        Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
        Rank.TEN: -1, Rank.JACK: -1, Rank.QUEEN: -1, Rank.KING: -1, Rank.ACE: -1
    }

    def __init__(self, render_mode: Optional[str] = None, num_decks: int = 6, blackjack_payout: float = 1.5,
                 allow_doubling: bool = True, allow_splitting: bool = True, count_cards: bool = True,
                 dealer_hits_on_soft_17: bool = True, reshuffle_threshold_pct: float = 0.25,
                 seed: Optional[int] = None):
        super().__init__(render_mode=render_mode, seed=seed) # Initialize the base class with seed

        self.num_decks = num_decks
        self.blackjack_payout = blackjack_payout
        self.allow_doubling = allow_doubling
        self.allow_splitting = allow_splitting
        self.count_cards = count_cards
        self.dealer_hits_on_soft_17 = dealer_hits_on_soft_17
        self.reshuffle_threshold_pct = reshuffle_threshold_pct

        # Define Observation Space
        # Player sum (min 2, max 21, plus 22 for bust)
        # Dealer showing (1-11, using 12 to include 11 as max for Discrete space)
        # Usable Ace (0 or 1)
        obs_components = [
            spaces.DiscreteSpace(22), # Player sum (0-21, so 22 discrete values)
            spaces.DiscreteSpace(12), # Dealer showing (1-11, so 12 discrete values for 0-11 range)
            spaces.DiscreteSpace(2)   # Usable Ace (0 or 1)
        ]
        if self.count_cards:
            # These ranges are illustrative, actual ranges depend on num_decks
            # A running count can go from -num_decks * 4 (all high cards) to +num_decks * 6 (all low cards)
            # For 6 decks: -24 to +36. Let's use a generous range for Discrete.
            # Max possible running count: 6 decks * 20 (approx 5 low cards per 10 cards) = 120
            # Min possible running count: 6 decks * -20 (approx 5 high cards per 10 cards) = -120
            # So, a range of 241 for running count (from -120 to 120, plus 1 for 0)
            obs_components.append(spaces.DiscreteSpace(241)) # Running count (e.g., -120 to +120)
            # True count can be running count / decks remaining. Decks remaining can be 0.1 to 6.
            # Max true count: 120 / 0.1 = 1200. Min true count: -120 / 6 = -20.
            # This is hard to discretize accurately. For simplicity, let's use a fixed large range
            # or consider a Box space if true count needs to be continuous.
            # For now, let's approximate a reasonable discrete range for true count.
            # True count typically ranges from -10 to +10 for practical purposes.
            obs_components.append(spaces.DiscreteSpace(21)) # True count (e.g., -10 to +10, 21 discrete values)
        self.observation_space = spaces.TupleSpace(obs_components)

        # Define Action Space
        # 0: Stand, 1: Hit, 2: Double Down, 3: Split
        self.action_space = spaces.DiscreteSpace(4) # Max 4 actions

        # Set the observation and action descriptions
        self.observation_description = (
            "(player_current_sum, dealer_card_showing, usable_ace"
            + (", running_count, true_count)" if self.count_cards else ")")
        )
        actions_desc = [f"{self.ACTION_STAND}: Stand", f"{self.ACTION_HIT}: Hit"]
        if self.allow_doubling:
            actions_desc.append(f"{self.ACTION_DOUBLE_DOWN}: Double Down")
        if self.allow_splitting:
            actions_desc.append(f"{self.ACTION_SPLIT}: Split")
        self.action_description = ", ".join(actions_desc)


        # Initialize Deck - pass the environment's RNG to the Deck
        self.deck: Deck = Deck(self.num_decks, rng=self._np_random)
        self.player_hands: List[PlayerHand] = []
        self.dealer_hand: Hand = Hand() # Dealer hand is a base Hand
        self.current_hand_index: int = 0
        self.running_count: int = 0

        # Internal state for tracking cards dealt since last shuffle to manage reshuffle threshold
        self._initial_deck_size: int = self.num_decks * 52
        self._reshuffle_threshold: int = int(self._initial_deck_size * self.reshuffle_threshold_pct)


        self.reset(seed=seed) # Call reset to set up initial game state

    @property
    def state_size(self) -> int:
        """
        Returns the size of the observation space (number of elements in the observation tuple).
        """
        return len(self.observation_space.spaces)

    @property
    def num_actions(self) -> int:
        """
        Returns the number of possible actions in the environment.
        """
        return self.action_space.n

    def _update_hand_value(self, hand_cards: List[Card]) -> Tuple[int, bool]:
        """
        Calculates the sum of cards in a hand, considering usable aces.
        """
        hand_sum_soft = 0 # Sum with all Aces as 11 initially
        num_aces_in_hand = 0
        for card in hand_cards:
            if card.rank == Rank.ACE:
                num_aces_in_hand += 1
                hand_sum_soft += 11
            else:
                hand_sum_soft += card.value # Use the card's value property

        # Adjust aces from 11 to 1 if busting
        current_sum = hand_sum_soft
        aces_remaining_as_11 = num_aces_in_hand

        while current_sum > 21 and aces_remaining_as_11 > 0:
            current_sum -= 10 # Convert an Ace from 11 to 1
            aces_remaining_as_11 -= 1 # One less ace contributing 11

        # A usable ace exists if there was at least one ace originally,
        # and after adjustments, at least one ace is still counted as 11.
        usable_ace = (current_sum <= 21 and aces_remaining_as_11 > 0)

        return current_sum, usable_ace

    def _deal_card(self, hand_obj: Union[PlayerHand, Hand], face_up: bool = True, is_initial_deal: bool = False) -> Card:
        """
        Deals a card from the deck and adds it to the specified hand.
        Manages card counting and reshuffling.
        """
        # Reshuffle logic based on the new Deck class which doesn't have internal reshuffle_threshold
        if len(self.deck) <= self._reshuffle_threshold:
            logger.debug(f"Deck count ({len(self.deck)}) below reshuffle threshold ({self._reshuffle_threshold}). Reshuffling.")
            self.deck.reset() # Recreates a fresh deck and shuffles it using its internal RNG (which is self._np_random)
            self.running_count = 0 # Reset count on new shoe
            self.visible_cards = [] # Clear visible cards on reshuffle

        card = self.deck.deal(num_cards=1)[0] # Deal one card

        hand_obj.add_card(card)
        if face_up: # Only add to visible_cards if it's face up
            self.visible_cards.append(card)

        # Update running count only for face-up cards, unless it's the initial deal reset logic
        if self.count_cards and face_up and not is_initial_deal:
            self.running_count += self.HI_LO_COUNT_VALUES[card.rank]
            logger.debug(f"Dealt {card}, running count updated to {self.running_count}")
        elif self.count_cards and not face_up and not is_initial_deal:
            # For dealer's hole card, count it when it's revealed at dealer's turn
            logger.debug(f"Dealt {card} face down.")

        return card

    def _get_observation(self) -> Tuple[int, ...]:
        """
        Generates the current observation tuple for the agent.
        """
        # The observation should reflect the *current active* player hand.
        # If all player hands are resolved (e.g., during dealer's turn),
        # the observation should still be valid, possibly reflecting the last played hand or a default.
        player_sum_for_obs = 0
        usable_ace_for_obs = 0

        if self.player_hands and self.current_hand_index < len(self.player_hands):
            current_player_hand_cards = self.player_hands[self.current_hand_index].cards
            player_sum_for_obs, usable_ace_for_obs = self._update_hand_value(current_player_hand_cards)
        elif self.player_hands: # All player hands are resolved, but we still have hands (e.g., after dealer plays)
             # In this case, use the first hand's value as a representative observation.
             # Or, depending on desired agent behavior, could return a terminal observation.
             player_sum_for_obs, usable_ace_for_obs = self._update_hand_value(self.player_hands[0].cards)
        else:
            # Should not happen if environment is reset correctly
            pass

        # Dealer's showing card value
        dealer_showing_value = self.dealer_hand.cards[0].value if self.dealer_hand.cards else 0

        obs: Tuple[int, ...] = (player_sum_for_obs, dealer_showing_value, int(usable_ace_for_obs))
        if self.count_cards:
            # Decks remaining calculation: total cards in deck, including those dealt
            # The `Deck` class now manages its own internal `_cards` list, so `len(self.deck)`
            # *is* the number of cards remaining in the shoe.
            decks_remaining = max(1e-6, len(self.deck) / 52.0)
            true_count = round(self.running_count / decks_remaining)
            obs += (self.running_count, true_count)

        return obs

    def _reset_game_state(self, seed: Optional[int] = None) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """
        Initializes the game-specific state for a new episode of Blackjack.
        """
        # The Deck's internal RNG is already re-seeded by BaseCardEnv.reset if seed is provided.
        self.deck.reset() # Resets the deck to full and unshuffled state, then shuffles
        self.running_count = 0 # Reset running count on new shoe
        self.visible_cards = [] # Clear visible cards

        self.player_hands = [PlayerHand()]
        self.dealer_hand = Hand()
        self.current_hand_index = 0

        # Initial deal
        # Player Card 1 (face up)
        self._deal_card(self.player_hands[0], face_up=True, is_initial_deal=True)
        # Dealer Up Card (face up)
        self._deal_card(self.dealer_hand, face_up=True, is_initial_deal=True)
        # Player Card 2 (face up)
        self._deal_card(self.player_hands[0], face_up=True, is_initial_deal=True)
        # Dealer Hole Card (face down) - will be added to visible_cards when revealed
        hole_card = self.deck.deal(num_cards=1)[0]
        self.dealer_hand.add_card(hole_card)
        # Do NOT add hole_card to self.visible_cards yet, it's face down.

        if self.count_cards:
            # Explicitly calculate running count for initial face-up cards
            self.running_count += self.HI_LO_COUNT_VALUES[self.player_hands[0].cards[0].rank]
            self.running_count += self.HI_LO_COUNT_VALUES[self.player_hands[0].cards[1].rank]
            self.running_count += self.HI_LO_COUNT_VALUES[self.dealer_hand.cards[0].rank]
            logger.debug(f"Reset: Initial running count: {self.running_count}")

        player_sum, _ = self._update_hand_value(self.player_hands[0].cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand.cards) # Includes hole card for blackjack check

        info: Dict[str, Any] = {"can_double": False, "can_split": False, "is_terminal": False}

        player_blackjack = (player_sum == 21 and len(self.player_hands[0].cards) == 2)
        dealer_blackjack = (dealer_sum == 21 and len(self.dealer_hand.cards) == 2)

        if player_blackjack and dealer_blackjack:
            self.player_hands[0].reward = 0.0
            info["is_terminal"] = True
            logger.info("Reset: Push (both blackjack).")
        elif player_blackjack:
            self.player_hands[0].reward = self.blackjack_payout
            info["is_terminal"] = True
            logger.info(f"Reset: Player Blackjack! Reward: {self.player_hands[0].reward}")
        elif dealer_blackjack:
            self.player_hands[0].reward = -1.0
            info["is_terminal"] = True
            # Reveal dealer hole card if they have blackjack at start
            if self.count_cards:
                self.running_count += self.HI_LO_COUNT_VALUES[self.dealer_hand.cards[1].rank]
                self.visible_cards.append(self.dealer_hand.cards[1]) # Hole card now visible
                logger.debug(f"Dealer Blackjack: Hole card revealed, running count: {self.running_count}")
            logger.info(f"Reset: Dealer Blackjack. Reward: {self.player_hands[0].reward}")

        if not info["is_terminal"]:
            if self.allow_doubling and len(self.player_hands[0].cards) == 2: # Only on first two cards
                info["can_double"] = True
            if self.allow_splitting and len(self.player_hands[0].cards) == 2 and \
               self.player_hands[0].cards[0].rank == self.player_hands[0].cards[1].rank:
                info["can_split"] = True

        # Set player hand to stood if game ends on initial deal (blackjack, push, or dealer blackjack)
        if info["is_terminal"]:
            self.player_hands[0].stood = True
            self.done = True # Mark environment as done

        observation = self._get_observation() # Get the initial observation based on the current state
        return observation, info

    def _step_game_logic(self, action: int) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]:
        """
        Applies the agent's action and advances the Blackjack game state.
        """
        terminated = False
        truncated = False # Blackjack usually doesn't have truncation
        info: Dict[str, Any] = {"can_double": False, "can_split": False, "is_terminal": False}
        total_reward = 0.0 # This will be the cumulative reward for the step

        if self.done: # If game is already over, return current state
            logger.warning("Step called on a terminated environment. Reset the environment.")
            return self._get_observation(), 0.0, True, False, {"can_double": False, "can_split": False, "is_terminal": True}

        if not (0 <= self.current_hand_index < len(self.player_hands)):
            logger.error("Step called when no active player hands remain. This should not happen if self.done is False.")
            return self._get_observation(), 0.0, True, False, {"can_double": False, "can_split": False, "is_terminal": True}

        current_player_hand_obj = self.player_hands[self.current_hand_index]
        current_player_hand_cards = current_player_hand_obj.cards

        is_first_action_on_hand = (len(current_player_hand_cards) == 2 and
                                   not current_player_hand_obj.stood and
                                   not current_player_hand_obj.double_down)

        current_hand_resolved = False # Flag to indicate if current hand's play is finished

        if action == self.ACTION_HIT: # Use class attribute
            logger.debug(f"Hand {self.current_hand_index + 1}: Player hits.")
            self._deal_card(current_player_hand_obj, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            if player_sum > 21:
                current_player_hand_obj.reward = -1.0
                if current_player_hand_obj.double_down:
                    current_player_hand_obj.reward *= 2
                logger.info(f"Hand {self.current_hand_index + 1}: Player busts ({player_sum}). Reward: {current_player_hand_obj.reward}")
                current_player_hand_obj.stood = True # Mark hand as stood when it busts
                current_hand_resolved = True
            # Else, hand is not resolved yet, player can continue to hit/stand
        elif action == self.ACTION_STAND: # Use class attribute
            logger.debug(f"Hand {self.current_hand_index + 1}: Player stands ({self._update_hand_value(current_player_hand_obj.cards)[0]}).")
            current_player_hand_obj.stood = True
            current_hand_resolved = True
        elif action == self.ACTION_DOUBLE_DOWN and self.allow_doubling and is_first_action_on_hand: # Use class attribute
            logger.debug(f"Hand {self.current_hand_index + 1}: Player doubles down.")
            self._deal_card(current_player_hand_obj, face_up=True)
            player_sum, _ = self._update_hand_value(current_player_hand_obj.cards)
            current_player_hand_obj.double_down = True
            current_player_hand_obj.stood = True # Automatically stands after double down
            if player_sum > 21:
                current_player_hand_obj.reward = -1.0 * 2 # Double penalty for bust on double down
                logger.info(f"Hand {self.current_hand_index + 1}: Player busts on double down ({player_sum}). Reward: {current_player_hand_obj.reward}")
            current_hand_resolved = True
        elif action == self.ACTION_SPLIT and self.allow_splitting and is_first_action_on_hand and \
             current_player_hand_cards[0].rank == current_player_hand_cards[1].rank: # Use class attribute
            logger.debug(f"Hand {self.current_hand_index + 1}: Player splits.")
            card1, card2 = current_player_hand_cards

            # Clear current hand and add one card back
            current_player_hand_obj.clear() # Use clear method from Hand
            current_player_hand_obj.add_card(card1)
            # Create new hand for the second card
            new_hand = PlayerHand(cards=[card2])

            # Special rule for splitting Aces
            if card1.rank == Rank.ACE:
                current_player_hand_obj.is_split_ace = True
                new_hand.is_split_ace = True
                logger.debug("Splitting Aces detected.")

            # Insert new hand right after the current one for sequential play
            self.player_hands.insert(self.current_hand_index + 1, new_hand)

            # Deal one card to each new hand
            self._deal_card(current_player_hand_obj, face_up=True)
            self._deal_card(new_hand, face_up=True)

            # If Aces were split, automatically stand these hands
            if current_player_hand_obj.is_split_ace:
                current_player_hand_obj.stood = True
                if new_hand.is_split_ace:
                    new_hand.stood = True # Ensure the newly created split ace hand is also stood

            # After split, the current hand (first split hand) is still active for actions
            # unless it was an Ace split.
            if not current_player_hand_obj.stood: # Only if not auto-stood (i.e., not split aces)
                current_hand_resolved = False # Player can still hit/stand on this hand
            else: # If it was an Ace split, it's auto-stood
                current_hand_resolved = True
        else:
            # Invalid action
            logger.warning(f"Hand {self.current_hand_index + 1}: Invalid action {action} performed. Penalizing.")
            current_player_hand_obj.reward = -1 # Penalty for illegal move
            current_player_hand_obj.stood = True # Forcing hand to stand after invalid move
            current_hand_resolved = True

        # Now, advance to the next hand if the current one is resolved, or resolve the game.
        if current_hand_resolved:
            terminated = self._advance_to_next_hand_or_resolve_game()
        else:
            terminated = False # Game is not done yet if hand is not resolved

        # Determine the observation *after* all state changes, including advancing to the next hand.
        next_observation = self._get_observation()

        # Update info for next observation (if game is not done)
        if not terminated and self.current_hand_index < len(self.player_hands):
            next_hand = self.player_hands[self.current_hand_index]
            next_hand_cards = next_hand.cards

            info["can_double"] = (self.allow_doubling and
                                  len(next_hand_cards) == 2 and
                                  not next_hand.stood and
                                  not next_hand.double_down and
                                  not next_hand.is_split_ace)

            info["can_split"] = (self.allow_splitting and
                                 len(next_hand_cards) == 2 and
                                 next_hand_cards[0].rank == next_hand_cards[1].rank and
                                 not next_hand.stood and
                                 not next_hand.double_down)
        else:
            # If terminated, no actions are possible
            info = {"can_double": False, "can_split": False}
            info["is_terminal"] = True # Mark as terminal if game ends

        # The reward returned by step is the reward for *this* step.
        # For Blackjack, the main reward is typically at the end of the game.
        # So, if terminated, sum up all hand rewards. Otherwise, 0.0 for intermediate steps.
        total_reward = sum(hand.reward for hand in self.player_hands) if terminated else 0.0

        return next_observation, total_reward, terminated, truncated, info

    def _advance_to_next_hand_or_resolve_game(self) -> bool:
        """
        Advances the current_hand_index to the next active player hand.
        If all player hands are resolved, the dealer plays and game ends.
        Returns True if the game is done, False otherwise.
        """
        while self.current_hand_index < len(self.player_hands) and self.player_hands[self.current_hand_index].stood:
            self.current_hand_index += 1

        if self.current_hand_index >= len(self.player_hands):
            # All player hands are done, dealer plays
            logger.info("All player hands resolved. Dealer's turn.")
            self._dealer_plays()
            # Calculate final rewards for all hands that didn't bust
            for i, hand in enumerate(self.player_hands):
                if hand.reward == 0.0: # Only calculate if not already busted or penalized
                    hand.reward = self._calculate_reward(player_hand_cards=hand.cards)
                    if hand.double_down:
                        hand.reward *= 2
                    logger.debug(f"Hand {i+1} final reward: {hand.reward}")
            return True # Game is done
        return False # Not all player hands are done yet

    def _dealer_plays(self) -> None:
        """Dealer hits until sum is 17 or more."""
        # Reveal dealer's hole card and update count
        if self.count_cards and len(self.dealer_hand.cards) == 2 and self.dealer_hand.cards[1] not in self.visible_cards:
            self.running_count += self.HI_LO_COUNT_VALUES[self.dealer_hand.cards[1].rank]
            self.visible_cards.append(self.dealer_hand.cards[1]) # Hole card now visible
            logger.debug(f"Dealer hole card revealed: {self.dealer_hand.cards[1]}, running count: {self.running_count}")

        dealer_sum, usable_ace = self._update_hand_value(self.dealer_hand.cards)
        logger.info(f"Dealer starts playing with {dealer_sum} (usable ace: {usable_ace}).")

        while True:
            if dealer_sum > 21:
                logger.info(f"Dealer busts with {dealer_sum}.")
                break
            # Dealer hits on 16 or less
            # Dealer hits on soft 17 if dealer_hits_on_soft_17 is True
            if dealer_sum < 17 or (dealer_sum == 17 and usable_ace and self.dealer_hits_on_soft_17):
                logger.debug(f"Dealer hits (current sum: {dealer_sum}, usable ace: {usable_ace}).")
                self._deal_card(self.dealer_hand, face_up=True) # Deal card, it will be added to visible_cards
                dealer_sum, usable_ace = self._update_hand_value(self.dealer_hand.cards)
            else:
                logger.info(f"Dealer stands with {dealer_sum}.")
                break

    def _calculate_reward(self, player_hand_cards: List[Card]) -> float:
        """
        Calculates the reward for a single player hand based on its outcome against the dealer.
        """
        player_sum, _ = self._update_hand_value(player_hand_cards)
        dealer_sum, _ = self._update_hand_value(self.dealer_hand.cards)

        logger.info(f"Calculating reward for player hand with sum {player_sum} against dealer sum {dealer_sum}. Dealer total: {dealer_sum}")

        # Player bust is handled immediately in step, so this is for hands that didn't bust.
        if player_sum > 21: # Should ideally not happen if called correctly
            logger.error("Error: _calculate_reward called for a busted hand that should have been resolved earlier.")
            return -1.0
        elif dealer_sum > 21:
            logger.info("Dealer busted. Player wins.")
            return 1.0 # Player wins because dealer busted
        elif player_sum > dealer_sum:
            logger.info("Player has higher sum. Player wins.")
            return 1.0
        elif player_sum < dealer_sum:
            logger.info("Dealer has higher sum. Player loses.")
            return -1.0
        else:
            logger.info("Player and Dealer have same sum. Push.")
            return 0.0 # Push

    def _is_game_over(self) -> bool:
        """
        Checks if the current Blackjack game episode is terminated.
        The game is over when all player hands are stood/busted and the dealer has played.
        """
        # This method is primarily used by the public `step` to set `self.done`.
        # The logic for advancing hands and dealer play already determines termination.
        return self.current_hand_index >= len(self.player_hands) and \
               all(hand.stood or self._update_hand_value(hand.cards)[0] > 21 for hand in self.player_hands)


    def _render_ansi(self) -> str:
        """
        Returns a string representation of the current Blackjack game state for ANSI output.
        """
        output = ["\n--- Blackjack Game (ANSI) ---"]

        for i, hand in enumerate(self.player_hands):
            hand_sum, usable_ace = self._update_hand_value(hand.cards)
            status = []
            if hand.stood: status.append("Stood")
            if hand.double_down: status.append("Doubled Down")
            if hand.is_split_ace: status.append("Split Ace Hand")
            if hand_sum > 21: status.append("Bust")
            if hand.reward != 0: status.append(f"Reward: {hand.reward:.2f}")

            status_str = ", ".join(status) if status else "Active"
            output.append(f"Player Hand {i+1} ({'Current' if i == self.current_hand_index else 'Other'}): "
                          f"{[str(c) for c in hand.cards]} (Sum: {hand_sum}, Usable Ace: {int(usable_ace)}) [{status_str}]")

        dealer_sum, _ = self._update_hand_value(self.dealer_hand.cards)
        if len(self.dealer_hand.cards) == 2 and self.current_hand_index < len(self.player_hands):
            dealer_cards_display = [str(self.dealer_hand.cards[0]), '??']
            dealer_total_display = '??'
        else:
            dealer_cards_display = [str(c) for c in self.dealer_hand.cards]
            dealer_total_display = dealer_sum
        output.append(f"Dealer Hand: {dealer_cards_display} (Showing: {self.dealer_hand.cards[0].value}, Total: {dealer_total_display})")

        if self.count_cards:
            decks_remaining = max(1e-6, len(self.deck) / 52.0)
            true_count = round(self.running_count / decks_remaining)
            output.append(f"Running Count: {self.running_count}, True Count: {true_count} (Decks Left: {decks_remaining:.1f})")
        output.append("----------------------")
        return "\n".join(output)

    def _render_human(self) -> None:
        """
        Renders the current state of the Blackjack game to the console for human readability.
        """
        print(self._render_ansi()) # Simply reuse ANSI rendering for human mode

    def close(self) -> None:
        """
        Performs any cleanup. For BlackjackEnv, no specific cleanup is needed.
        """
        pass

