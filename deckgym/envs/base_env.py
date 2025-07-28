# deckgym/envs/base_env.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union, List
import logging
import numpy as np
from numpy.random import Generator

# Import the custom spaces module
from deckgym.core.utils import spaces
from deckgym.core.game_components.card import Card # For visible_cards type hinting


# Configure logging for the base environment
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set to INFO for general messages, DEBUG for detailed tracing

class BaseCardEnv(ABC):
    """
    Abstract Base Class for Reinforcement Learning environments for card games.
    This class defines the core API that all specific card game environments
    (e.g., Blackjack, Poker, Uno) should implement, similar to Gymnasium's Env.

    Attributes:
        render_mode (Optional[str]): The rendering mode ('human', 'ansi', None).
        observation_space (spaces.Space): Defines the structure and bounds of observations.
        action_space (spaces.Space): Defines the structure and bounds of actions.
        _np_random (numpy.random.Generator): The random number generator for the environment.
        done (bool): Whether the current episode has ended.
        reward (float): The cumulative reward for the current episode.
        visible_cards (List[Card]): A list of cards that are currently visible to the agent.
                                    Useful for card counting or partial observability.
        observation_description (str): A string describing the observation space.
        action_description (str): A string describing the action space.
    """

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        self.render_mode = render_mode

        # Initialize random number generator
        self._np_random: Generator = np.random.default_rng(seed)

        # Define abstract observation and action spaces
        # These will be defined concretely in derived classes
        self.observation_space: spaces.Space # Will be set by subclass
        self.action_space: spaces.Space     # Will be set by subclass

        # Game state components (placeholders for common elements)
        self.done: bool = False
        self.reward: float = 0.0
        self.visible_cards: List[Card] = [] # For tracking cards exposed during the game

        # Descriptions for observation and action spaces
        self.observation_description: str = "No observation description provided."
        self.action_description: str = "No action description provided."

    @abstractmethod
    def _reset_game_state(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Initializes the game-specific state for a new episode.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (Optional[int]): An optional seed for reproducibility. If provided,
                                  the environment's internal RNG will be re-seeded.
            options (Optional[Dict[str, Any]]): Optional dictionary of options for resetting.

        Returns:
            Tuple[Any, Dict[str, Any]]:
                - observation (Any): The initial observation of the environment's state.
                - info (Dict): Auxiliary information (e.g., allowed actions).
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            logger.debug(f"Environment re-seeded with {seed}.")

        self.done = False
        self.reward = 0.0
        self.visible_cards = [] # Reset visible cards for new round

        observation, info = self._reset_game_state(seed=seed) # Pass seed to game-specific reset

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            print(self._render_ansi())

        logger.info(f"Environment reset. Observation: {observation}, Info: {info}")
        return observation, info

    @abstractmethod
    def _step_game_logic(self, action: Union[int, Any]) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Applies the agent's action and advances the game state according to game rules.
        This method should be implemented by subclasses.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
                - observation (Any): The new observation after taking the action.
                - reward (float): The reward received from the previous action.
                - terminated (bool): Whether the episode has ended due to a terminal state.
                - truncated (bool): Whether the episode has ended due to truncation (e.g., time limit).
                - info (Dict): Auxiliary information (e.g., allowed actions, game state).
        """
        raise NotImplementedError

    def step(self, action: Union[int, Any]) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Takes an action and returns the next observation, reward,
        whether the episode is terminated or truncated, and info.

        Args:
            action (Union[int, Any]): The action to be taken.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
                - observation (Any): The new observation after taking the action.
                - reward (float): The reward received from the previous action.
                - terminated (bool): Whether the episode has ended due to a terminal state.
                - truncated (bool): Whether the episode has ended due to truncation (e.g., time limit).
                - info (Dict): Auxiliary information.
        """
        observation, reward, terminated, truncated, info = self._step_game_logic(action)

        self.done = terminated or truncated
        self.reward = reward # Update cumulative reward if needed, or keep as step reward

        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            print(self._render_ansi())

        logger.debug(f"Step completed. Action: {action}, Next Obs: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        return observation, reward, terminated, truncated, info

    @abstractmethod
    def _get_observation(self) -> Any:
        """Constructs and returns the current observation for the agent."""
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, **kwargs) -> float:
        """
        Calculates and returns the reward for the current state or specific outcome.
        Subclasses should define parameters as needed (e.g., player_hand, dealer_hand).
        """
        raise NotImplementedError

    @abstractmethod
    def _is_game_over(self) -> bool:
        """Checks if the current game episode is terminated."""
        raise NotImplementedError

    def render(self) -> Optional[Union[np.ndarray, str]]:
        """
        Renders the current state of the environment based on the render_mode.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()
        return None

    @abstractmethod
    def _render_ansi(self) -> str:
        """Returns a string representation of the current game state for ANSI output."""
        raise NotImplementedError

    @abstractmethod
    def _render_human(self) -> None:
        """Renders the game state to the console or GUI for human readability."""
        raise NotImplementedError

    def close(self) -> None:
        """
        Performs any necessary cleanup (e.g., closing rendering windows).
        Subclasses should implement this if cleanup is needed.
        """
        pass

