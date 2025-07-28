# deckgym/core/utils/spaces.py

from typing import Any, Tuple, Union, List

class Space:
    """
    Base class for defining observation and action spaces.
    Mimics a simplified version of Gymnasium's spaces.
    """
    def __init__(self, shape: Union[Tuple[int, ...], int], dtype: Any):
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = shape
        self.dtype = dtype

    def sample(self) -> Any:
        """Generates a random sample from the space."""
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        """Checks if a value is contained within the space."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return self.shape == other.shape and self.dtype == other.dtype


class DiscreteSpace(Space):
    """
    A discrete space, representing a finite set of non-negative integers.
    The valid actions are {0, 1, ..., n-1}.
    """
    def __init__(self, n: int):
        super().__init__(shape=(), dtype=int)
        self.n = n

    def sample(self) -> int:
        """Returns a random integer from 0 to n-1."""
        import random # Import locally to avoid circular dependencies if this file is imported early
        return random.randrange(self.n)

    def contains(self, x: Any) -> bool:
        """Checks if x is an integer within the range [0, n-1]."""
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self) -> str:
        return f"DiscreteSpace({self.n})"


class TupleSpace(Space):
    """
    A space that is a product of other spaces.
    E.g., Tuple(Discrete(2), Box(0, 1, shape=(3,)))
    """
    def __init__(self, spaces: List[Space]):
        self.spaces = spaces
        # Shape is the concatenation of shapes of individual spaces
        # For simple cases like (int, int, int), shape will be (3,)
        # For more complex nested spaces, this might need adjustment.
        # For now, assuming simple scalar or 1D array components.
        combined_shape = ()
        for space in spaces:
            if space.shape: # If it has a shape (e.g., Box)
                combined_shape += space.shape
            else: # If it's a scalar (e.g., Discrete)
                combined_shape += (1,) # Represent scalar as a dimension of 1 for consistent tuple obs

        # Determine a common dtype or use object if mixed
        # For simplicity, we'll assume a consistent dtype or handle conversion in env
        # For now, just pick the dtype of the first space, or object if empty.
        common_dtype = spaces[0].dtype if spaces else object

        super().__init__(shape=combined_shape, dtype=common_dtype)

    def sample(self) -> Tuple[Any, ...]:
        """Samples from each subspace and returns a tuple of samples."""
        return tuple(space.sample() for space in self.spaces)

    def contains(self, x: Any) -> bool:
        """Checks if x is a tuple and each element is contained in its corresponding subspace."""
        if not isinstance(x, tuple) or len(x) != len(self.spaces):
            return False
        return all(self.spaces[i].contains(x[i]) for i in range(len(self.spaces)))

    def __repr__(self) -> str:
        return f"TupleSpace({', '.join(repr(s) for s in self.spaces)})"

