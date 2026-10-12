from typing import Any, Callable


class Registry:
    """Maps a string name to a factory function."""

    def __init__(self, kind: str, expected_type: type | None = None):
        """
        Args:
            kind: The kind of object being registered (e.g., 'chunker', 'embedding').
            expected_type: The expected type of the objects created by the factory functions.
                           The objects created by the factory functions should be instances of this type. 
                           If None, no type checking is performed.
        """
        self.kind = kind
        self.expected_type = expected_type
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, *, overwrite: bool = False):
        """
        This decorator registers a factory function under a given name. 
        The factory function should take named parameters and return an instance of the expected type.
        
        Args:
            name: The name to register the factory function under.
            overwrite: If True, overwrite an existing factory function with the same name.
            
        Returns:
            The decorator function.
        """
        def deco(factory):
            """
            Function that actually registers the factory function. 
            It checks if the name is already registered and raises an error if overwrite is False.
            """
            if name in self._factories and not overwrite:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory
        return deco

    def build(self, name: str, **params) -> Any:
        """
        Builds an object from the factory function registered under the given name.

        Args:
            name: The name of the factory function to use.
            **params: The parameters to pass to the factory function.
                      Note that the factory function should accept named parameters.

        Returns:
            The built object.

        """
        if name not in self._factories:
            raise ValueError(
                f"Unknown {self.kind} '{name}'. Available: {self.available()}"
            )
        obj = self._factories[name](**params)

        # type check for safety: Check if the object is of the expected type, if an expected type is provided
        if self.expected_type and not isinstance(obj, self.expected_type):
            raise TypeError(
                f"{self.kind} '{name}' returned {type(obj).__name__}, "
                f"expected {self.expected_type.__name__}"
            )
        return obj

    def available(self) -> list[str]:
        return sorted(self._factories)