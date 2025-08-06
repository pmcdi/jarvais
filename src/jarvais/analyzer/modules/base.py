from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, Field


class AnalyzerModule(BaseModel, ABC):
    """
    Base class for all analyzer modules.
    
    This class defines the common interface that all analyzer modules should follow:
    - All modules are Pydantic BaseModel classes for easy serialization and validation
    - All modules have an `enabled` field to control execution
    - All modules implement a `build` classmethod for easy instantiation with sensible defaults
    - All modules implement a `__call__` method that processes DataFrames
    - Most modules follow a pd.DataFrame -> pd.DataFrame structure, except visualization modules
    
    Example:
        ```python
        class MyModule(AnalyzerModule):
            my_param: str = Field(description="Example parameter")
            
            @classmethod
            def build(cls, **kwargs) -> "MyModule":
                return cls(my_param="default_value")
            
            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                # Process the DataFrame
                return df
        ```
    """
    
    enabled: bool = Field(
        default=True,
        description="Whether this module is enabled and should be executed."
    )
    
    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs) -> "AnalyzerModule":
        """
        Factory method to create an instance of the module with sensible defaults.
        
        This method should be implemented by each module to provide an easy way
        to instantiate the module with appropriate default settings based on
        the provided column information or other context.
        
        Args:
            *args: Variable positional arguments, typically column lists
            **kwargs: Variable keyword arguments for additional configuration
            
        Returns:
            An instance of the module with appropriate default settings
        """
        pass
    
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Process a DataFrame through this module.
        
        Most modules should follow a pd.DataFrame -> pd.DataFrame pattern,
        transforming the input data and returning the modified DataFrame.
        
        Visualization modules are an exception and may return None while
        generating plots or other side effects.
        
        Args:
            df: The input DataFrame to process
            
        Returns:
            The processed DataFrame, or None for modules that don't transform data
        """
        pass
