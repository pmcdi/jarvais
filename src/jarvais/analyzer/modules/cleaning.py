import re
from typing import Dict, Any, List

import pandas as pd
from pydantic import Field


from .base import AnalyzerModule
from jarvais.loggers import logger


class JanitorModule(AnalyzerModule):
    """
    Module for cleaning data with numeric values that have auxiliary symbols.
    
    This module handles cases where numeric columns contain values like "30+", "<15", ">=10", etc.
    and converts them to usable numeric formats for downstream processing.
    
    Uses the same auxiliary symbols pattern as infer_types: r'[<>≤≥±~+\-]+'
    """
    
    columns: List[str] = Field(
        default_factory=list,
        description="List of column names to apply data cleaning to. If empty, auto-detection will be used."
    )
    
    auto_detect: bool = Field(
        default=True,
        description="Whether to automatically detect columns that need cleaning based on numeric patterns with auxiliary symbols."
    )
    
    conversion_threshold: float = Field(
        default=0.5,
        description="Minimum proportion of values that must convert to numeric for a column to be cleaned."
    )
    
    auxiliary_symbols_pattern: str = Field(
        default=r'(<=|>=|[<>≤≥±~+\-])+',
        description="Regex pattern for auxiliary symbols to remove from numeric values (enhanced to handle <=, >= explicitly)"
    )
    
    na_patterns: List[str] = Field(
        default_factory=lambda: ['na', 'none', 'null', 'n/a', 'nan', 'missing', 'never', 'unknown', ''],
        description="String representations of missing values to filter out"
    )
    
    report: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report of cleaning operations performed"
    )
    
    @classmethod
    def build(
        cls,
        columns: List[str] = None,
        auto_detect: bool = True,
        conversion_threshold: float = 0.5,
        **kwargs
    ) -> "JanitorModule":
        """
        Build a JanitorModule with specified parameters.
        
        Args:
            columns: List of columns to clean. If None, auto-detection will be used.
            auto_detect: Whether to automatically detect columns needing cleaning.
            conversion_threshold: Minimum conversion rate to consider a column for cleaning.
            **kwargs: Additional parameters.
            
        Returns:
            JanitorModule instance
        """
        return cls(
            columns=columns or [],
            auto_detect=auto_detect,
            conversion_threshold=conversion_threshold,
            **kwargs
        )
    
    def _detect_numeric_with_auxiliary_symbols(self, series: pd.Series) -> tuple[bool, float]:
        """
        Detect if a series contains numeric values with auxiliary symbols.
        Uses the same logic as infer_types function.
        
        Args:
            series: pandas Series to analyze
            
        Returns:
            Tuple of (should_clean, conversion_rate)
        """
        if series.dtype != 'object':
            return False, 0.0
            
        # Convert to string and filter out common string representations of missing values
        string_values = series.dropna().astype(str).str.lower()
        
        if len(string_values) == 0:
            return False, 0.0
        
        # Remove rows with string representations of missing values
        filtered_values = string_values[~string_values.isin(self.na_patterns)]
        
        if len(filtered_values) == 0:
            return False, 0.0
        
        # Remove auxiliary symbols and try to convert to numeric
        cleaned_values = filtered_values.str.replace(self.auxiliary_symbols_pattern, '', regex=True)
        
        # Remove empty strings that might result from cleaning
        cleaned_values = cleaned_values[cleaned_values.str.strip() != '']
        
        if len(cleaned_values) == 0:
            return False, 0.0
        
        # Try to convert to numeric
        numeric_converted = pd.to_numeric(cleaned_values, errors='coerce')
        
        # Calculate conversion rate
        valid_numeric = numeric_converted.notna().sum()
        total_cleaned = len(cleaned_values)
        
        conversion_rate = valid_numeric / total_cleaned if total_cleaned > 0 else 0.0
        
        # Should clean if conversion rate meets threshold
        should_clean = conversion_rate >= self.conversion_threshold
        
        return should_clean, conversion_rate
    
    def _clean_column(self, series: pd.Series) -> pd.Series:
        """
        Clean a single column by removing auxiliary symbols and converting to numeric.
        
        Args:
            series: pandas Series to clean
            
        Returns:
            Cleaned pandas Series (numeric where possible, original values where not convertible)
        """
        if series.dtype != 'object':
            return series
            
        # Work with a copy
        cleaned_series = series.copy()
        
        # For object dtype columns, process string representations
        string_mask = cleaned_series.notna()
        
        # Convert to string for processing
        string_values = cleaned_series[string_mask].astype(str)
        
        # Remove auxiliary symbols using the same pattern as infer_types
        cleaned_strings = string_values.str.replace(self.auxiliary_symbols_pattern, '', regex=True)
        
        # Try to convert to numeric
        numeric_values = pd.to_numeric(cleaned_strings, errors='coerce')
        
        # For values that successfully converted to numeric, use the numeric version
        # For values that didn't convert, keep the original
        result = cleaned_series.copy()
        
        # Update with numeric values where conversion was successful
        numeric_mask = string_mask & numeric_values.notna()
        result.loc[numeric_mask] = numeric_values.loc[numeric_mask]
        
        return result
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned columns
        """
        if not self.enabled:
            return df
            
        logger.info("Performing data cleaning for numeric values with auxiliary symbols...")
        
        df_cleaned = df.copy()
        columns_to_clean = []
        
        # Determine which columns to clean
        if self.columns:
            columns_to_clean = [col for col in self.columns if col in df.columns]
            if len(columns_to_clean) < len(self.columns):
                missing = set(self.columns) - set(columns_to_clean)
                logger.warning(f"Some specified columns not found in data: {missing}")
        
        if self.auto_detect:
            # Auto-detect columns that need cleaning using the same logic as infer_types
            auto_detected = []
            for col in df.columns:
                if col not in columns_to_clean:
                    should_clean, conversion_rate = self._detect_numeric_with_auxiliary_symbols(df[col])
                    if should_clean:
                        auto_detected.append(col)
                        logger.info(f"Auto-detected column for cleaning: {col} (conversion rate: {conversion_rate:.2%})")
                        
            columns_to_clean.extend(auto_detected)
        
        if not columns_to_clean:
            logger.info("No columns identified for data cleaning.")
            return df_cleaned
        
        # Clean each identified column
        for col in columns_to_clean:
            logger.info(f"Cleaning column: {col}")
            
            original_series = df[col]
            cleaned_series = self._clean_column(original_series)
            
            # Update the dataframe
            df_cleaned[col] = cleaned_series
            
            # Generate report
            original_non_null = original_series.notna().sum()
            cleaned_non_null = cleaned_series.notna().sum()
            
            # Count how many values were actually converted to numeric
            if original_series.dtype == 'object' and cleaned_series.dtype != 'object':
                final_type = 'numeric'
                numeric_count = cleaned_series.notna().sum()
            else:
                # Check if any values were converted to numeric (mixed types)
                numeric_count = 0
                if original_series.dtype == 'object':
                    for i in range(len(cleaned_series)):
                        if pd.notna(cleaned_series.iloc[i]) and pd.notna(original_series.iloc[i]):
                            try:
                                if isinstance(cleaned_series.iloc[i], (int, float)) and not isinstance(original_series.iloc[i], (int, float)):
                                    numeric_count += 1
                            except:
                                pass
                final_type = 'mixed' if numeric_count > 0 else str(cleaned_series.dtype)
            
            conversion_rate = numeric_count / original_non_null if original_non_null > 0 else 0.0
            
            self.report[col] = {
                'original_type': str(original_series.dtype),
                'final_type': final_type,
                'conversion_rate': conversion_rate,
                'original_non_null': original_non_null,
                'final_non_null': cleaned_non_null,
                'values_converted': numeric_count,
                'sample_conversions': self._get_sample_conversions(original_series, cleaned_series)
            }
        
        # Log summary of cleaning operations
        if self.report:
            logger.info(f"Data cleaning completed for {len(self.report)} columns")
            for col, report in self.report.items():
                logger.info(
                    f"  {col}: {report['original_type']} → {report['final_type']} "
                    f"(conversion rate: {report['conversion_rate']:.2%}, "
                    f"values converted: {report['values_converted']})"
                )
                if report['sample_conversions']:
                    logger.info(f"    Sample conversions: {report['sample_conversions']}")
        
        return df_cleaned
    
    def _get_sample_conversions(self, original: pd.Series, cleaned: pd.Series, n_samples: int = 3) -> List[tuple]:
        """
        Get sample conversions to show in the report.
        
        Args:
            original: Original series
            cleaned: Cleaned series
            n_samples: Number of sample conversions to return
            
        Returns:
            List of (original_value, cleaned_value) tuples
        """
        samples = []
        count = 0
        
        for i in range(len(original)):
            if count >= n_samples:
                break
                
            orig_val = original.iloc[i]
            clean_val = cleaned.iloc[i]
            
            # Look for cases where conversion happened
            if pd.notna(orig_val) and pd.notna(clean_val):
                if str(orig_val) != str(clean_val):
                    samples.append((str(orig_val), str(clean_val)))
                    count += 1
        
        return samples


if __name__ == "__main__":
    # Test cases for JanitorModule
    print("Testing JanitorModule...")
    
    # Test case 1: Basic smoking pack-years data (like RADCURE dataset)
    print("\n1. Testing smoking pack-years data:")
    smoking_data = pd.DataFrame({
        'patient_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'smoking_py': ['30+', '<15', '25', '>=20', '~10', '45+', 'never', '18', '<5', '35+'],
        'age': [65, 45, 55, 60, 70, 50, 40, 75, 80, 55],
        'status': ['active', 'former', 'never', 'active', 'former', 'active', 'never', 'former', 'former', 'active']
    })
    
    print("Original data:")
    print(smoking_data)
    print(f"Original dtypes:\n{smoking_data.dtypes}")
    
    cleaning_module = JanitorModule.build()
    cleaned_data = cleaning_module(smoking_data)
    
    print("\nCleaned data:")
    print(cleaned_data)
    print(f"Cleaned dtypes:\n{cleaned_data.dtypes}")
    print(f"Cleaning report:\n{cleaning_module.report}")
    
    # Test case 2: Various auxiliary symbols including <= and >=
    print("\n2. Testing various auxiliary symbols (including <= and >=):")
    symbol_data = pd.DataFrame({
        'measurement_1': ['<10', '>=50', '<=5', '~25', '15+', '30-40', '100'],
        'measurement_2': ['>20', '<=15', '+/-3', '~8', '25+', '>=60', '75'],
        'text_col': ['yes', 'no', 'maybe', 'yes', 'no', 'yes', 'maybe']  # Should not be converted
    })
    
    print("Original data:")
    print(symbol_data)
    print(f"Original dtypes:\n{symbol_data.dtypes}")
    
    cleaning_module2 = JanitorModule.build()
    cleaned_symbol_data = cleaning_module2(symbol_data)
    
    print("\nCleaned data:")
    print(cleaned_symbol_data)
    print(f"Cleaned dtypes:\n{cleaned_symbol_data.dtypes}")
    print(f"Cleaning report:\n{cleaning_module2.report}")
    
    # Test case 3: Mixed data with missing values
    print("\n3. Testing mixed data with missing values:")
    mixed_data = pd.DataFrame({
        'dose': ['30+', None, '<15', 'N/A', '25', '>=20', 'missing', '18', '', '35+'],
        'response': ['CR', 'PR', 'SD', 'PD', 'CR', 'PR', 'SD', 'PD', 'CR', 'PR'],
        'numeric_col': [1.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0, 9.5, 10.0]  # Already numeric
    })
    
    print("Original data:")
    print(mixed_data)
    print(f"Original dtypes:\n{mixed_data.dtypes}")
    
    cleaning_module3 = JanitorModule.build()
    cleaned_mixed_data = cleaning_module3(mixed_data)
    
    print("\nCleaned data:")
    print(cleaned_mixed_data)
    print(f"Cleaned dtypes:\n{cleaned_mixed_data.dtypes}")
    print(f"Cleaning report:\n{cleaning_module3.report}")
    
    # Test case 4: Specify columns explicitly
    print("\n4. Testing with explicitly specified columns:")
    cleaning_module4 = JanitorModule.build(
        columns=['smoking_py'],  # Only clean this column
        auto_detect=False  # Don't auto-detect
    )
    
    cleaned_specific = cleaning_module4(smoking_data)
    print("Cleaned data (only smoking_py column):")
    print(cleaned_specific)
    print(f"Cleaning report:\n{cleaning_module4.report}")
    
    # Test case 5: Low conversion threshold
    print("\n5. Testing with low conversion threshold:")
    low_conversion_data = pd.DataFrame({
        'mixed_quality': ['30+', 'poor', '<15', 'good', '25', 'excellent', '>=20', 'fair', '~10', 'bad'],
        'mostly_text': ['hello', 'world', '5+', 'test', 'data', 'sample', '<3', 'more', 'text', 'here']
    })
    
    print("Original data:")
    print(low_conversion_data)
    
    # High threshold - should not convert mostly_text
    cleaning_module5a = JanitorModule.build(conversion_threshold=0.8)
    cleaned_high_thresh = cleaning_module5a(low_conversion_data)
    
    print(f"\nHigh threshold (0.8) - Cleaning report:\n{cleaning_module5a.report}")
    
    # Low threshold - should convert both
    cleaning_module5b = JanitorModule.build(conversion_threshold=0.2)
    cleaned_low_thresh = cleaning_module5b(low_conversion_data)
    
    print(f"Low threshold (0.2) - Cleaning report:\n{cleaning_module5b.report}")
    
    print("\nAll tests completed!")
