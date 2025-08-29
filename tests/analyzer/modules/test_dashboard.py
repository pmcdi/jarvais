"""Comprehensive test suite for the DashboardModule class.

This module provides unit and integration tests for the DashboardModule,
which is responsible for generating statistical dashboards from data analysis.

Test Organization:
- Fixtures: Reusable test data and configurations
- Initialization Tests: Module creation and configuration
- Core Functionality Tests: Main __call__ method behavior
- Property Tests: Accessor methods and state management
- Edge Case Tests: Boundary conditions and error handling
- Integration Tests: End-to-end workflow validation
- Performance Tests: Large dataset handling and efficiency

Note: Some tests use mocks to isolate functionality, while integration
tests verify the complete pipeline with minimal mocking.
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from jarvais.analyzer.modules.dashboard import DashboardModule


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with continuous and categorical columns.
    
    Returns:
        pd.DataFrame: 8-row DataFrame with 3 continuous and 3 categorical columns.
        Continuous: age (25-60), tumor_size (1.2-3.5), survival_rate (0.79-0.95)
        Categorical: gender (M/F), treatment_type (A/B), tumor_stage (I/II/III)
    """
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "tumor_size": [1.2, 2.3, 3.1, 1.8, 2.9, 3.5, 2.1, 1.9],
        "survival_rate": [0.95, 0.88, 0.82, 0.91, 0.85, 0.79, 0.87, 0.90],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "treatment_type": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "tumor_stage": ["I", "II", "III", "I", "II", "III", "I", "II"],
    })


@pytest.fixture
def dataframe_with_nulls():
    """Create a DataFrame with missing values for edge case testing.
    
    Returns:
        pd.DataFrame: DataFrame containing NaN values in various columns.
    """
    return pd.DataFrame({
        "age": [25, np.nan, 35, 40, np.nan, 50],
        "tumor_size": [1.2, 2.3, np.nan, 1.8, 2.9, np.nan],
        "survival_rate": [0.95, 0.88, 0.82, np.nan, 0.85, 0.79],
        "gender": ["M", "F", None, "F", "M", "F"],
        "treatment_type": ["A", None, "A", "B", "A", "B"],
        "tumor_stage": ["I", "II", "III", None, "II", "III"],
    })


@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for performance testing.
    
    Returns:
        pd.DataFrame: 10,000-row DataFrame with mixed column types.
    """
    np.random.seed(42)
    n_rows = 10000
    return pd.DataFrame({
        "age": np.random.randint(20, 80, n_rows),
        "tumor_size": np.random.uniform(0.5, 5.0, n_rows),
        "survival_rate": np.random.uniform(0.5, 1.0, n_rows),
        "biomarker_level": np.random.exponential(2.0, n_rows),
        "gender": np.random.choice(["M", "F"], n_rows),
        "treatment_type": np.random.choice(["A", "B", "C", "D"], n_rows),
        "tumor_stage": np.random.choice(["I", "II", "III", "IV"], n_rows),
        "response": np.random.choice(["Complete", "Partial", "None"], n_rows),
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def sample_significant_results():
    """Create sample significant results that would be returned by find_top_multiplots.
    
    Returns:
        List[Dict]: Two significant results with p-values < 0.05.
    """
    return [
        {
            "categorical_col": "gender",
            "continuous_col": "age",
            "p_value": 0.01,
            "effect_size": 0.35,
            "plot_path": "figures/multiplots/gender_age.png"
        },
        {
            "categorical_col": "treatment_type",
            "continuous_col": "tumor_size",
            "p_value": 0.03,
            "effect_size": 0.28,
            "plot_path": "figures/multiplots/treatment_type_tumor_size.png"
        }
    ]


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing log messages."""
    with patch("jarvais.analyzer.modules.dashboard.logger") as mock:
        yield mock


# ============================================================================
# MODULE INITIALIZATION TESTS
# ============================================================================

class TestModuleInitialization:
    """Test suite for DashboardModule initialization and configuration."""
    def test_build_with_default_parameters(self):
        """Test DashboardModule.build() creates module with default parameter values.
        
        Given: Required parameters only
        When: Module is built using build() class method
        Then: Module should have default values for optional parameters
        """
        module = DashboardModule.build(
            output_dir="test_output",
            continuous_columns=["age", "tumor_size"],
            categorical_columns=["gender", "treatment_type"]
        )
        
        assert module.output_dir == "test_output"
        assert module.continuous_columns == ["age", "tumor_size"]
        assert module.categorical_columns == ["gender", "treatment_type"]
        assert module.n_top == 10
        assert module.significance_threshold == 0.05
        assert module.enabled is True
        # Verify private attributes are initialized
        assert module._significant_results == []
        assert module._dashboard_plot_path is None


    def test_build_with_custom_parameters(self):
        """Test DashboardModule.build() accepts and stores custom parameter values.
        
        Given: All parameters with custom values including Path object
        When: Module is built with custom parameters
        Then: All custom values should be correctly stored
        """
        module = DashboardModule.build(
            output_dir=Path("/custom/output"),
            continuous_columns=["col1"],
            categorical_columns=["col2"],
            n_top=5,
            significance_threshold=0.01
        )
        
        assert module.output_dir == Path("/custom/output")
        assert isinstance(module.output_dir, Path)
        assert module.continuous_columns == ["col1"]
        assert module.categorical_columns == ["col2"]
        assert module.n_top == 5
        assert module.significance_threshold == 0.01

    def test_direct_initialization(self):
        """Test direct initialization using constructor instead of build().
        
        Given: Direct instantiation of DashboardModule
        When: Using __init__ directly
        Then: Module should be properly initialized
        """
        module = DashboardModule(
            output_dir="direct_output",
            continuous_columns=["c1", "c2"],
            categorical_columns=["cat1"]
        )
        
        assert module.output_dir == "direct_output"
        assert module.continuous_columns == ["c1", "c2"]
        assert module.categorical_columns == ["cat1"]
        assert module.enabled is True

    @pytest.mark.parametrize("output_dir,expected_type", [
        ("string_path", str),
        (Path("path_object"), Path),
        (Path("/absolute/path"), Path),
    ])
    def test_output_dir_type_handling(self, output_dir, expected_type):
        """Test that output_dir handles both string and Path types correctly.
        
        Given: Various output_dir types (string, relative Path, absolute Path)
        When: Module is initialized
        Then: output_dir should maintain its type
        """
        module = DashboardModule.build(
            output_dir=output_dir,
            continuous_columns=["c1"],
            categorical_columns=["cat1"]
        )
        
        assert isinstance(module.output_dir, expected_type)
        assert str(module.output_dir) == str(output_dir)


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

class TestCoreCallFunctionality:
    """Test suite for the main __call__ method of DashboardModule."""
    @patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
    @patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
    def test_call_basic_workflow(
        self,
        mock_find_top_multiplots,
        mock_plot_dashboard,
        sample_dataframe,
        sample_significant_results,
        tmp_path
    ):
        """Test basic call workflow executes all steps correctly.
        
        Given: Valid DataFrame and module configuration
        When: Module is called with DataFrame
        Then: Should find significant results, generate dashboard, and return DataFrame unchanged
        """
        # Setup mocks
        mock_find_top_multiplots.return_value = sample_significant_results
        mock_plot_dashboard.return_value = tmp_path / "dashboard.png"
        
        # Create module
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age", "tumor_size", "survival_rate"],
            categorical_columns=["gender", "treatment_type", "tumor_stage"]
        )
        
        # Store original DataFrame for comparison
        original_df = sample_dataframe.copy()
        
        # Call module
        result_df = module(sample_dataframe)
        
        # Verify DataFrame is returned unchanged and not modified in place
        assert result_df is sample_dataframe
        assert_frame_equal(result_df, original_df)
        
        # Verify find_top_multiplots was called with correct arguments
        mock_find_top_multiplots.assert_called_once_with(
            data=sample_dataframe,
            categorical_columns=["gender", "treatment_type", "tumor_stage"],
            continuous_columns=["age", "tumor_size", "survival_rate"],
            output_dir=tmp_path,
            n_top=10,
            significance_threshold=0.05
        )
        
        # Verify plot_dashboard was called with correct arguments
        mock_plot_dashboard.assert_called_once()
        call_args = mock_plot_dashboard.call_args
        assert call_args.args[0] == sample_significant_results
        assert_frame_equal(call_args.args[1], sample_dataframe)
        assert call_args.args[2] == tmp_path / "figures"
        
        # Verify module state is updated
        assert module.significant_results == sample_significant_results
        assert module.dashboard_plot_path == tmp_path / "dashboard.png"


@patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_call_with_original_data(
    mock_find_top_multiplots,
    mock_plot_dashboard,
    sample_dataframe,
    sample_significant_results,
    tmp_path
):
    """Test call with separate original_data parameter."""
    mock_find_top_multiplots.return_value = sample_significant_results
    mock_plot_dashboard.return_value = tmp_path / "dashboard.png"
    
    module = DashboardModule.build(
        output_dir=tmp_path,
        continuous_columns=["age"],
        categorical_columns=["gender"]
    )
    
    # Create a modified DataFrame to pass as the first argument
    modified_df = sample_dataframe.copy()
    modified_df["new_col"] = 1
    
    result_df = module(modified_df, original_data=sample_dataframe)
    
    # Verify that original_data was passed to find_top_multiplots
    mock_find_top_multiplots.assert_called_once()
    call_args = mock_find_top_multiplots.call_args
    pd.testing.assert_frame_equal(call_args.kwargs["data"], sample_dataframe)
    
    # Result should be the modified DataFrame
    pd.testing.assert_frame_equal(result_df, modified_df)


# ============================================================================
# PROPERTY ACCESSOR TESTS
# ============================================================================

class TestPropertyAccessors:
    """Test suite for property accessors and state management."""
    
    def test_significant_results_property(self):
        """Test significant_results property returns correct private attribute value.
        
        Given: Module with _significant_results private attribute
        When: Accessing significant_results property
        Then: Should return the value of _significant_results
        """
        module = DashboardModule.build(
            output_dir="test",
            continuous_columns=["col1"],
            categorical_columns=["col2"]
        )
        
        # Initial state should be empty list
        assert module.significant_results == []
        assert isinstance(module.significant_results, list)
        
        # Set private attribute and verify property reflects change
        test_results = [{"test": "data", "p_value": 0.01}]
        module._significant_results = test_results
        assert module.significant_results == test_results
        assert module.significant_results is test_results  # Should be same object


    def test_dashboard_plot_path_property(self):
        """Test dashboard_plot_path property returns correct private attribute value.
        
        Given: Module with _dashboard_plot_path private attribute
        When: Accessing dashboard_plot_path property
        Then: Should return the value of _dashboard_plot_path
        """
        module = DashboardModule.build(
            output_dir="test",
            continuous_columns=["col1"],
            categorical_columns=["col2"]
        )
        
        # Initial state should be None
        assert module.dashboard_plot_path is None
        
        # Set private attribute and verify property reflects change
        test_path = Path("/test/path.png")
        module._dashboard_plot_path = test_path
        assert module.dashboard_plot_path == test_path
        assert isinstance(module.dashboard_plot_path, Path)


# ============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# ============================================================================

class TestEdgeCasesAndErrors:
    """Test suite for edge cases, error conditions, and disabled states."""
    @patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
    @patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
    def test_module_disabled(
        self,
        mock_find_top_multiplots,
        mock_plot_dashboard,
        sample_dataframe,
        mock_logger
    ):
        """Test that disabled module returns DataFrame without processing.
        
        Given: Module with enabled=False
        When: Module is called
        Then: Should return DataFrame unchanged and log warning
        """
        module = DashboardModule(
            enabled=False,
            output_dir="test",
            continuous_columns=["age"],
            categorical_columns=["gender"]
        )
        
        # Store original for comparison
        original_df = sample_dataframe.copy()
        
        result_df = module(sample_dataframe)
        
        # DataFrame should be returned unchanged
        assert_frame_equal(result_df, sample_dataframe)
        assert_frame_equal(result_df, original_df)
        
        # Processing functions should not be called
        mock_find_top_multiplots.assert_not_called()
        mock_plot_dashboard.assert_not_called()
        
        # Warning should be logged
        mock_logger.warning.assert_called_once_with("Dashboard is disabled.")
        
        # State should remain initial
        assert module.significant_results == []
        assert module.dashboard_plot_path is None


    def test_empty_dataframe(self, empty_dataframe, tmp_path):
        """Test module handles empty DataFrame gracefully.
        
        Given: Empty DataFrame
        When: Module processes the DataFrame
        Then: Should return empty DataFrame and have empty results
        """
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["col1"],
            categorical_columns=["col2"]
        )
        
        with patch("jarvais.analyzer.modules.dashboard.find_top_multiplots") as mock_find:
            mock_find.return_value = []
            result_df = module(empty_dataframe)
            
            # Should return empty DataFrame unchanged
            assert_frame_equal(result_df, empty_dataframe)
            assert len(result_df) == 0
            assert module.significant_results == []
            assert module.dashboard_plot_path is None


@patch("jarvais.analyzer.modules.dashboard.logger")
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_no_significant_results(
    mock_find_top_multiplots,
    mock_logger,
    sample_dataframe,
    tmp_path
):
    """Test when no significant results are found."""
    mock_find_top_multiplots.return_value = []
    
    module = DashboardModule.build(
        output_dir=tmp_path,
        continuous_columns=["age"],
        categorical_columns=["gender"]
    )
    
    result_df = module(sample_dataframe)
    
    # DataFrame should be returned unchanged
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    
    # Should log warning about no significant results
    mock_logger.warning.assert_called_once_with(
        "No significant results found for dashboard plot. Skipping dashboard image generation."
    )
    assert module.dashboard_plot_path is None


# Test exception handling
@patch("jarvais.analyzer.modules.dashboard.logger")
@patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_plot_generation_failure(
    mock_find_top_multiplots,
    mock_plot_dashboard,
    mock_logger,
    sample_dataframe,
    sample_significant_results,
    tmp_path
):
    """Test graceful handling of plot generation failure."""
    mock_find_top_multiplots.return_value = sample_significant_results
    mock_plot_dashboard.side_effect = Exception("Plot generation failed")
    
    module = DashboardModule.build(
        output_dir=tmp_path,
        continuous_columns=["age"],
        categorical_columns=["gender"]
    )
    
    result_df = module(sample_dataframe)
    
    # DataFrame should still be returned
    pd.testing.assert_frame_equal(result_df, sample_dataframe)
    
    # Should have significant results but no plot path
    assert module.significant_results == sample_significant_results
    assert module.dashboard_plot_path is None
    
    # Should log warning about failure
    mock_logger.warning.assert_called_once_with(
        "Failed to generate dashboard plot: Plot generation failed"
    )


# Test integration with dependencies
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_find_top_multiplots_integration(
    mock_find_top_multiplots,
    sample_dataframe,
    tmp_path
):
    """Test that correct parameters are passed to find_top_multiplots."""
    mock_find_top_multiplots.return_value = []
    
    continuous_cols = ["age", "tumor_size"]
    categorical_cols = ["gender", "treatment_type"]
    
    module = DashboardModule.build(
        output_dir=tmp_path,
        continuous_columns=continuous_cols,
        categorical_columns=categorical_cols,
        n_top=7,
        significance_threshold=0.01
    )
    
    module(sample_dataframe)
    
    # Verify correct arguments were passed
    mock_find_top_multiplots.assert_called_once_with(
        data=sample_dataframe,
        categorical_columns=categorical_cols,
        continuous_columns=continuous_cols,
        output_dir=tmp_path,
        n_top=7,
        significance_threshold=0.01
    )


@patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_plot_dashboard_integration(
    mock_find_top_multiplots,
    mock_plot_dashboard,
    sample_dataframe,
    sample_significant_results,
    tmp_path
):
    """Test that correct parameters are passed to plot_dashboard."""
    mock_find_top_multiplots.return_value = sample_significant_results
    mock_plot_dashboard.return_value = tmp_path / "dashboard.png"
    
    module = DashboardModule.build(
        output_dir=tmp_path,
        continuous_columns=["age"],
        categorical_columns=["gender"]
    )
    
    module(sample_dataframe)
    
    # Verify correct arguments were passed to plot_dashboard
    mock_plot_dashboard.assert_called_once()
    call_args = mock_plot_dashboard.call_args
    
    assert call_args.args[0] == sample_significant_results
    pd.testing.assert_frame_equal(call_args.args[1], sample_dataframe)
    assert call_args.args[2] == tmp_path / "figures"


# Test output directory creation
@patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
@patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
def test_dashboard_module_creates_figures_directory(
    mock_find_top_multiplots,
    mock_plot_dashboard,
    sample_dataframe,
    sample_significant_results,
    tmp_path
):
    """Test that figures directory is created under output_dir."""
    mock_find_top_multiplots.return_value = sample_significant_results
    mock_plot_dashboard.return_value = tmp_path / "figures" / "dashboard.png"
    
    output_dir = tmp_path / "test_output"
    
    module = DashboardModule.build(
        output_dir=output_dir,
        continuous_columns=["age"],
        categorical_columns=["gender"]
    )
    
    module(sample_dataframe)
    
    # Verify figures directory was created
    figures_dir = output_dir / "figures"
    assert figures_dir.exists()
    assert figures_dir.is_dir()


# Test with Path object as output_dir
def test_dashboard_module_with_path_object(tmp_path):
    """Test that module works correctly with Path object as output_dir."""
    output_path = tmp_path / "output"
    
    module = DashboardModule.build(
        output_dir=output_path,
        continuous_columns=["col1"],
        categorical_columns=["col2"]
    )
    
    assert isinstance(module.output_dir, Path)
    assert module.output_dir == output_path


    def test_dataframe_with_nulls(self, dataframe_with_nulls, tmp_path):
        """Test module handles DataFrames with null values.
        
        Given: DataFrame containing NaN and None values
        When: Module processes the DataFrame
        Then: Should process without errors
        """
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age", "tumor_size"],
            categorical_columns=["gender", "treatment_type"]
        )
        
        with patch("jarvais.analyzer.modules.dashboard.find_top_multiplots") as mock_find:
            mock_find.return_value = []
            
            # Should not raise exception
            result_df = module(dataframe_with_nulls)
            
            # Verify function was called with DataFrame containing nulls
            mock_find.assert_called_once()
            call_args = mock_find.call_args
            assert_frame_equal(call_args.kwargs["data"], dataframe_with_nulls)
            
            # Result should be unchanged
            assert_frame_equal(result_df, dataframe_with_nulls)


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test suite for input parameter validation and error handling."""
    
    @pytest.mark.parametrize("columns,error_msg", [
        (["col1", "col1", "col2"], "duplicate"),  # Duplicate columns
        (["col-1", "col@2"], "special"),  # Special characters
        (["" * 100], "long"),  # Very long column name
    ])
    def test_column_names_edge_cases(self, columns, error_msg, tmp_path):
        """Test module handles various column name edge cases.
        
        Given: Various problematic column names
        When: Module is initialized
        Then: Should handle gracefully
        """
        # Module should accept these without validation errors at init
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=columns,
            categorical_columns=["cat1"]
        )
        
        assert module.continuous_columns == columns
    
    def test_mismatched_columns(self, sample_dataframe, tmp_path):
        """Test module handles column names not present in DataFrame.
        
        Given: Module configured with columns not in DataFrame
        When: Module processes DataFrame
        Then: Should handle the mismatch appropriately
        """
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["nonexistent_col1", "nonexistent_col2"],
            categorical_columns=["missing_cat1"]
        )
        
        with patch("jarvais.analyzer.modules.dashboard.find_top_multiplots") as mock_find:
            mock_find.return_value = []
            
            # Should pass columns to find_top_multiplots regardless
            result_df = module(sample_dataframe)
            
            # Verify the specified columns were passed
            mock_find.assert_called_once()
            call_args = mock_find.call_args
            assert call_args.kwargs["continuous_columns"] == ["nonexistent_col1", "nonexistent_col2"]
            assert call_args.kwargs["categorical_columns"] == ["missing_cat1"]
    
    @pytest.mark.parametrize("n_top,threshold", [
        (0, 0.05),  # Zero n_top
        (1000, 0.05),  # Very large n_top
        (10, 0.0),  # Zero threshold
        (10, 1.0),  # Threshold of 1.0
        (10, 0.000001),  # Very small threshold
    ])
    def test_parameter_boundary_values(self, n_top, threshold, tmp_path):
        """Test module handles boundary values for parameters.
        
        Given: Boundary values for n_top and significance_threshold
        When: Module is initialized
        Then: Should accept values without error
        """
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["c1"],
            categorical_columns=["cat1"],
            n_top=n_top,
            significance_threshold=threshold
        )
        
        assert module.n_top == n_top
        assert module.significance_threshold == threshold


# ============================================================================
# MODULE STATE PERSISTENCE TESTS
# ============================================================================

class TestStatePersistence:
    """Test suite for module state persistence across calls."""
    @patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
    @patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
    def test_state_updates_between_calls(
        self,
        mock_find_top_multiplots,
        mock_plot_dashboard,
        sample_dataframe,
        sample_significant_results,
        tmp_path
    ):
        """Test that module state updates correctly across multiple calls.
        
        Given: Module called multiple times with different results
        When: Each call returns different significant results
        Then: Module state should update to reflect most recent call
        """
        # First call setup
        mock_find_top_multiplots.return_value = sample_significant_results
        mock_plot_dashboard.return_value = tmp_path / "dashboard1.png"
        
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age"],
            categorical_columns=["gender"]
        )
        
        # First call
        module(sample_dataframe)
        first_results = module.significant_results.copy()
        first_path = module.dashboard_plot_path
        
        # Verify first call state
        assert module.significant_results == sample_significant_results
        assert module.dashboard_plot_path == tmp_path / "dashboard1.png"
        
        # Second call with different results
        new_results = [
            {"categorical_col": "new", "continuous_col": "new", "p_value": 0.001, "effect_size": 0.5}
        ]
        mock_find_top_multiplots.return_value = new_results
        mock_plot_dashboard.return_value = tmp_path / "dashboard2.png"
        
        module(sample_dataframe)
        
        # State should be completely updated
        assert module.significant_results == new_results
        assert module.dashboard_plot_path == tmp_path / "dashboard2.png"
        assert module.significant_results != first_results
        assert module.dashboard_plot_path != first_path
        
        # Verify call counts
        assert mock_find_top_multiplots.call_count == 2
        assert mock_plot_dashboard.call_count == 2


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test suite for performance and resource usage."""
    
    @patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
    @patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
    def test_large_dataframe_handling(self, mock_find, mock_plot, large_dataframe, tmp_path):
        """Test module handles large DataFrames efficiently.
        
        Given: Large DataFrame with 10,000 rows
        When: Module processes the DataFrame
        Then: Should complete within reasonable time
        """
        mock_find.return_value = []
        
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age", "tumor_size", "survival_rate", "biomarker_level"],
            categorical_columns=["gender", "treatment_type", "tumor_stage", "response"]
        )
        
        start_time = time.time()
        result_df = module(large_dataframe)
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (mocked functions)
        assert elapsed_time < 1.0  # Should be much faster with mocks
        
        # Verify DataFrame unchanged
        assert result_df is large_dataframe
        assert len(result_df) == 10000
    
    def test_multiple_calls_no_memory_leak(self, sample_dataframe, tmp_path):
        """Test repeated calls don't cause memory leaks.
        
        Given: Module called repeatedly
        When: Called 100 times
        Then: Should not accumulate state unnecessarily
        """
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age"],
            categorical_columns=["gender"]
        )
        
        with patch("jarvais.analyzer.modules.dashboard.find_top_multiplots") as mock_find:
            mock_find.return_value = [{"test": i} for i in range(5)]
            
            for i in range(100):
                module(sample_dataframe)
            
            # State should only contain last results
            assert len(module.significant_results) == 5
            assert mock_find.call_count == 100


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with minimal mocking."""
    
    def test_directory_creation(self, sample_dataframe, tmp_path):
        """Test that figures directory is created when it doesn't exist.
        
        Given: Output directory without figures subdirectory
        When: Module processes data with results
        Then: Should create figures directory
        """
        output_dir = tmp_path / "test_output"
        output_dir.mkdir()
        
        module = DashboardModule.build(
            output_dir=output_dir,
            continuous_columns=["age"],
            categorical_columns=["gender"]
        )
        
        with patch("jarvais.analyzer.modules.dashboard.find_top_multiplots") as mock_find:
            with patch("jarvais.analyzer.modules.dashboard.plot_dashboard") as mock_plot:
                mock_find.return_value = [{"test": "result"}]
                mock_plot.return_value = output_dir / "figures" / "dashboard.png"
                
                module(sample_dataframe)
                
                # Verify figures directory was created
                figures_dir = output_dir / "figures"
                assert figures_dir.exists()
                assert figures_dir.is_dir()
    
    @patch("jarvais.analyzer.modules.dashboard.plot_dashboard")
    @patch("jarvais.analyzer.modules.dashboard.find_top_multiplots")
    def test_full_pipeline_with_original_data(
        self,
        mock_find,
        mock_plot,
        sample_dataframe,
        tmp_path
    ):
        """Test complete pipeline with separate original_data parameter.
        
        Given: Modified DataFrame and original DataFrame
        When: Module called with both parameters
        Then: Should use original_data for analysis but return modified DataFrame
        """
        # Create modified DataFrame
        modified_df = sample_dataframe.copy()
        modified_df["new_column"] = range(len(modified_df))
        modified_df["age"] = modified_df["age"] * 2  # Modify existing column
        
        mock_find.return_value = [{"result": "test"}]
        mock_plot.return_value = tmp_path / "dashboard.png"
        
        module = DashboardModule.build(
            output_dir=tmp_path,
            continuous_columns=["age"],
            categorical_columns=["gender"]
        )
        
        result_df = module(modified_df, original_data=sample_dataframe)
        
        # Verify original_data was used for analysis
        mock_find.assert_called_once()
        call_args = mock_find.call_args
        assert_frame_equal(call_args.kwargs["data"], sample_dataframe)
        
        # Verify modified DataFrame was returned
        assert_frame_equal(result_df, modified_df)
        assert "new_column" in result_df.columns
        assert result_df["age"].iloc[0] == modified_df["age"].iloc[0]
