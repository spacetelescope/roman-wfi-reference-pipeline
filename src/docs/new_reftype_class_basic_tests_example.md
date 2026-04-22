# Reference Type Testing Guide

This document outlines how to construct a basic test file for a new reference type (`{RefType}`) in the WFI Reference Pipeline.

Each reference type should include a corresponding test file: test_{ref_type}.py


---

## Testing Overview

Tests should follow a consistent structure across all reference types and focus on validating:

- Correct object instantiation
- Metadata validation
- Input data validation
- Data model population
- Output file naming

Fixtures should be used to provide reusable test inputs. Pytest fixtures are reusable setup functions, defined with the @pytest.fixture decorator, that provide test data or state to your tests in a clean and modular way. They help reduce duplication by allowing multiple tests to share common initialization logic.

---

## Required Fixtures

### Metadata Fixture

Always provide metadata that matches the reference type:

```python
@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMeta{RefType} metadata."""
```
## 2D Data Fixture

### Most reference types use a 2D image:

```python
@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid 2D reference type data."""
```

## Object Fixture

### Instantiate the {RefType} object with valid inputs:

```python
@pytest.fixture
def {ref_type}_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a {RefType} object with valid data."""
```

## Core Tests

### Instantiation with Valid Data
- **Object should be created successfully**
- **{ref_type}_image should be initialized with correct shape**

## Invalid Metadata
- **Passing incorrect metadata should raise TypeError**

## Invalid Data
- **Passing invalid data should raise TypeError**

## Data Model Population
- **populate_datamodel_tree() should return:**
  - meta
  - data
- **Data should have correct shape and dtype**

## Default Output Filename
- **Should follow convention:**

## Example Test Template
```python
import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_{REF_TYPE},
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.{ref_type}.{ref_type} import {RefType}
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMeta{RefType} metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_{REF_TYPE})
    return test_meta.meta_{ref_type}


@pytest.fixture
def valid_ref_type_data_array():
    """Fixture for generating valid reference type data."""
    return np.random.random(
        (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
    )


@pytest.fixture
def {ref_type}_object_with_data_array(valid_meta_data, valid_ref_type_data_array):
    """Fixture for initializing a {RefType} object with valid data."""
    obj = {RefType}(
        meta_data=valid_meta_data,
        ref_type_data=valid_ref_type_data_array
    )
    yield obj


class Test{RefType}:

    def test_{ref_type}_instantiation_with_valid_ref_type_data(self, {ref_type}_object_with_data_array):
        """Test successful instantiation with valid input data."""
        assert isinstance({ref_type}_object_with_data_array, {RefType})
        assert {ref_type}_object_with_data_array.{ref_type}_image.shape == (
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )

    def test_{ref_type}_instantiation_with_invalid_metadata(self, valid_ref_type_data_array):
        """Test that invalid metadata raises TypeError."""
        bad_test_meta = MakeTestMeta(ref_type=REF_TYPE_READNOISE)
        with pytest.raises(TypeError):
            {RefType}(
                meta_data=bad_test_meta.meta_readnoise,
                ref_type_data=valid_ref_type_data_array
            )

    def test_{ref_type}_instantiation_with_invalid_ref_type_data(self, valid_meta_data):
        """Test that invalid reference type data raises TypeError."""
        with pytest.raises(TypeError):
            {RefType}(meta_data=valid_meta_data, ref_type_data="invalid_ref_data")

    def test_populate_datamodel_tree(self, {ref_type}_object_with_data_array):
        """Test that the data model tree is correctly populated."""
        data_model_tree = {ref_type}_object_with_data_array.populate_datamodel_tree()

        assert "meta" in data_model_tree
        assert "data" in data_model_tree

        assert data_model_tree["data"].shape == (
            DETECTOR_PIXEL_X_COUNT,
            DETECTOR_PIXEL_Y_COUNT,
        )
        assert data_model_tree["data"].dtype == np.float32

    def test_{ref_type}_outfile_default(self, {ref_type}_object_with_data_array):
        """Test that the default outfile name is correct."""
        assert {ref_type}_object_with_data_array.outfile == "roman_{ref_type}.asdf"

```