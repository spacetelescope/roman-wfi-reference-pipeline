# WFI Reference Pipeline: New Reference Type Guide

This document provides guidance on:

- Creating a new reference file type (`{RefType}`)
- Writing consistent tests (`test_{ref_type}.py`)
- Following pipeline conventions

---

# Creating a New Reference File Type

This guide outlines the steps required to add a new reference file type to the WFI Reference Pipeline.

For this example, the new reference file type will be called:

```
NEW_EFFECT
```

---

## Step-by-Step Instructions

### 1. Define the Reference Type Constant

Update:

```
src/wfi_reference_pipeline/constants.py
```

- Under **Ref Type Strings**, add:

```python
REF_TYPE_NEW_EFFECT = "NEW_EFFECT"
```

- Add the new type to the `WFI_REF_TYPES` dictionary.

---

### 2. Create the Metadata Class

Create:

```
src/wfi_reference_pipeline/resources/wfi_meta_new_effect.py
```

```python
class WFIMetaNewEffect(WFIMetadata):
```

- Add additional metadata in `__post_init__()` if needed.

---

### 3. Add Development Metadata Support

Update:

```
src/wfi_reference_pipeline/resources/make_dev_meta.py
```

- Add:

```python
_create_dev_meta_new_effect()
```

- In `__init__`:

```python
if ref_type == "NEW_EFFECT":
```

---

### 4. Add Testing Metadata Support

Update:

```
src/wfi_reference_pipeline/resources/make_test_meta.py
```

- Add:

```python
_create_test_meta_new_effect()
```

- In `__init__`:

```python
if ref_type == "NEW_EFFECT":
```

---

### 5. Create the Reference Type Module

Create:

```
src/wfi_reference_pipeline/reference_types/new_effect/
```

Inside:

- `new_effect.py`
- `__init__.py`

```python
class NewEffect(ReferenceType):
```

- Validate metadata against `WFIMetaNewEffect`

---

### 6. Handle Input Data Requirements

If no input data is required, update:

```python
WFI_REF_TYPES_WITHOUT_INPUT_DATA
```

in `constants.py`.

---

### 7. Implement the Class

- Inherit from `ReferenceType`
- Implement required methods:
  - Initialization
  - Processing
  - Data model population

---

### 8. Add Tests

Create:

```
src/wfi_reference_pipeline/reference_types/tests/test_new_effect.py
```

Update:

```
test_rfp_schema.py
```

Add:

```python
test_rfp_new_effect_schema()
```

---

## Testing Requirements

Tests should verify:

- Valid instantiation
- Metadata validation
- Input validation
- Output structure
- Correct execution flow
