# Creating and Editing a PARS (ASDF Parameter) File

This guide explains how to generate and modify a `pars` file for the Roman WFI pipeline using `asdf`.

---

## 1. Generate a Default PARS File

In a new environment, run:

```bash
strun roman_elp --save-parameters roman_elp_defaults.asdf
```

---

## 2. Help Command

To inspect available options:

```bash
strun -h
```

Example usage:

```
usage: strun [--debug] [--save-parameters SAVE_PARAMETERS] [--disable-crds-steppars]
             [--logcfg LOGCFG] [--verbose] [--log-level LOG_LEVEL]
             [--log-file LOG_FILE] [--log-stream]
```

---

## 3. Load the PARS File in Python

```python
import asdf

af = asdf.open('roman_elp_defaults.asdf')
```

---

## 4. Inspect Metadata

```python
af.tree['meta']
```

Example output:

```python
{
 'author': '<SPECIFY>',
 'date': '2026-02-24T20:07:08',
 'description': 'Parameters for calibration step romancal.pipeline.exposure_pipeline.ExposurePipeline',
 'instrument': {'name': 'WFI'},
 'origin': '<SPECIFY>',  # Allowed values are: "STSCI", "STSCI/SOC", "IPAC/SSC"
 'pedigree': '<SPECIFY>',  # Per Technical Report MESA2021-01 allowed values are "DUMMY", "MODEL", "GROUND", or "INFLIGHT"
 'reftype': '<SPECIFY>',
 'telescope': 'ROMAN',
 'useafter': '<SPECIFY>'
}
```

---

## 5. Update Metadata

Example for a FLAT pipeline configuration:

```python
af.tree["meta"]["author"] = "Richard G. Cosentino"
af.tree["meta"]["date"] = "2026-02-23T18:53:33"
af.tree["meta"]["description"] = (
    "Parameters for ExposurePipeline for WFI FLAT processing through assign wcs step."
)
af.tree["meta"]["instrument"]["name"] = "WFI"
af.tree["meta"]["origin"] = "STScI"
af.tree["meta"]["pedigree"] = "GROUND"
af.tree["meta"]["reftype"] = "pars-exposurepipeline"
af.tree["meta"]["telescope"] = "ROMAN"
af.tree["meta"]["useafter"] = "2025-08-01T00:00:00"
```

---

## 6. Add Exposure Metadata

```python
af.tree["meta"]["exposure"] = {"type": "WFI_FLAT"}
```

---

## 7. Inspect Pipeline Steps

```python
af.tree["steps"]
```

Example structure:

```python
[
  {
    "class": "romancal.dq_init.dq_init_step.DQInitStep",
    "name": "dq_init",
    "parameters": {
      "input_dir": "",
      "output_dir": None,
      "output_ext": ".asdf",
      "output_file": None,
      "output_use_index": True,
      "output_use_model": False,
      "post_hooks": [],
      "pre_hooks": [],
      "save_results": False,
      "search_output_file": True,
      "skip": False,
      "suffix": None,
      "update_version": False
    }
  },
  {
    "class": "romancal.saturation.saturation_step.SaturationStep",
    "name": "saturation",
    "parameters": {
      "input_dir": "",
      "output_dir": None,
      "output_ext": ".asdf",
      "output_file": None,
      "output_use_index": True,
      "output_use_model": False,
      "post_hooks": [],
      "pre_hooks": [],
      "save_results": False,
      "search_output_file": True,
      "skip": False,
      "suffix": None
    }
  }
]
```

---

## 8. Modify Pipeline Steps

Skip selected steps:

```python
for step in af.tree["steps"]:
    if step["name"] in ["flatfield", "photom", "source_catalog", "tweakreg"]:
        step["parameters"]["skip"] = True
```

---

## 9. Write the Modified PARS File

```python
af.write_to("pars-exposurepipeline.asdf")
```

---

## Summary

This workflow:

- Generates a default PARS file with `strun`
- Opens and inspects ASDF structure
- Updates metadata
- Modifies pipeline step execution
- Writes out a customized parameter file
```