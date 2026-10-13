import subprocess
from glob import glob

import asdf
from astropy.time import Time

# Parameters that are part of the generic stpipe infrastructure
# and should NOT be included in CRDS science parameter reference files.
EXCLUDED_PARAMETERS = {
    "pre_hooks",
    "post_hooks",
    "output_file",
    "output_dir",
    "output_ext",
    "output_use_model",
    "output_use_index",
    "save_results",
    "search_output_file",
    "input_dir",
    "skip",
    "suffix",
    "update_version",
}

# ----------------------------------------------------------------------
# Generate a temporary ASDF file containing the current pipeline defaults.
#
# We intentionally pull defaults directly from the installed romancal
# codebase rather than hard-coding them. This ensures newly added
# parameters or changed defaults are automatically captured.
# ----------------------------------------------------------------------
result = subprocess.run(
    [
        "strun",
        "roman_elp",
        "--save-parameters",
        "roman_elp_defaults.asdf",
    ],
    capture_output=True,
    text=True,
)

print("RETURN CODE:", result.returncode)
print("\nSTDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)

# Open the generated defaults file and access the individual pipeline steps.
default_filename = "roman_elp_defaults.asdf"
af = asdf.open(default_filename)
steps = af.tree["steps"]

# Standardized metadata date for all generated parameter reference files.
DATE = Time("2026-09-01T00:00:00", scale="utc").isot


def make_meta(reftype, title, description):
    """
    Construct the CRDS-style metadata block used by all parameter
    reference files.

    Parameters
    ----------
    reftype : str
        CRDS reference type (e.g. pars-rampfitstep).

    title : str
        Human-readable title stored in metadata.

    description : str
        Description of the parameter file contents.

    Returns
    -------
    dict
        Metadata dictionary to be written into the ASDF reference file.
    """
    return {
        "author": "Roman Calibration Pipeline",
        "date": DATE,
        "description": description,
        "exposure": {
            "p_exptype": "WFI_IMAGE|WFI_GRISM|WFI_PRISM|",
            "type": "WFI_IMAGE",
        },
        "instrument": {
            "detector": "WFI01",
            "name": "WFI",
            "optical_element": "F062",
            "p_detector": (
                "WFI01|WFI02|WFI03|WFI04|"
                "WFI05|WFI06|WFI07|WFI08|"
                "WFI09|WFI10|WFI11|WFI12|"
                "WFI13|WFI14|WFI15|WFI16|"
                "WFI17|WFI18|"
            ),
            "p_optical_element": (
                "F062|F087|F106|F129|"
                "F146|F158|F184|F213|"
                "GRISM|PRISM|DARK|"
            ),
        },
        "origin": "STSCI",
        "pedigree": "DUMMY",
        "reftype": reftype,
        "telescope": "ROMAN",
        "title": title,
        "useafter": DATE,
    }


# Re-open the saved defaults file and iterate through each pipeline step.
af = asdf.open("roman_elp_defaults.asdf")

for step in af.tree["steps"]:

    # Step identifier used by the pipeline (e.g. "rampfit").
    step_name = step["name"]

    # Fully-qualified Python class implementing the step.
    step_class = step["class"]

    # Retain only science-facing parameters.
    # Generic stpipe bookkeeping parameters are removed.
    science_pars = {
        k: v
        for k, v in step["parameters"].items()
        if k not in EXCLUDED_PARAMETERS
    }

    # Construct the CRDS reftype name.
    reftype = f"pars-{step_name.replace('_', '')}step"

    # Assemble the ASDF tree matching the existing Roman
    # parameter reference file structure.
    tree = {
        "class": step_class,
        "meta": make_meta(
            reftype=reftype,
            title=f"{step_name} Parameters",
            description=f"Parameter File for WFI Romancal {step_name}",
        ),
        "name": step_name,
        "parameters": science_pars,
    }

    # Output filename for the generated parameter reference file.
    outfile = f"roman_wfi_{reftype}_BuildXX.asdf"

    # Write the reference file.
    asdf.AsdfFile(tree).write_to(outfile)

    print(f"Wrote {outfile}")


# ----------------------------------------------------------------------
# Verification step.
#
# Open every generated parameter file and print a short summary
# showing the step name and number of science parameters written.
# ----------------------------------------------------------------------


for filename in sorted(glob("roman_wfi_pars-*.asdf")):
    with asdf.open(filename) as af:
        print(
            filename,
            af.tree["name"],
            len(af.tree["parameters"]),
        )