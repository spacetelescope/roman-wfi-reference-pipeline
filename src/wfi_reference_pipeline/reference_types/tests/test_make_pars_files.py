import subprocess
from pathlib import Path

import asdf
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "parameters" / "make_pars_files.py"

@pytest.fixture
def run_make_pars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _run():
        return subprocess.run(
            ["python", str(SCRIPT)],
            capture_output=True,
            text=True,
        )

    return _run, tmp_path

# ------------------------------------------------------------
# Test the script runs successfully
# ------------------------------------------------------------
def test_script_runs(run_make_pars):
    run, _ = run_make_pars

    result = run()

    assert result.returncode == 0, (
        f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ------------------------------------------------------------
# Test defaults ASDF file is created and readable
# ------------------------------------------------------------
def test_defaults_file_created(run_make_pars):
    run, tmp_path = run_make_pars

    run()

    defaults_file = tmp_path / "roman_elp_defaults.asdf"
    assert defaults_file.exists()

    af = asdf.open(str(defaults_file))
    assert "steps" in af.tree
    assert isinstance(af.tree["steps"], list)
    assert len(af.tree["steps"]) > 0


# ------------------------------------------------------------
# Test parameter files are generated
# ------------------------------------------------------------
def test_parameter_files_generated(run_make_pars):
    run, tmp_path = run_make_pars

    run()

    files = list(tmp_path.glob("roman_wfi_pars-*.asdf"))
    assert len(files) > 0


# ------------------------------------------------------------
# Test each generated file has required structure
# ------------------------------------------------------------
def test_generated_files_have_required_structure(run_make_pars):
    run, tmp_path = run_make_pars

    run()

    files = list(tmp_path.glob("roman_wfi_pars-*.asdf"))

    for f in files:
        af = asdf.open(f)

        tree = af.tree
        assert "meta" in tree
        assert "parameters" in tree
        assert "name" in tree

        assert isinstance(tree["parameters"], dict)


# ------------------------------------------------------------
# Test excluded parameters are removed
# ------------------------------------------------------------
def test_excluded_parameters_removed(run_make_pars):
    from wfi_reference_pipeline.reference_types.parameters.make_pars_files import (
        EXCLUDED_PARAMETERS,
    )

    run, tmp_path = run_make_pars

    run()

    files = list(tmp_path.glob("roman_wfi_pars-*.asdf"))

    for f in files:
        af = asdf.open(f)
        params = af.tree["parameters"]

        for excluded in EXCLUDED_PARAMETERS:
            assert excluded not in params


# ------------------------------------------------------------
# Test no empty or broken step entries
# ------------------------------------------------------------
def test_steps_are_valid(run_make_pars):
    run, tmp_path = run_make_pars

    run()

    defaults_file = tmp_path / "roman_elp_defaults.asdf"
    af = asdf.open(str(defaults_file))

    steps = af.tree["steps"]

    assert isinstance(steps, list)

    for step in steps:
        assert "name" in step
        assert "parameters" in step
        assert isinstance(step["parameters"], dict)