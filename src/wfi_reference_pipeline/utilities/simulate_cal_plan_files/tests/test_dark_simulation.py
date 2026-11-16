# tests/test_dark_simulation.py

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan import (
    BaseDarkSimulation,
    LongDarkSimulation,
    WeeklyDarkSimulation,
)

# ============================================================
# Fixtures
# ============================================================


# Stand-in read patterns so tests never hit CRDS.
FAKE_READ_PATTERNS = {
    1010: [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12], [13], [14, 15], [16]],
    9003: [[i] for i in range(1, 56)],
}

class FakeMATableHandler:
    def _get_table_specific_info(self, ma_table_id=9010):
        return FAKE_READ_PATTERNS[ma_table_id], None
    
@pytest.fixture(autouse=True)
def fake_ma_table_handler(monkeypatch):
    monkeypatch.setattr(BaseDarkSimulation, "_ma_table_handler", FakeMATableHandler())

@pytest.fixture(autouse=True)
def fake_rate_image(monkeypatch):
    """Small, seed-tagged rate images: value = seed, plus the params used to make it."""
    made = []
    def fake_make(seed=None, **params):
        made.append({"seed": seed, **params})
        return np.full((4, 4), float(seed))
    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.make_dark_rate_image",
        fake_make,
    )
    return made

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "darks"


@pytest.fixture
def basic_dark(output_dir):
    return BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=2,
        scas=[1, "WFI03"],
        ma_table_number=1010,
    )


@pytest.fixture
def config_dict():

    return {
        "defaults": {
            "gain": 2.0,
            "seed": 123,
        },
        "sca_overrides": {
            3: {
                "gain": 5.0,
            }
        },
    }


# ============================================================
# SCA parsing
# ============================================================

def test_parse_all_wfi(output_dir):

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas="ALL_WFI",
        ma_table_number=1010,
    )

    assert sim.scas == list(range(1, 19))


def test_parse_mixed_scas(basic_dark):

    assert basic_dark.scas == [1, 3]


@pytest.mark.parametrize(
    "bad_scas",
    [
        "WFI01",
        [0],
        [19],
        ["BAD"],
        [None],
    ],
)
def test_invalid_scas_raise(output_dir, bad_scas):

    with pytest.raises(ValueError):
        BaseDarkSimulation(
            output_dir=output_dir,
            program="00999",
            start_time="2026-01-01T00:00:00",
            num_exposures=1,
            scas=bad_scas,
            ma_table_number=1010,
        )


# ============================================================
# MA table config
# ============================================================

def test_get_ma_table_config_weekly(basic_dark):

    cfg = basic_dark._get_ma_table_config()

    assert cfg["read_pattern"] == FAKE_READ_PATTERNS[1010]
    assert cfg["n_resultants"] == 8
    assert cfg["n_reads"] == FAKE_READ_PATTERNS[1010][-1][-1]

def test_get_ma_table_config_truncated(output_dir):
    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[1],
        ma_table_number=9003,
        truncate=15,
    )
    cfg = sim._get_ma_table_config()
    assert cfg["n_resultants"] == 15
    assert cfg["n_reads"] == 15

def test_long_dark_uses_diagnostic_table(output_dir):
    sim = LongDarkSimulation(output_dir=output_dir, scas=[1])
    assert sim.ma_table_number == 9003
    assert sim.truncate == 15
    assert sim._get_ma_table_config()["n_reads"] == 15

def test_run_romanisim_truncate_flag(monkeypatch, output_dir):
    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr("subprocess.run", mock_run)
    sim = LongDarkSimulation(output_dir=output_dir, scas=[1])
    sim._run_romanisim(Path("t.asdf"), sim.start_time, sca=1)
    cmd = mock_run.call_args[0][0]
    assert cmd[cmd.index("--truncate") + 1] == "15"
    assert cmd[cmd.index("--ma_table_number") + 1] == "9003"

def test_post_process_passes_read_pattern(monkeypatch, tmp_path, basic_dark):
    captured = {}
    def fake_simulate(n_reads, read_pattern=None, **kwargs):
        captured["n_reads"] = n_reads
        captured["read_pattern"] = read_pattern
        return np.ones((len(read_pattern), 4, 4)), None
    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        fake_simulate,
    )
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value.tree = {"roman": {"meta": {"instrument": {}}}}
    monkeypatch.setattr("asdf.open", lambda *a, **k: mock_ctx)
    basic_dark._post_process(filename=tmp_path / "fake.asdf", sca=1)
    assert captured["read_pattern"] == FAKE_READ_PATTERNS[1010]
    assert captured["n_reads"] == captured["read_pattern"][-1][-1]

def test_invalid_ma_table(output_dir):
    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[1],
        ma_table_number=9999,
    )

    with pytest.raises(ValueError):
        sim._get_ma_table_config()


# ============================================================
# Config loading
# ============================================================

def test_load_config_and_overrides(output_dir, config_dict, monkeypatch):

    monkeypatch.setattr(
        BaseDarkSimulation,
        "_load_config",
        lambda self, _: config_dict,
    )

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[3],
        ma_table_number=1010,
        config_file="dummy.yml",
    )

    params = sim._get_sca_dark_params(3)

    assert params["gain"] == 5.0


def test_missing_config_raises(output_dir):

    with pytest.raises(FileNotFoundError):
        BaseDarkSimulation(
            output_dir=output_dir,
            program="00999",
            start_time="2026-01-01T00:00:00",
            num_exposures=1,
            scas=[1],
            ma_table_number=1010,
            config_file="does_not_exist.yml",
        )


# ============================================================
# Filename generation
# ============================================================

def test_make_filename_uncal(basic_dark):

    filename = basic_dark._make_filename(exp=5, sca=3)

    expected = (
        "r0099901001001001001_0005_wfi03_f213_uncal.asdf"
    )

    assert filename.name == expected


def test_make_filename_cal(output_dir):

    sim = BaseDarkSimulation(
        output_dir=output_dir,
        program="00999",
        start_time="2026-01-01T00:00:00",
        num_exposures=1,
        scas=[1],
        ma_table_number=1010,
        level=2,
    )

    filename = sim._make_filename(exp=1, sca=1)

    assert filename.name.endswith("_cal.asdf")


# ============================================================
# Romanisim command construction
# ============================================================

def test_run_romanisim_calls_subprocess(monkeypatch, basic_dark):

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    basic_dark._run_romanisim(
        filename=Path("test.asdf"),
        current_time=basic_dark.start_time,
        sca=1,
    )

    assert mock_run.called

    cmd = mock_run.call_args[0][0]

    assert "romanisim-make-image" in cmd
    assert "--sca" in cmd
    assert "1" in cmd


def test_run_romanisim_failure(monkeypatch, basic_dark):

    mock_run = MagicMock()
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "failure"

    monkeypatch.setattr(
        "subprocess.run",
        mock_run,
    )

    with pytest.raises(RuntimeError):
        basic_dark._run_romanisim(
            filename=Path("bad.asdf"),
            current_time=basic_dark.start_time,
            sca=1,
        )


# ============================================================
# run()
# ============================================================

def test_run_calls_internal_methods(monkeypatch, basic_dark):

    run_mock = MagicMock()
    post_mock = MagicMock()

    monkeypatch.setattr(
        basic_dark,
        "_run_romanisim",
        run_mock,
    )

    monkeypatch.setattr(
        basic_dark,
        "_post_process",
        post_mock,
    )

    basic_dark.run()

    expected_calls = (
        basic_dark.num_exposures * len(basic_dark.scas)
    )

    assert run_mock.call_count == expected_calls
    assert post_mock.call_count == expected_calls


# ============================================================
# WeeklyDarkSimulation subclass
# ============================================================

def test_weekly_dark_defaults(output_dir):

    sim = WeeklyDarkSimulation(
        output_dir=output_dir,
        num_exposures=5,
        scas=[1],
    )

    assert sim.program == "00901"
    assert sim.ma_table_number == 1010
    assert sim.num_exposures == 5


# ============================================================
# _post_process smoke test
# ============================================================

def test_post_process_smoke(monkeypatch, tmp_path, basic_dark):

    fake_cube = np.ones((8, 4, 4), dtype=np.uint16)

    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        lambda n_reads, **kwargs: (fake_cube, None),
    )

    mock_asdf = MagicMock()

    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_asdf

    monkeypatch.setattr(
        "asdf.open",
        lambda *args, **kwargs: mock_ctx,
    )

    mock_asdf.tree = {
        "roman": {
            "meta": {
                "instrument": {}
            }
        }
    }

    basic_dark._post_process(
        filename=tmp_path / "fake.asdf",
        sca=1,
    )

    assert (
        mock_asdf.tree["roman"]["meta"]["instrument"]["optical_element"]
        == "DARK"
    )

    assert "data" in mock_asdf.tree["roman"]


# ============================================================
# Rate image reuse, seeding, and integer conversion
# ============================================================
def _patch_asdf(monkeypatch, store):
    ctx = MagicMock()
    ctx.__enter__.return_value.tree = store
    monkeypatch.setattr("asdf.open", lambda *a, **k: ctx)

def test_post_process_reuses_rate_image_per_sca(monkeypatch, tmp_path, basic_dark, fake_rate_image):
    seen = []
    def fake_simulate(n_reads, read_pattern=None, rate_image=None, **kwargs):
        seen.append(rate_image)
        return np.ones((len(read_pattern), 4, 4)), rate_image
    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        fake_simulate,
    )
    _patch_asdf(monkeypatch, {"roman": {"meta": {"instrument": {}}}})
    basic_dark._post_process(tmp_path / "a.asdf", sca=1)
    basic_dark._post_process(tmp_path / "b.asdf", sca=1)
    basic_dark._post_process(tmp_path / "c.asdf", sca=3)
    assert len(fake_rate_image) == 2               # one rate image per SCA, not per exposure
    assert seen[0] is seen[1]                      # both SCA 1 exposures use the same one
    assert seen[0][0, 0] == 1001                   # default seed 1000 + SCA 1
    assert seen[2][0, 0] == 1003                   # SCA 3 has its own

def test_rate_image_same_across_simulations(output_dir):
    weekly = WeeklyDarkSimulation(output_dir=output_dir, scas=[7])
    long = LongDarkSimulation(output_dir=output_dir, scas=[7])
    assert weekly._get_rate_image_seed(7) == long._get_rate_image_seed(7)
    np.testing.assert_array_equal(weekly._get_rate_image(7, {}), long._get_rate_image(7, {}))

def test_rate_image_seed_from_config(monkeypatch, output_dir):
    config = {
        "defaults": {"rate_image_seed": 500},
        "sca_overrides": {2: {"rate_image_seed": 77}},
    }
    monkeypatch.setattr(BaseDarkSimulation, "_load_config", lambda self, _: config)
    sim = BaseDarkSimulation(
        output_dir=output_dir, program="00999", start_time="2026-01-01T00:00:00",
        num_exposures=1, scas=[1, 2], ma_table_number=1010, config_file="dummy.yml",
    )
    assert sim._get_rate_image_seed(1) == 501
    assert sim._get_rate_image_seed(2) == 77
    assert "rate_image_seed" not in sim._get_sca_dark_params(2)

def test_rate_image_uses_detector_counts(monkeypatch, output_dir, fake_rate_image):
    config = {
        "defaults": {"num_hot_pix": 10, "noise_std": 2.0},
        "sca_overrides": {4: {"num_hot_pix": 99, "num_dead_pix": 5}},
    }
    monkeypatch.setattr(BaseDarkSimulation, "_load_config", lambda self, _: config)
    sim = BaseDarkSimulation(
        output_dir=output_dir, program="00999", start_time="2026-01-01T00:00:00",
        num_exposures=1, scas=[4], ma_table_number=1010, config_file="dummy.yml",
    )
    sim._get_rate_image(4, sim._get_sca_dark_params(4))
    assert fake_rate_image[-1]["num_hot_pix"] == 99
    assert fake_rate_image[-1]["num_dead_pix"] == 5
    assert "noise_std" not in fake_rate_image[-1]   # noise settings are not rate image settings

def test_post_process_clips_and_rounds_to_uint16(monkeypatch, tmp_path, basic_dark):
    cube = np.array([-3.5, 0.4, 0.6, 70000.0], dtype=np.float32).reshape(4, 1, 1)
    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        lambda n_reads, **kwargs: (cube, None),
    )
    store = {"roman": {"meta": {"instrument": {}}}}
    _patch_asdf(monkeypatch, store)
    basic_dark._post_process(tmp_path / "a.asdf", sca=1)
    data = store["roman"]["data"]
    assert data.dtype == np.uint16
    assert data.ravel().tolist() == [0, 0, 1, 65535]

def test_seed_applied_once_per_sca(monkeypatch, output_dir, config_dict):
    monkeypatch.setattr(BaseDarkSimulation, "_load_config", lambda self, _: config_dict)
    sim = BaseDarkSimulation(
        output_dir=output_dir, program="00999", start_time="2026-01-01T00:00:00",
        num_exposures=1, scas=[1], ma_table_number=1010, config_file="dummy.yml",
    )
    seeds = []
    monkeypatch.setattr(np.random, "seed", lambda s: seeds.append(s))
    monkeypatch.setattr(
        "wfi_reference_pipeline.utilities.simulate_cal_plan_files.simulate_dark_cal_plan.simulate_dark_reads",
        lambda n_reads, rate_image=None, **kwargs: (np.ones((8, 2, 2)), rate_image),
    )
    _patch_asdf(monkeypatch, {"roman": {"meta": {"instrument": {}}}})
    sim._post_process(output_dir / "a.asdf", sca=1)
    sim._post_process(output_dir / "b.asdf", sca=1)
    assert seeds == [123]
    assert "seed" not in sim._get_sca_dark_params(1)

def _load_real_dark_config():
    import yaml
    config_path = Path(__file__).resolve().parents[1] / "simulate_darks_config.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # A detector listed with nothing after it (e.g. "1:") loads as None and uses the defaults.
    overrides = {sca: (entry or {}) for sca, entry in (config.get("sca_overrides") or {}).items()}
    return config, overrides


def test_real_config_parses_numbers():
    config, overrides = _load_real_dark_config()

    for section in [config["defaults"], *overrides.values()]:
        for key, value in section.items():
            assert isinstance(value, (int, float)), f"{key}={value!r} is not a number"


def test_real_config_detector_populations(monkeypatch, output_dir):
    config, overrides = _load_real_dark_config()
    defaults = config["defaults"]
    def value(sca, key):
        return overrides.get(sca, {}).get(key, defaults[key])
    # Every detector gets its own rate image seed, so hot/warm/dead pixel
    # locations differ between detectors even when their counts are the defaults.
    monkeypatch.setattr(BaseDarkSimulation, "_load_config", lambda self, _: config)
    sim = BaseDarkSimulation(
        output_dir=output_dir, program="00999", start_time="2026-01-01T00:00:00",
        num_exposures=1, scas="ALL_WFI", ma_table_number=1010, config_file="dummy.yml",
    )
    seeds = [sim._get_rate_image_seed(sca) for sca in range(1, 19)]
    assert len(set(seeds)) == 18

    dead = np.array([float(value(sca, "num_dead_pix")) for sca in range(1, 19)])
    assert (dead > 10 * np.median(dead)).sum() == 1

    # Counts are exact (no random spread) so they stay fixed per detector.
    for k in ("num_hot_pix_var", "num_warm_pix_var", "num_dead_pix_var"):
        assert defaults[k] == 0