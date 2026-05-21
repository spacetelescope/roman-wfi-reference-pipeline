"""Tests for the ReadNoise reference type (reference_types/readnoise/readnoise.py).

Covers the math/data layer. The pipeline plumbing layer is covered in
src/wfi_reference_pipeline/pipelines/tests/test_readnoise_pipeline.py.

ReadNoise has three construction modes - each test picks the cheapest one
that exercises the behavior under test:
    1. 2D array     -> object is "done"; readnoise_image is the input array.
    2. 3D data cube -> data_cube populated; make_readnoise_image() finishes.
    3. file_list    -> reads files from disk AND builds the image.

TEST_DETECTOR_PIXEL_COUNT (32) instead of the real 4096 keeps tests fast.
The math is identical at any size.
"""

import asdf
import numpy as np
import pytest

from wfi_reference_pipeline.constants import (
    DETECTOR_PIXEL_X_COUNT,
    DETECTOR_PIXEL_Y_COUNT,
    REF_TYPE_DARK,
    REF_TYPE_READNOISE,
)
from wfi_reference_pipeline.reference_types.readnoise.readnoise import ReadNoise
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads

TEST_DETECTOR_PIXEL_COUNT = 32


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def meta_data():
    """Valid WFIMetaReadNoise - just needs to satisfy the isinstance check in __init__."""
    return MakeTestMeta(ref_type=REF_TYPE_READNOISE).meta_readnoise


@pytest.fixture(scope="session")
def make_cube():
    """Factory: make_cube(n) returns a simulated n-read dark cube."""
    def _make(num_reads):
        cube, _ = simulate_dark_reads(num_reads, ni=TEST_DETECTOR_PIXEL_COUNT)
        return cube
    return _make


@pytest.fixture
def image_2d():
    """A 2D random array shaped like a final read-noise image (full detector size)."""
    return np.random.random((DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT))


@pytest.fixture(scope="module")
def asdf_filelist(tmp_path_factory, make_cube):
    """Three on-disk asdf files with 1, 2, 3 reads - for exercising file selection."""
    data_path = tmp_path_factory.mktemp("data")
    files = []
    for n in range(1, 4):
        path = data_path / f"data_num_{n}.asdf"
        asdf.AsdfFile({"roman": {"data": make_cube(n)}}).write_to(path)
        files.append(str(path))
    return files


@pytest.fixture
def rn_with_image(meta_data, image_2d):
    """ReadNoise built from a 2D image - object is already 'done'."""
    return ReadNoise(meta_data=meta_data, ref_type_data=image_2d)


@pytest.fixture
def rn_with_cube(meta_data, make_cube):
    """ReadNoise built from a 3D cube - readnoise_image is still None."""
    return ReadNoise(meta_data=meta_data, ref_type_data=make_cube(3))


@pytest.fixture
def rn_with_files(meta_data, asdf_filelist):
    """ReadNoise built from a file list - closest to the real pipeline flow."""
    return ReadNoise(meta_data=meta_data, file_list=asdf_filelist)


# ---- __init__ ---------------------------------------------------------------

def test_init_with_2d_array_sets_readnoise_image(rn_with_image):
    # 2D input IS the final image - shape must pass through unchanged.
    assert rn_with_image.readnoise_image.shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)


def test_init_with_3d_cube_does_not_build_image_yet(rn_with_cube):
    # Staged construction: user can inspect data_cube before kicking off the slow ramp fit.
    assert rn_with_cube.data_cube is not None
    assert rn_with_cube.readnoise_image is None


def test_init_with_wrong_meta_type_raises(image_2d):
    # Handing ReadNoise Dark metadata would silently deliver wrong CRDS metadata.
    dark_meta = MakeTestMeta(ref_type=REF_TYPE_DARK).meta_dark
    with pytest.raises(TypeError):
        ReadNoise(meta_data=dark_meta, ref_type_data=image_2d)


def test_init_with_non_array_data_raises(meta_data):
    # ref_type_data must be ndarray or Quantity. A string should fail loud.
    with pytest.raises(TypeError):
        ReadNoise(meta_data=meta_data, ref_type_data="not_an_array")


def test_init_with_file_list_records_num_files(meta_data, mocker):
    # num_files must reflect len(file_list); patching asdf.open avoids real disk I/O.
    mock_open = mocker.patch("asdf.open")
    mock_open.return_value.__enter__.return_value.tree = {
        "roman": {"data": np.zeros((3, 10, 10))}
    }

    rn = ReadNoise(meta_data=meta_data, file_list=["a.asdf", "b.asdf"])

    assert rn.num_files == 2


def test_default_outfile_name(rn_with_image):
    # 'roman_readnoise.asdf' is public API - any user script could depend on it.
    assert rn_with_image.outfile == "roman_readnoise.asdf"


# ---- noise estimators -------------------------------------------------------

def test_swap_in_cds_noise_as_estimator(rn_with_cube, mocker):
    # comp_cds_noise is a diagnostic alternative to ramp-residual variance.
    # This test pins the swap-in pattern without exercising the CDS math.
    rn_with_cube.comp_cds_noise = mocker.MagicMock(return_value="sentinel")

    rn_with_cube.readnoise_image = rn_with_cube.comp_cds_noise()

    rn_with_cube.comp_cds_noise.assert_called_once()
    assert rn_with_cube.readnoise_image == "sentinel"


def test_comp_ramp_res_var_returns_2d_variance_image(rn_with_cube, make_cube, mocker):
    # Isolate comp_ramp_res_var from fit_cube / make_ramp_model bugs by mocking data_cube.
    fake_cube = mocker.Mock(
        num_i_pixels=TEST_DETECTOR_PIXEL_COUNT,
        num_j_pixels=TEST_DETECTOR_PIXEL_COUNT,
        ramp_model=make_cube(3),
        data=make_cube(3),
    )
    mocker.patch.object(rn_with_cube, "data_cube", fake_cube)

    result = rn_with_cube.comp_ramp_res_var()

    assert result.shape == (TEST_DETECTOR_PIXEL_COUNT, TEST_DETECTOR_PIXEL_COUNT)


def test_comp_cds_noise_returns_2d_image(rn_with_cube, make_cube, mocker):
    # Companion to test_comp_ramp_res_var_returns_2d_variance_image, for the CDS estimator.
    # (This test was previously named `est_...` and silently never ran.)
    fake_cube = mocker.Mock(
        num_i_pixels=TEST_DETECTOR_PIXEL_COUNT,
        num_j_pixels=TEST_DETECTOR_PIXEL_COUNT,
        ramp_model=make_cube(3),
        data=make_cube(3),
        num_reads=3,
    )
    mocker.patch.object(rn_with_cube, "data_cube", fake_cube)

    result = rn_with_cube.comp_cds_noise()

    assert result.shape == (TEST_DETECTOR_PIXEL_COUNT, TEST_DETECTOR_PIXEL_COUNT)


# ---- end-to-end and cube-fitting --------------------------------------------

def test_make_readnoise_image_end_to_end(rn_with_files):
    # Full flow from file list: select cube -> fit -> ramp model -> residual variance.
    # Asserts shape only; component math is tested in dedicated tests above.
    rn_with_files.make_readnoise_image()

    assert rn_with_files.readnoise_image is not None
    assert rn_with_files.readnoise_image.shape == (TEST_DETECTOR_PIXEL_COUNT, TEST_DETECTOR_PIXEL_COUNT)


def test_select_data_cube_picks_file_with_most_reads(rn_with_files):
    # Fixture has files with 1, 2, 3 reads - selector must pick the 3-read one.
    with pytest.raises(AttributeError):
        _ = rn_with_files.data_cube  # not built until selection runs

    rn_with_files._select_data_cube_from_file_list()

    assert rn_with_files.data_cube.num_reads == 3


def test_make_rate_image_produces_2d_slope_and_intercept(rn_with_cube):
    # Both images must match a single detector slice. Wrong shape here breaks everything downstream.
    rn_with_cube.make_rate_image_from_data_cube()

    assert rn_with_cube.data_cube.rate_image.shape == (TEST_DETECTOR_PIXEL_COUNT, TEST_DETECTOR_PIXEL_COUNT)
    assert rn_with_cube.data_cube.intercept_image.shape == (TEST_DETECTOR_PIXEL_COUNT, TEST_DETECTOR_PIXEL_COUNT)


# ---- datamodel output -------------------------------------------------------

def test_populate_datamodel_tree_meets_schema(rn_with_image):
    # CRDS rejects the file if 'meta'/'data' are missing or 'data' isn't float32.
    tree = rn_with_image.populate_datamodel_tree()

    assert "meta" in tree
    assert "data" in tree
    assert tree["data"].shape == (DETECTOR_PIXEL_X_COUNT, DETECTOR_PIXEL_Y_COUNT)
    assert tree["data"].dtype == np.float32


# ============================================================================
# MISSING TESTS (priority order)
# ============================================================================
# 1. __init__ ValueError branch for 1D/4D arrays (right type, wrong dimensionality).
# 2. __init__ with an astropy Quantity input - the .value-stripping path is uncovered.
# 3. _select_data_cube ordering: build files with reads [2,5,1,4] and assert 5 wins.
#    Current test only proves "max == last in fixture", not real sort correctness.
# 4. ReadNoiseDataCube.fit_cube with degree=2 (the code has a TODO that flags this).
# 5. ReadNoiseDataCube.make_ramp_model with order=2 (works) and order=3 (should raise).
# 6. ReadNoise.calculate_error and update_data_quality_array are pass-only no-ops -
#    one-liner tests would pin the no-op contract.
# 7. clobber/outfile behavior: clobber=False must raise if outfile exists; clobber=True must overwrite.
# 8. conftest.py only cleans up "WFI01_superdark.asdf". If any readnoise test writes
#    "roman_readnoise.asdf" to the cwd it will persist - either extend conftest or
#    keep all readnoise output under tmp_path.
