import pytest

from wfi_reference_pipeline.constants import REF_TYPE_DARKDECAYSIGNAL
from wfi_reference_pipeline.reference_types.dark_decay_signal.dark_decay_signal import (
    DARK_DECAY_TABLE,
    DarkDecaySignal,
    get_darkdecay_values,
)
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta


@pytest.fixture
def valid_meta_data():
    """Fixture for generating valid WFIMetaDarkDecaySignal metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_DARKDECAYSIGNAL)
    return test_meta.meta_dark_decay_signal


@pytest.fixture
def dark_decay_signal_object(valid_meta_data):
    """Fixture for initializing a DarkDecaySignal object with valid metadata."""
    return DarkDecaySignal(meta_data=valid_meta_data)


class TestDarkDecaySignal:

    def test_dark_decay_signal_instantiation_with_valid_metadata(
        self, dark_decay_signal_object
    ):
        """
        Test that DarkDecaySignal object is created successfully with valid metadata.
        """
        assert isinstance(dark_decay_signal_object, DarkDecaySignal)

    def test_default_description_is_set_if_empty(self, valid_meta_data):
        """
        Test that a default description is populated if metadata description is empty.
        """
        valid_meta_data.description = ""
        dark_decay = DarkDecaySignal(meta_data=valid_meta_data)

        assert dark_decay.meta_data.description == (
            "Roman WFI Dark Decay Signal reference file for each detector table."
        )

    def test_populate_datamodel_tree(self, dark_decay_signal_object):
        """
        Test that the data model tree is correctly populated.
        """
        data_model_tree = dark_decay_signal_object.populate_datamodel_tree()

        assert "meta" in data_model_tree
        assert "decay_table" in data_model_tree

        decay_table = data_model_tree["decay_table"]
        assert isinstance(decay_table, dict)
        assert len(decay_table) == 18

        for detector, values in decay_table.items():
            assert detector.startswith("WFI")
            assert "amplitude" in values
            assert "time_constant" in values
            assert isinstance(values["amplitude"], (int, float))
            assert isinstance(values["time_constant"], (int, float))

    def test_dark_decay_signal_outfile_default(self, dark_decay_signal_object):
        """
        Test that the default outfile name is correct.
        """
        assert dark_decay_signal_object.outfile == "roman_dark_decay_signal.asdf"


class TestDarkDecaySignalHelpers:

    def test_get_darkdecay_values_valid_detector(self):
        """
        Test helper function returns amplitude and time constant for a valid detector.
        """
        values = get_darkdecay_values("WFI01")

        assert isinstance(values, list)
        assert len(values) == 2
        assert values[0] == DARK_DECAY_TABLE["WFI01"]["amplitude"]
        assert values[1] == DARK_DECAY_TABLE["WFI01"]["time_constant"]

    def test_get_darkdecay_values_invalid_detector(self):
        """
        Test helper function raises KeyError for invalid detector ID.
        """
        with pytest.raises(KeyError):
            get_darkdecay_values("WFI99")
            

@pytest.mark.parametrize("detector_id, values", DARK_DECAY_TABLE.items())
def test_dark_decay_table_schema_and_value_ranges(detector_id, values):
    """
    Ensure each detector entry has exactly one amplitude and one time constant
    and that values fall within valid range. Times less than 3 seconds are
    shorter than the WFI frame time in imaging mode so that is not physical.
    Others were chosen to be consistent but within a reasonable tolerance
    of the data provided in the DARK DECAY TABLE from T. Brandt.
    """
    # Enforce exactly two keys
    assert isinstance(values, dict)
    assert set(values.keys()) == {"amplitude", "time_constant"}

    amplitude = values["amplitude"]
    time_constant = values["time_constant"]

    # Amplitude must be > 0 and < 1
    assert isinstance(amplitude, (int, float))
    assert 0.001 <= amplitude <= 0.99, (
        f"{detector_id}: amplitude {amplitude} outside valid range [0.001, 0.99]"
    )

    # Time constant must be between 3 and 100 seconds
    assert isinstance(time_constant, (int, float))
    assert 3.0 <= time_constant <= 100.0, (
        f"{detector_id}: time_constant {time_constant} outside valid range [3, 100]"
    )