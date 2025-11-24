import numpy as np
import pytest

from wfi_reference_pipeline.reference_types.pedestal.pedestal import Pedestal 
from wfi_reference_pipeline.resources.make_test_meta import MakeTestMeta

from wfi_reference_pipeline.constants import REF_TYPE_PEDESTAL


@pytest.fixture
def valid_pedestal_data():
    """Fixture for generating valid WFIMetaPedestal metadata."""
    test_meta = MakeTestMeta(ref_type=REF_TYPE_PEDESTAL)
    return test_meta.meta_pedestal

@pytest.fixture 
def pedestal_object(valid_pedestal_data):
    return Pedestal(valid_pedestal_data)

class TestPedestal:

    def test_pedestal_creation(self, pedestal_object):
        
        assert isinstance(pedestal_object, Pedestal)

