import os
import stat
from types import SimpleNamespace

import asdf
import numpy as np
import pytest

from wfi_reference_pipeline.reference_types.reference_type import ReferenceType

DEFAULT = object()

class StubReferenceType(ReferenceType):

    def calculate_error(self):
        return super().calculate_error() 

    def update_data_quality_array(self):
        return super().update_data_quality_array()
    
    def populate_datamodel_tree(self):
        tree = {
            "metadata" : {
                "a" : "A",
                "b" : "B",
            },
            "date" : "12-12-2025"
            
        }
        return tree 

# NOTE: not using make_test_meta because we want to make invalid metadata, also because we don't want to add another metadata type for Stub (maybe add in test)

@pytest.fixture(scope="session")
def valid_test_metadata():

    metadata = SimpleNamespace(
        reference_type = "stub_ref_type",
        description = "For RFP testing.",
        author = "RFP Test Suite",
        origin = "STSCI",
        instrument = "WFI",
        detector = "WFI01"
    )

    return metadata

@pytest.fixture(scope="session")
def default_perms():
    return 0o666

@pytest.fixture(scope="session")
def readonly_perms():
    return 0o644

@pytest.fixture(scope="session")
def valid_test_filelist():
    file_list = ["testfile1.md", "testfile2.md"]
    return file_list

@pytest.fixture(scope="session")
def valid_test_referencedata():
    data = np.zeros((20, 20), dtype=np.uint32)
    return data

@pytest.fixture(scope="session")
def valid_bitmask():
    bitmask = np.zeros((2,2), dtype=np.uint32)
    return bitmask

@pytest.fixture(scope="session")
def valid_datatree():
    tree = {
            "metadata" : {
                "c" : "ABCD",
                "d" : "3",
            },
            "date" : "10-30-2020"
            
        }
    return tree

# Necessary so that all dependencies have the same tmp_path
@pytest.fixture 
def valid_outfile(tmp_path):
    outfile = tmp_path / "outfile.asdf"
    return outfile


"""
NOTE SYE: Explaining Factory

This fixture is a factory. It returns a function (not a value, like a typical fixture)
that can be passed into a test like a typical fixture. This function can then be called
inside the test. Factories are similar to having a helper function that returns an object.
"""
@pytest.fixture 
def make_test_ref(valid_test_metadata, valid_test_filelist, valid_test_referencedata, valid_outfile):
    
    def _make(metadata=DEFAULT, filelist=DEFAULT, ref_data=DEFAULT, clobber=DEFAULT, bitmask=DEFAULT, masksize=DEFAULT, outfile=DEFAULT):
        return StubReferenceType(
            meta_data = valid_test_metadata if metadata is DEFAULT else metadata, 
            file_list = valid_test_filelist if filelist is DEFAULT else filelist,
            ref_type_data = valid_test_referencedata if ref_data is DEFAULT else ref_data,
            clobber = False if clobber is DEFAULT else clobber,
            outfile = valid_outfile if outfile is DEFAULT else outfile,
            bit_mask = None if bitmask is DEFAULT else bitmask,
            mask_size = (20, 20) if masksize is DEFAULT else masksize,
        )
    return _make


### Initialization Tests ###

def test_successful_creation_defaults_filelist_passes(make_test_ref):

    ref_type = make_test_ref(ref_data=None)

    assert ref_type is not None

def test_successful_creation_defaults_referencedata_passes(make_test_ref):

    ref_type = make_test_ref(filelist=None)

    assert ref_type is not None

def test_file_list_not_list_fails(make_test_ref):
    
    bad_file_list = "testfile1.md"

    with pytest.raises(ValueError):
        _ = make_test_ref(filelist=bad_file_list, ref_data=None)

def test_too_many_inputs_fails(make_test_ref):

    with pytest.raises(ValueError):
        _ = make_test_ref()

def test_no_inputs_fails(make_test_ref):

    with pytest.raises(ValueError):
        _ = make_test_ref(filelist=None, ref_data=None)

def test_valid_external_bitmask_passes(make_test_ref):

    valid_bitmask = np.zeros((2,2), dtype=np.uint32)

    ref_type = make_test_ref(ref_data=None, bitmask=valid_bitmask)

    assert ref_type is not None

def test_bad_bitmask_wrong_type_fails(make_test_ref):

    bad_bitmask = [0]

    with pytest.raises(TypeError):
        _ = make_test_ref(ref_data=None, bitmask=bad_bitmask)

def test_bad_bitmask_wrong_datatype_fails(make_test_ref):

    bad_bitmask = np.zeros((2, 2), dtype=np.int32)

    with pytest.raises(ValueError):
        _ = make_test_ref(ref_data=None, bitmask=bad_bitmask)

def test_bad_bitmask_wrong_data_dimension_fails(make_test_ref):

    bad_bitmask = np.zeros((2, 2, 2), dtype=np.uint32)

    with pytest.raises(ValueError):
        _ = make_test_ref(ref_data=None, bitmask=bad_bitmask)


### Check Outfile Tests ###

def test_check_no_outfile_fails(make_test_ref):

    ref_type = make_test_ref(outfile=None, ref_data=None)
    
    with pytest.raises(ValueError):
        ref_type.check_outfile()

def test_check_outfile_no_clobber_with_file_fails(make_test_ref, valid_outfile):

    valid_outfile.write_text("")

    ref_type = make_test_ref(ref_data=None, clobber=False)

    with pytest.raises(FileExistsError):
        ref_type.check_outfile()

def test_check_outfile_clobber_with_file_passes(make_test_ref, valid_outfile):
    
    valid_outfile.write_text("")

    ref_type = make_test_ref(ref_data=None, clobber=True)

    ref_type.check_outfile()

    assert not valid_outfile.exists()

def test_check_outfile_no_clobber_no_file_passes(make_test_ref, valid_outfile):

    ref_type = make_test_ref(ref_data=None, clobber=False)

    ref_type.check_outfile()

    assert not valid_outfile.exists()

def test_check_outfile_clobber_no_file_passes(make_test_ref, valid_outfile):
    
    ref_type = make_test_ref(ref_data=None, clobber=True)

    ref_type.check_outfile()

    assert not valid_outfile.exists()

### Generate Outfile Tests ###

def test_generate_outfile_no_outfile_fails(make_test_ref):

    ref_type = make_test_ref(outfile=None, ref_data=None)

    with pytest.raises(ValueError):
        ref_type.generate_outfile()

def test_generate_outfile_datamodel_default_perms_passes(make_test_ref, valid_datatree, valid_outfile, default_perms):

    ref_type=make_test_ref(ref_data=None)

    ref_type.generate_outfile(datamodel_tree=valid_datatree)

    assert valid_outfile.exists()
    assert stat.S_IMODE(os.stat(valid_outfile).st_mode) == default_perms
    with asdf.open(valid_outfile) as af:
        assert af.tree["roman"]["metadata"]["c"] == "ABCD"
        assert af.tree["roman"]["metadata"]["d"] == "3"
        assert af.tree["roman"]["date"] == "10-30-2020"

def test_generate_outfile_datamodel_set_perms_passes(make_test_ref, valid_datatree, valid_outfile, readonly_perms):

    ref_type = make_test_ref(ref_data=None)

    ref_type.generate_outfile(datamodel_tree=valid_datatree, file_permission=readonly_perms)

    assert valid_outfile.exists()
    assert stat.S_IMODE(os.stat(valid_outfile).st_mode) == readonly_perms
    with asdf.open(valid_outfile) as af:
        assert af.tree["roman"]["metadata"]["c"] == "ABCD"
        assert af.tree["roman"]["metadata"]["d"] == "3"
        assert af.tree["roman"]["date"] == "10-30-2020"

def test_generate_outfile_generate_default_perms_passes(make_test_ref, valid_outfile, default_perms):

    ref_type = make_test_ref(ref_data=None)

    ref_type.generate_outfile()

    assert valid_outfile.exists()
    assert stat.S_IMODE(os.stat(valid_outfile).st_mode) == default_perms
    with asdf.open(valid_outfile) as af:
        assert af.tree["roman"]["metadata"]["a"] == "A"
        assert af.tree["roman"]["metadata"]["b"] == "B"
        assert af.tree["roman"]["date"] == "12-12-2025"

def test_generate_outfile_generate_set_perms_passes(make_test_ref, valid_outfile, readonly_perms):

    ref_type = make_test_ref(ref_data=None)

    ref_type.generate_outfile(file_permission=readonly_perms)

    assert valid_outfile.exists()
    assert stat.S_IMODE(os.stat(valid_outfile).st_mode) == readonly_perms
    with asdf.open(valid_outfile) as af:
        assert af.tree["roman"]["metadata"]["a"] == "A"
        assert af.tree["roman"]["metadata"]["b"] == "B"
        assert af.tree["roman"]["date"] == "12-12-2025"