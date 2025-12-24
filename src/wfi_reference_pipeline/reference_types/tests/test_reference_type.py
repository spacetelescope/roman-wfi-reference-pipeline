from types import SimpleNamespace

import numpy as np
import pytest

from wfi_reference_pipeline.reference_types.reference_type import ReferenceType


class DummyReferenceType(ReferenceType):

    def calculate_error(self):
        return super().calculate_error() 

    def update_data_quality_array(self):
        return super().update_data_quality_array()
    
    def populate_datamodel_tree(self):
        return super().populate_datamodel_tree()


# NOTE: not using make_test_meta because we want to make invalid metadata, also because we don't want to add another metadata type for dummy (maybe add in test)

@pytest.fixture 
def valid_dummy_metadata():

    metadata = SimpleNamespace(
        reference_type = "dummy_ref_type",
        description = "For RFP testing.",
        author = "RFP Test Suite",
        origin = "STSCI",
        instrument = "WFI",
        detector = "WFI01"
    )

    return metadata

@pytest.fixture
def valid_dummy_filelist():
    file_list = ["dummyfile1.md", "dummyfile2.md"]
    return file_list

@pytest.fixture 
def valid_dummy_referencedata():
    data = np.zeros((100, 100), dtype=np.uint32)
    return data

# Necessary so that all dependencies have the same tmp_path
@pytest.fixture 
def valid_outfile(tmp_path):
    outfile = tmp_path / "outfile.asdf"
    return outfile

@pytest.fixture
def valid_dummy_ref(valid_dummy_metadata, valid_dummy_filelist):
    
    ref_type = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist)

    return ref_type

@pytest.fixture
def valid_dummy_ref_with_outfile(valid_dummy_metadata, valid_dummy_filelist, valid_outfile):
    ref_type = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, outfile=valid_outfile)

    return ref_type

@pytest.fixture
def valid_dummy_ref_with_outfile_clobber(valid_dummy_metadata, valid_dummy_filelist, valid_outfile):
    ref_type = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, outfile=valid_outfile, clobber=True)
    return ref_type

## TODO: add fixture for adding files

## TODO: check if this is a reasonable fixture 


### Initialization Tests ###

def test_successful_creation_defaults_filelist(valid_dummy_ref):

    ref_type = valid_dummy_ref

    assert ref_type is not None

def test_successful_creation_defaults_referencedata(valid_dummy_metadata, valid_dummy_referencedata):

    ref_type = DummyReferenceType(meta_data=valid_dummy_metadata, ref_type_data=valid_dummy_referencedata)

    assert ref_type is not None

def test_file_list_not_list(valid_dummy_metadata):
    
    bad_file_list = "dummyfile1.md"

    with pytest.raises(ValueError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=bad_file_list)

def test_too_many_inputs(valid_dummy_filelist, valid_dummy_metadata, valid_dummy_referencedata):

    with pytest.raises(ValueError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, ref_type_data=valid_dummy_referencedata)

def test_no_inputs(valid_dummy_metadata):

    with pytest.raises(ValueError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata)

def test_valid_external_bitmask(valid_dummy_metadata, valid_dummy_filelist):

    valid_bitmask = np.zeros((2,2), dtype=np.uint32)

    ref_type = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, bit_mask=valid_bitmask)

    assert ref_type is not None

def test_bad_bitmask_wrong_type(valid_dummy_metadata, valid_dummy_filelist):

    bad_bitmask = [0]

    with pytest.raises(TypeError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, bit_mask=bad_bitmask)

def test_bad_bitmask_wrong_datatype(valid_dummy_metadata, valid_dummy_filelist):

    bad_bitmask = np.zeros((2, 2), dtype=np.int32)

    with pytest.raises(ValueError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, bit_mask=bad_bitmask)

def test_bad_bitmask_wrong_data_dimension(valid_dummy_metadata, valid_dummy_filelist):

    bad_bitmask = np.zeros((2, 2, 2), dtype=np.uint32)

    with pytest.raises(ValueError):
        _ = DummyReferenceType(meta_data=valid_dummy_metadata, file_list=valid_dummy_filelist, bit_mask=bad_bitmask)


### Check Outfile Tests ###

def test_check_no_outfile(valid_dummy_ref):
    
    with pytest.raises(ValueError):
        valid_dummy_ref.check_outfile()

def test_check_outfile_no_clobber_with_file(valid_dummy_ref_with_outfile, valid_outfile):

    valid_outfile.write_text("")

    with pytest.raises(FileExistsError):
        valid_dummy_ref_with_outfile.check_outfile()

def test_check_outfile_clobber_with_file(valid_dummy_ref_with_outfile_clobber, valid_outfile):
    
    valid_outfile.write_text("")

    valid_dummy_ref_with_outfile_clobber.check_outfile()

    assert not valid_outfile.exists()

def test_check_outfile_no_clobber_no_file(valid_dummy_ref_with_outfile, valid_outfile):

    valid_dummy_ref_with_outfile.check_outfile()

    assert not valid_outfile.exists()

def test_check_outfile_clobber_no_file(valid_dummy_ref_with_outfile_clobber, valid_outfile):
    
    valid_dummy_ref_with_outfile_clobber.check_outfile()

    assert not valid_outfile.exists()

### Generate Outfile Tests ###

# TODO