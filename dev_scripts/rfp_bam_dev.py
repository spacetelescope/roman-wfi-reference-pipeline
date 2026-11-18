from wfi_reference_pipeline.reference_types.boresight_alignment_matrix.boresight_alignment_matrix import (
    BAM_DELIVERIES,
    BoresightAlignmentMatrix,
)
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
 
# Pick one entry from the BAM delivery history:
# "GROUND", "COMMISSIONING", "PFAC1", "CAR-86.15", "CAR-86.17"
bam_entry_key = "CAR-86.17"
bam_entry = BAM_DELIVERIES[bam_entry_key]
 
outfile = f'/PATH/TO/scratch/roman_wfi_bam_{bam_entry_key}.asdf'
 
tmp = MakeDevMeta(ref_type='BAM')
tmp.meta_bam.author = bam_entry["author"]
tmp.meta_bam.description = bam_entry["description"]
# Pedigree with the date range of the data (e.g. INFLIGHT 2026-10-13 2026-10-13).
# Set after construction because WFIMetadata only validates the pedigree keyword on init.
tmp.meta_bam.pedigree = bam_entry["pedigree"]
tmp.meta_bam.use_after = bam_entry["use_after"]
tmp.meta_bam.origin = 'STSCI/SOC'
# Common meta telescope ('ROMAN') and instrument name ('WFI') come from MakeDevMeta.
# The BAM is agnostic of the detector, so instrument.detector is always NOT_CONFIGURED.
 
# Boresight V2, V3 (arcsec) and V3 Y angle (deg).
# TODO: replace these placeholders with the measured values before delivery.
tmp.meta_bam.v2 = 0.0
tmp.meta_bam.v3 = 0.0
tmp.meta_bam.v3_yangle = 0.0
 
rfp_bam = BoresightAlignmentMatrix(meta_data=tmp.meta_bam,
                                   ref_type_data=bam_entry["bam_quaternion"],
                                   outfile=outfile,
                                   clobber=True)
rfp_bam.generate_outfile()
 
# To make a file for every entry, loop over BAM_DELIVERIES.items() and repeat the
# steps above with a new MakeDevMeta object and outfile for each entry.
