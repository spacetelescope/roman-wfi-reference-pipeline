from astropy.time import Time

from wfi_reference_pipeline.reference_types.distortion.distortion import Distortion
from wfi_reference_pipeline.reference_types.pixel_area.pixel_area import PixelArea
from wfi_reference_pipeline.resources.make_meta import MakeMeta

for i in range(1, 19):
    wfi_id = f'WFI{i:02}'

    tmp = MakeMeta(ref_type="DISTORTION", instrument_detector="WFI01")
    tmp.meta_distortion.author = "Richard G Cosentino"
    tmp.meta_distortion.description = (
        "The Geometric Distortion reference file on Roman "
        "CRDS reflects newest changes to the pysiaf package "
        "corresponding to versions v0.27.0."
    )
    tmp.meta_distortion.useafter = Time(
        "2026-09-01T00:00:00.000",
        format="isot",
    )
    tmp.meta_distortion.pedigree = "GROUND"
    tmp.meta_distortion.instrument_detector = wfi_id
    print(tmp.meta_distortion.export_asdf_meta())

    output_dir = '/grp/roman/RFP/DEV/run_for_record/batch2/'
    outfile = output_dir + 'roman_rfp_rgc_distortion_'+wfi_id + '.asdf'

    rfp_distortion = Distortion(meta_data=tmp.meta_distortion, outfile=outfile, clobber=True)
    rfp_distortion.make_siaf_distortion(rfp_distortion.meta_data.instrument_detector)
    rfp_distortion.generate_outfile()

    print('Made reference file', rfp_distortion.outfile)


for i in range(1, 19):
    wfi_id = f'WFI{i:02}'

    tmp = MakeMeta(ref_type="AREA", instrument_detector="WFI01")
    tmp.meta_pixelarea.author = "Richard G Cosentino"
    tmp.meta_pixelarea.description = (
        "The Pixel Area reference file on Roman "
        "CRDS reflects newest changes to the pysiaf package "
        "corresponding to versions v0.27.0."
    )
    tmp.meta_pixelarea.useafter = Time(
        "2026-09-01T00:00:00.000",
        format="isot",
    )
    tmp.meta_pixelarea.pedigree = "GROUND"
    tmp.meta_pixelarea.instrument_detector = wfi_id
    print(tmp.meta_pixelarea.export_asdf_meta())

    output_dir = '/grp/roman/RFP/DEV/run_for_record/batch2/'
    outfile = output_dir + 'roman_rfp_rgc_pixelarea_' + wfi_id + '.asdf'

    rfp_pixelarea = PixelArea(meta_data=tmp.meta_pixelarea, outfile=outfile, clobber=True)
    rfp_pixelarea.make_pixel_area_from_siaf_file(
        filename="newsiaf_20260727.xml",
        basepath="/grp/roman/RFP/DEV/build_files/Build_26Q4_B23/",
    )
    rfp_pixelarea.generate_outfile()

    print('Made reference file', rfp_pixelarea.outfile)