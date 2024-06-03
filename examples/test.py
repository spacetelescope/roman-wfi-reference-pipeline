import os
import numpy as np
from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
from wfi_reference_pipeline.utilities.simulate_reads import simulate_dark_reads
from wfi_reference_pipeline.utilities.quality_control.dark_quality_control import DarkQualityControl

from wfi_reference_pipeline.reference_types.dark.dark import Dark
from wfi_reference_pipeline.constants import WFI_MODE_WIM, WFI_MODE_WSM, WFI_FRAME_TIME, \
    WFI_TYPE_IMAGE, WFI_TYPE_GRISM, WFI_TYPE_PRISM, WFI_P_EXPTYPE_IMAGE, WFI_P_EXPTYPE_GRISM, WFI_P_EXPTYPE_PRISM

dates = ['2020-01-01T00:00:00.000', '2021-08-01T11:11:11.111']
dates_strs = ['D1', 'D2']
date_str = dates_strs[0]

write_path = '/grp/roman/RFP/DEV/build_files/Build_24Q3_B14/'

# making CRDS dark WIM
make_WIM_darks = 1
if make_WIM_darks == 1:
    for d in range(0, 1):
        for det in range(1, 2):
            tmp = MakeDevMeta(ref_type='DARK')
            print(tmp.meta_dark)
            if tmp.meta_dark.use_after == '2020-01-01T00:00:00.000':
                date_str = 'D1'
            if tmp.meta_dark.use_after == '2020-01-01T00:00:00.000':
                date_str = 'D2'

            tmp.meta_dark.ma_table_number = 110
            tmp.meta_dark.ma_table_name = '16_RESULTANT_TEST'
            tmp.meta_dark.nframes = 3
            tmp.meta_dark.ngroups = 16

            #sim_read_cube, _ = simulate_dark_reads(16)
            # verifying new averaging methods lgoic flow etc.
            sim_read_cube = np.ones((48, 10, 10))
            for i in range(48):
               sim_read_cube[i] *= (i + 1)
            fl_name = 'roman_dark_' + tmp.meta_dark.instrument_detector + '_' + \
                      tmp.meta_dark.type + '_' + date_str + '.asdf'
            outfile = write_path + fl_name
            RFPdark = Dark(meta_data=tmp.meta_dark,
                           file_list=None,
                           data_array=sim_read_cube,
                           outfile=outfile,
                           clobber=True)
            print(RFPdark.meta_data)

            dark_qc = DarkQualityControl(RFPdark)
            dark_qc.do_checks()



            # num_resultants = RFPdark.meta_data.ngroups
            # num_rds_per_res = RFPdark.meta_data.nframes
            # RFPdark.make_ma_table_resampled_dark(num_resultants=num_resultants, num_rds_per_res=num_rds_per_res)
            # RFPdark.make_dark_rate_image()
            #RFPdark.update_dq_mask()
            # RFPdark.generate_outfile()
            #print('made file', outfile)
            # Set file permissions to read+write for owner, group and global
            #os.chmod(outfile, 0o666)