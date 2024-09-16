import glob
import time
from wfi_reference_pipeline.utilities.submit_files_to_crds import WFISubmit
from wfi_reference_pipeline.utilities.config_handler import get_crds_submission_config
from wfi_reference_pipeline.utilities.manifest import make_manifest, print_manifest, print_meta_fields_together


# Change to True to complete submission info
update_dict = False

# Load configuration settings
config = get_crds_submission_config()

# Gather new files based on the pattern in the config file
new_files = glob.glob(config['files_to_submit']['crds_ready_dir'] + '/MASK/GSFC/*.asdf')
new_files.sort()

# Initialize the WFISubmit instance
# The server selection is set to 'ops' for testing. All Roman CRDS submissions should go to 'test'.
# All TVAC CRDS submissions should go to 'tvac'
submission = WFISubmit(new_files, server='ops')

# User has to modify the following entries in the submission dictionary.
# Rick Cosentino example
if update_dict:
    submission.submission_dict['instrument'] = 'WFI'
    submission.submission_dict['deliverer'] = 'Richard Cosentino'
    submission.submission_dict['other_email'] = 'rcosenti@stsci.edu'
    submission.submission_dict['file_type'] = 'MASk'
    submission.submission_dict['useafter_matches'] = 'N/A'
    submission.submission_dict['compliance_verified'] = 'N/A'
    submission.submission_dict['calpipe_version'] = 'N/A'
    submission.submission_dict['replacing_badfiles'] = 'No'
    submission.submission_dict['jira_issue'] = 'RTB-000'
    submission.submission_dict['table_rows_changed'] = 'N/A'
    submission.submission_dict['modes_affected'] = 'Both WFI Imaging WIM and Spectral WSM modes.'
    submission.submission_dict['change_level'] = 'MODERATE'  # allowed entries - TRIVIAL, SEVERE, MODERATE
    submission.submission_dict['correctness_testing'] = 'Files were made with latest versions of roman data models and' \
                                                        'roman attribute dictionary.'
    submission.submission_dict['additional_considerations'] = 'New Additional Considerations'
    submission.submission_dict['description'] = 'New MASK reference files identifying GSFC derived optical bad pixels from TVAC data.'

# Update the submission form and submit to CRDS
submission.update_crds_submission_form()

check_files = True
if check_files:
    # Check manifest of meta data before submitting.
    meta_manifest = make_manifest(new_files)

    # This below will print the meta for each file.
    print_manifest(meta_manifest)

    # This will print the meta by each key in the meta for every file.
    print_meta_fields_together(meta_manifest)

    # Certify files with CRDS command.
    submission.certify_files()

# Boolean switch to make sure no one delivers anything on the fly by just running this without looking at it.
submit_on = False
if submit_on:
    # Start timing the submission process
    start_time = time.time()
    print("Start Time:", start_time)
    submission.submit_to_crds()
    # Measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("End Time:", end_time)
    print("Elapsed Time:", elapsed_time)