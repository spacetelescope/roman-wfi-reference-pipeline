import glob
import time
from wfi_reference_pipeline.utilities.submit_files_to_crds import WFISubmit, SubmissionForm
from wfi_reference_pipeline.utilities.config_handler import get_crds_submission_config

update_dict = False

# Load configuration settings
config = get_crds_submission_config()

# Load submission form details
submission_info = config['submission_form']

# Gather new files based on the pattern in the config file
new_files = glob.glob(config['files_to_submit']['crds_ready_dir'] + '/MASK/GSFC/*.asdf')
new_files.sort()

# Initialize the WFISubmit instance
# The server selection is set to 'ops' for testing. All Roman CRDS submissions should go to 'test'.
# All TVAC CRDS submissions should go to 'tvac'
submission = WFISubmit(new_files, submission_info, server='ops')

# User has to modify the following entries in the submission dictionary.
# Rick Cosentino example
if update_dict:
    submission.submission_dict['deliverer'] = 'Richard Cosentino'
    submission.submission_dict['other_email'] = 'rcosenti@stsci.edu'
    submission.submission_dict['file_type'] = 'MASk'
    submission.submission_dict['useafter_matches'] = 'N/A'
    submission.submission_dict['compliance_verified'] = 'N/A'
    submission.submission_dict['calpipe_version'] = 'N/A'
    submission.submission_dict['replacing_badfiles'] = 'No'
    submission.submission_dict['jira_issue'] = 'RTB-000'
    submission.submission_dict['table_rows_changed'] = 'N/A'
    submission.submission_dict['modes_affected'] = 'Both WFI Imaging - WIM and Spectral WSM modes.'
    submission.submission_dict['change_level'] = 'MODERATE'  # allowed entries - TRIVIAL, SEVERE, MODERATE
    submission.submission_dict['correctness_testing'] = 'Files were made with latest versions of roman data models and' \
                                                        'roman attribute dictionary.'
    submission.submission_dict['additional_considerations'] = 'New Additional Considerations'
    submission.submission_dict['description'] = 'New MASK reference files identifying GSFC derived optical bad pixels ' \
                                                'from TVAC data.'

# Update the submission form and submit to CRDS
submission.update_crds_submission_form()

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