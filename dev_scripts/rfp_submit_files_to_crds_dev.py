import glob
import time
from wfi_reference_pipeline.utilities.submit_files_to_crds import WFISubmit
from wfi_reference_pipeline.utilities.config_handler import get_crds_submission_config

# Load configuration settings
config = get_crds_submission_config()

# Gather new files based on the pattern in the config file
new_files = glob.glob(config['data_files']['RFP_buildpath'] + config['data_files']['file_pattern'])
new_files.sort()

# Load submission form details
submission_info = config['submission_form']

# Start timing the submission process
start_time = time.time()
print("Start Time:", start_time)

# Initialize the WFISubmit instance
submission = WFISubmit(new_files, submission_info, server='test')

# Update the submission form and submit to CRDS
submission.update_form()
submission.submit_to_crds()

# Measure elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print("End Time:", end_time)
print("Elapsed Time:", elapsed_time)