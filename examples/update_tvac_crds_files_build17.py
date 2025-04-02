import os
from wfi_reference_pipeline.utilities.update_reference_files import UpdateReferences 


def main():
    """
    Main script to update all allowed reference files for the Roman Space Telescope
    to the latest data model.
    """
    input_dir = "/path/to/old/references/"  # Update this path
    output_dir = "/path/to/updated/references/"  # Update this path

    # Custom update message (Override the default)
    update_message = " Updated for Build 17 versions of romancal, rad, and rdm."

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all allowed reference types and process them
    for ref_type in UpdateReferences.ALLOWED_REF_TYPES:
        print(f"Processing reference type: {ref_type}")

        updater = UpdateReferences(ref_type, input_dir, output_dir)
        updater.process_files(update_message)  # Explicitly passing a message

        print(f"Finished updating {ref_type} reference files.\n")

if __name__ == "__main__":
    main()
