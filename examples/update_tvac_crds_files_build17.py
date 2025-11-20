import os

from wfi_reference_pipeline.utilities.update_reference_files import UpdateReferences


def main():
    """
    Main script to update all reference files that now dont have units
    to the latest data model.
    """
    input_dir = "/PATH/TO/INPUT/TVAC/"
    output_dir = "/PATH/TO/OUTPUT/"

    # Replace default message with this to append to description.
    update_message = " Updated for Build 17 versions of romancal, rad, and rdm."

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all allowed reference types and update each
    updater = UpdateReferences(input_dir, output_dir)
    updater.update_reference_types(update_message)

if __name__ == "__main__":
    main()
