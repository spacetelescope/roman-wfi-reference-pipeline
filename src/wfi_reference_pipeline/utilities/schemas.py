# Define the schema for config.json
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {  # List all the possible entries and their types
        "logging": {
            "type": "object",
            "properties": {
                "log_dir": {"type": "string"},
                "log_level": {"type": "string"},
                "log_tag": {"type": "string"},
            },
            "required": ["log_dir", "log_level"],
        },
        "data_files": {
            "type": "object",
            "properties": {
                "ingest_dir": {"type": "string"},
                "prep_dir": {"type": "string"},
                "crds_ready_dir": {"type": "string"},
            },
            "required": ["ingest_dir", "prep_dir", "crds_ready_dir"],
        },
    },
    # List which entries are needed (all of them)
    "required": ["Logging", "DataFiles"],
}

# Define the schema for quality_control_config.json
# NOTE: All elements added to any ref_type control schema MUST be required
QC_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {

        # Dark Control Settings
        "dark_control": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {
                        "check_mean_dark_rate": {"type": "boolean"},
                        "check_med_dark_rate": {"type": "boolean"},
                        "check_std_dark_rate": {"type": "boolean"},
                        "check_num_hot_pix": {"type": "boolean"},
                        "check_num_dead_pix": {"type": "boolean"},
                        "check_num_unreliable_pix": {"type": "boolean"},
                        "check_num_warm_pix": {"type": "boolean"},
                    },
                    "required": [
                        "check_mean_dark_rate",
                        "check_med_dark_rate",
                        "check_std_dark_rate",
                        "check_num_hot_pix",
                        "check_num_dead_pix",
                        "check_num_unreliable_pix",
                        "check_num_warm_pix",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "max_mean_dark_rate_reference_value": {"type": "number"},
                        "max_med_dark_rate_reference_value": {"type": "number"},
                        "max_std_dark_rate_reference_value": {"type": "number"},
                        "max_num_hot_pix_reference_value": {"type": "number"},
                        "max_num_dead_pix_reference_value": {"type": "number"},
                        "max_num_unreliable_pix_reference_value": {"type": "number"},
                        "max_num_warm_pix_reference_value": {"type": "number"},
                    },
                    "required": [
                        "max_mean_dark_rate_reference_value",
                        "max_med_dark_rate_reference_value",
                        "max_std_dark_rate_reference_value",
                        "max_num_hot_pix_reference_value",
                        "max_num_dead_pix_reference_value",
                        "max_num_unreliable_pix_reference_value",
                        "max_num_warm_pix_reference_value",
                    ],
                },
            },
        },

        # Readnoise Settings
        "readnoise_control": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {
                        "check_mean_readnoise": {"type": "boolean"},
                        "check_med_readnoise": {"type": "boolean"},
                        "check_std_readnoise": {"type": "boolean"},
                    },
                    "required": [
                        "check_mean_readnoise",
                        "check_med_readnoise",
                        "check_std_readnoise",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "max_mean_dark_rate_reference_value": {"type": "number"},
                        "max_med_dark_rate_reference_value": {"type": "number"},
                        "max_std_dark_rate_reference_value": {"type": "number"},
                    },
                    "required": [
                        "max_mean_dark_rate_reference_value",
                        "max_med_dark_rate_reference_value",
                        "max_std_dark_rate_reference_value",
                    ],
                },
            },
        },
    },
}