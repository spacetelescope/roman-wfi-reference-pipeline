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
QC_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {  # List all the possible entries and their types
        "dark_controls": {
            "type": "object",
            "properties": {
                "check_mean_dark_rate_flag": {"type": "boolean"},
                "check_med_dark_rate_flag": {"type": "boolean"},
                "check_std_dark_rate_flag": {"type": "boolean"},
                "check_num_hot_pix_flag": {"type": "boolean"},
                "check_num_dead_pix_flag": {"type": "boolean"},
                "check_num_unreliable_pix_flag": {"type": "boolean"},
                "check_num_warm_pix_flag": {"type": "boolean"},
                "max_mean_dark_rate_reference_value": {"type": "number"},
                "max_med_dark_rate_reference_value": {"type": "number"},
                "max_std_dark_rate_reference_value": {"type": "number"},
                "max_num_hot_pix_reference_value": {"type": "number"},
                "max_num_dead_pix_reference_value": {"type": "number"},
                "max_num_unreliable_pix_reference_value": {"type": "number"},
                "max_num_warm_pix_reference_value": {"type": "number"},
            },
            "required": [
                "check_mean_dark_rate_flag",
                "check_med_dark_rate_flag",
                "check_std_dark_rate_flag",
                "check_num_hot_pix_flag",
                "check_num_dead_pix_flag",
                "check_num_unreliable_pix_flag",
                "check_num_warm_pix_flag",
                "max_mean_dark_rate_reference_value",
                "max_med_dark_rate_reference_value",
                "max_std_dark_rate_reference_value",
                "max_num_hot_pix_reference_value",
                "max_num_dead_pix_reference_value",
                "max_num_unreliable_pix_reference_value",
                "max_num_warm_pix_reference_value",
            ],
        },
        "readnoise_controls": {
            "type": "object",
            "properties": {
                "check_mean_readnoise_flag": {"type": "boolean"},
                "check_med_readnoise_flag": {"type": "boolean"},
                "check_std_readnoise_flag": {"type": "boolean"},
                "max_mean_dark_rate_reference_value": {"type": "number"},
                "max_med_dark_rate_reference_value": {"type": "number"},
                "max_std_dark_rate_reference_value": {"type": "number"},
            },
            "required": [
                "check_mean_readnoise_flag",
                "check_med_readnoise_flag",
                "check_std_readnoise_flag",
                "max_mean_dark_rate_reference_value",
                "max_med_dark_rate_reference_value",
                "max_std_dark_rate_reference_value",
            ],
        },
    },
}