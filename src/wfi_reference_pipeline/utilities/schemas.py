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
    "required": ["logging", "data_files"],
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
    "required": [
        "dark_control",
        "readnoise_control",
    ],
}

# Define the schema for crds_submission_config.json
CRDS_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "crds_files": {
            "type": "object",
            "properties": {
                "crds_ready_dir": {"type": "string"},
            },
            "required": ["crds_ready_dir"],
        },
        "submission_form": {
            "type": "object",
            "properties": {
                "deliverer": {"type": "string"},
                "other_email": {"type": "string"},
                "file_type": {"type": "string"},
                "history_updated": {"type": "boolean"},
                "pedigree_updated": {"type": "boolean"},
                "keywords_checked": {"type": "boolean"},
                "descrip_updated": {"type": "boolean"},
                "useafter_updated": {"type": "boolean"},
                "useafter_matches": {"type": "string"},
                "compliance_verified": {"type": "string"},
                "etc_delivery": {"type": "boolean"},
                "calpipe_version": {"type": "string"},
                "replacement_files": {"type": "boolean"},
                "old_reference_files": {"type": "string"},
                "replacing_badfiles": {"type": "string"},
                "jira_issue": {"type": "string"},
                "table_rows_changed": {"type": "string"},
                "reprocess_affected": {"type": "boolean"},
                "modes_affected": {"type": "string"},
                "change_level": {"type": "string"},
                "correctness_testing": {"type": "string"},
                "additional_considerations": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": [
                "deliverer",
                "other_email",
                "file_type",
                "history_updated",
                "pedigree_updated",
                "keywords_checked",
                "descrip_updated",
                "useafter_updated",
                "useafter_matches",
                "compliance_verified",
                "etc_delivery",
                "calpipe_version",
                "replacement_files",
                "old_reference_files",
                "replacing_badfiles",
                "jira_issue",
                "table_rows_changed",
                "reprocess_affected",
                "modes_affected",
                "change_level",
                "correctness_testing",
                "additional_considerations",
                "description",
            ],
        },
    },
    "required": ["crds_files", "submission_form"],
}