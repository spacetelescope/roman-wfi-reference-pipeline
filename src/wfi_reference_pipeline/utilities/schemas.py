# Define the schema for config.yml
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

# Define the schema for crds_submission_config.yml
CRDS_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "files_to_submit": {
            "type": "object",
            "properties": {
                "crds_ready_dir": {"type": "string"},
            },
            "required": ["crds_ready_dir"],
        },
        "form_info": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
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
                "instrument",
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
    "required": ["files_to_submit", "form_info"],
}

# Define the schema for pipelines_config.yml
PIPELINES_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {  # List all the possible entries and their types
        "dark": {
            "type": "object",
            "properties": {
                "multiprocess_superdark": {"type": "boolean"},
            },
            "required": ["multiprocess_superdark"],
        },
    },
    # List which entries are needed (all of them)
    "required": ["dark"],
}


# Define the schema for quality_control_config.yml
# NOTE: All elements added to any ref_type control schema MUST be required
QC_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {

        # Dark Quality Control Settings
        "dark_quality_control": {
            "type": "object",
            "properties": {
                # Prep pipeline checks for romancal steps
                "prep_pipeline_checks": {
                    "type": "object",
                    "properties": {
                        "dqinit_step": {"type": "boolean"},
                        "saturation_step": {"type": "boolean"},
                        "refpix_step": {"type": "boolean"},
                        # Final check of prep pipeline
                        "prep_pipeline_step": {"type": "boolean"},
                    },
                    "required": ["dqinit_step", "saturation_step", "refpix_step", "prep_pipeline_step"]
                },
                "prep_pipeline_values": {
                    "type": "object",
                    "properties": {
                        "dqinit_step_status": {
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"]
                        },
                        "saturation_step_status": {
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"]
                        },
                        "refpix_step_status": {
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"]
                        },
                        "prep_pipeline_status": {
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"]
                            #  Only set prep pipeline status to Complete if all step status
                            #  above are Complete
                        },
                    },
                    "required": ["dqinit_step_status", "saturation_step_status", "refpix_step_status", 
                                 "prep_pipeline_status"]
                },
                # Superdark checks
                "superdark_checks": {
                    "type": "object",
                    "properties": {
                        "superdark_step": {"type": "boolean"},
                    },
                    "required": ["superdark_step"]
                },
                "superdark_values": {
                    "type": "object",
                    "properties": {
                        "superdark_step_status": {
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"]
                        },
                    },
                    "required": ["superdark_step_status"]
                },
                # Dark pipeline checks
                "dark_pipeline_checks": {
                    "type": "object",
                    "properties": {
                        "check_mean_dark_rate": {"type": "boolean"},
                        "check_med_dark_rate": {"type": "boolean"},
                        "check_std_dark_rate": {"type": "boolean"},
                        "check_num_hot_pix": {"type": "boolean"},
                        "check_num_dead_pix": {"type": "boolean"},
                        "check_num_unreliable_pix": {"type": "boolean"},
                        "check_num_warm_pix": {"type": "boolean"},
                        # Final check of ref_type pipeline
                        "dark_pipeline_step": { "type": "boolean" },
                    },
                    "required": [
                        "check_mean_dark_rate",
                        "check_med_dark_rate",
                        "check_std_dark_rate",
                        "check_num_hot_pix",
                        "check_num_dead_pix",
                        "check_num_unreliable_pix",
                        "check_num_warm_pix",
                        "dark_pipeline_step",  
                    ],
                },
                "dark_pipeline_values": {
                    "type": "object",
                    "properties": {
                        "max_mean_dark_rate_reference_value": {"type": "number"},
                        "max_med_dark_rate_reference_value": {"type": "number"},
                        "max_std_dark_rate_reference_value": {"type": "number"},
                        "max_num_hot_pix_reference_value": {"type": "number"},
                        "max_num_dead_pix_reference_value": {"type": "number"},
                        "max_num_unreliable_pix_reference_value": {"type": "number"},
                        "max_num_warm_pix_reference_value": {"type": "number"},
                        "dark_pipeline_status": {                            
                            "type": "string",
                            "enum": ["Incomplete", "Complete", "N/A"],  
                            #  Only set ref_type pipeline status to Complete if all checks
                            #  above are True or Ignored.
                        },
                    },
                    "required": [
                        "max_mean_dark_rate_reference_value",
                        "max_med_dark_rate_reference_value",
                        "max_std_dark_rate_reference_value",
                        "max_num_hot_pix_reference_value",
                        "max_num_dead_pix_reference_value",
                        "max_num_unreliable_pix_reference_value",
                        "max_num_warm_pix_reference_value",
                        "dark_pipeline_status",
                    ],
                },
            },
            "required": [
                "prep_pipeline_checks",
                "prep_pipeline_values",
                "superdark_checks",
                "superdark_values",
                "dark_pipeline_checks",
                "dark_pipeline_values",
            ]
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