from .quality_control.schema.dark_qc_schema import QC_CONFIG_SCHEMA_DARK
from .quality_control.schema.flat_qc_schema import QC_CONFIG_SCHEMA_FLAT
from .quality_control.schema.readnoise_qc_schema import QC_CONFIG_SCHEMA_READNOISE
from .quality_control.schema.refpix_qc_schema import QC_CONFIG_SCHEMA_REFPIX

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

QC_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "dark": QC_CONFIG_SCHEMA_DARK,
        "flat": QC_CONFIG_SCHEMA_FLAT,
        "readnoise": QC_CONFIG_SCHEMA_READNOISE,
        "refpix": QC_CONFIG_SCHEMA_REFPIX,
    },
    "required": ["dark", "flat", 'readnoise', 'refpix']
}