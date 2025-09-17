QC_CONFIG_SCHEMA_DARK = {
    "type": "object",
    "properties": {
        "dark": {
            "type": "object",
            "properties": {
                "prep_pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "dqinit_step": {"type": "boolean"},
                                "saturation_step": {"type": "boolean"},
                                "refpix_step": {"type": "boolean"},
                                "prep_pipeline_step": {"type": "boolean"}
                            },
                            "required": ["dqinit_step", "saturation_step", "refpix_step", "prep_pipeline_step"]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "dqinit_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "saturation_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "refpix_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "prep_pipeline_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]}
                            },
                            "required": ["dqinit_step_status", "saturation_step_status", "refpix_step_status", "prep_pipeline_step_status"]
                        }
                    }
                },
                "superdark": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "superdark_step": {"type": "boolean"}
                            },
                            "required": ["superdark_step"]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "superdark_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                }
                            },
                            "required": ["superdark_step_status"]
                        }
                    },
                    "required": ["checks", "values"]
                },
                "pipeline": {
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
                                "check_num_warm_pix": {"type": "boolean"},
                                "dark_pipeline_step": {"type": "boolean"}
                            },
                            "required": [
                                "check_mean_dark_rate",
                                "check_med_dark_rate",
                                "check_std_dark_rate",
                                "check_num_hot_pix",
                                "check_num_dead_pix",
                                "check_num_warm_pix",
                                "dark_pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "max_mean_dark_rate_reference_value": {"type": "number"},
                                "max_med_dark_rate_reference_value": {"type": "number"},
                                "max_std_dark_rate_reference_value": {"type": "number"},
                                "max_num_hot_pix_reference_value": {"type": "number"},
                                "max_num_dead_pix_reference_value": {"type": "number"},
                                "max_num_warm_pix_reference_value": {"type": "number"},
                                "pipeline_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                }
                            },
                            "required": [
                                "max_mean_dark_rate_reference_value",
                                "max_med_dark_rate_reference_value",
                                "max_std_dark_rate_reference_value",
                                "max_num_hot_pix_reference_value",
                                "max_num_dead_pix_reference_value",
                                "max_num_warm_pix_reference_value",
                                "pipeline_step_status"
                            ]
                        }
                    },
                    "required": ["checks", "values"]
                },
                "pre_delivery_checks": {
                    "type": "object",
                    "properties": {
                        "romancal_dark_step": {"type": "boolean"}
                    },
                    "required": ["romancal_dark_step"]
                },
                "pre_delivery_values": {
                    "type": "object",
                    "properties": {
                        "romancal_dark_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]}
                    },
                    "required": ["romancal_dark_step_status"]
                }
            },
            "required": [
                "prep_pipeline",
                "superdark",
                "pipeline",
                "pre_delivery_checks",
                "pre_delivery_values"
            ]
        }
    }
}