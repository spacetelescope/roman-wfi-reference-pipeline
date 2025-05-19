QC_CONFIG_SCHEMA_READNOISE = {
    "type": "object",
    "properties": {
        "readnoise": {
            "type": "object",
            "properties": {
                # Prep pipeline checks for romancal steps
                "prep_pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "dqinit_step": {"type": "boolean"},
                                "saturation_step": {"type": "boolean"},
                                "refpix_step": {"type": "boolean"},
                                "linearity_step": {"type": "boolean"},
                                "prep_pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "dqinit_step", "saturation_step", "refpix_step",
                                "linearity_step", "prep_pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "dqinit_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "saturation_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "refpix_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "linearity_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "prep_pipeline_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                            },
                            "required": [
                                "dqinit_step_status", "saturation_step_status", "refpix_step_status",
                                "linearity_step_status", "prep_pipeline_step_status"
                            ]
                        },
                    }
                },
                # Readnoise pipeline checks
                "pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "check_mean_readnoise": {"type": "boolean"},
                                "check_med_readnoise": {"type": "boolean"},
                                "check_std_readnoise": {"type": "boolean"},
                                "pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "check_mean_readnoise", "check_med_readnoise",
                                "check_std_readnoise", "pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "max_mean_readnoise_reference_value": {"type": "number"},
                                "max_med_readnoise_reference_value": {"type": "number"},
                                "max_std_readnoise_reference_value": {"type": "number"},
                                "pipeline_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                },
                            },
                            "required": [
                                "max_mean_readnoise_reference_value", "max_med_readnoise_reference_value",
                                "max_std_readnoise_reference_value", "pipeline_step_status"
                            ]
                        },
                    }
                },
                # Pre-delivery checks
                "pre_delivery": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "romancal_rampfit_step": {"type": "boolean"}
                            },
                            "required": ["romancal_rampfit_step"]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "romancal_rampfit_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                }
                            },
                            "required": ["romancal_rampfit_step_status"]
                        }
                    }
                },
                # Summary
                "readnoise_quality_summary": {"type": "string"}
            },
            "required": ["prep_pipeline", "pipeline", "pre_delivery", "readnoise_quality_summary"]
        }
    }
}