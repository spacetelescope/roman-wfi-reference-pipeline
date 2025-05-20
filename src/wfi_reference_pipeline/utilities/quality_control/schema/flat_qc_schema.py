QC_CONFIG_SCHEMA_FLAT = {
    "type": "object",
    "properties": {
        "flat": {
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
                                "refpix_step": {"type": "boolean"},
                                "saturation_step": {"type": "boolean"},
                                "linearity_step": {"type": "boolean"},
                                "darkcurrent_step": {"type": "boolean"},
                                "rampfit_step": {"type": "boolean"},
                                "prep_pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "dqinit_step", "refpix_step", "saturation_step",
                                "linearity_step", "darkcurrent_step", "rampfit_step",
                                "prep_pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "dqinit_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "refpix_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "saturation_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "linearity_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "darkcurrent_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "rampfit_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                                "prep_pipeline_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]},
                            },
                            "required": [
                                "dqinit_step_status", "refpix_step_status", "saturation_step_status",
                                "linearity_step_status", "darkcurrent_step_status", "rampfit_step_status",
                                "prep_pipeline_step_status"
                            ]
                        },
                    }
                },
                # Flat pipeline checks
                "pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "check_mean_flat_rate": {"type": "boolean"},
                                "check_med_flat_rate": {"type": "boolean"},
                                "check_std_flat_rate": {"type": "boolean"},
                                "check_num_lowqe_pix": {"type": "boolean"},
                                "pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "check_mean_flat_rate", "check_med_flat_rate",
                                "check_std_flat_rate", "check_num_lowqe_pix",
                                "pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "mean_flat_rate_reference_value": {"type": "number"},
                                "med_flat_rate_reference_value": {"type": "number"},
                                "max_std_flat_rate_reference_value": {"type": "number"},
                                "max_num_lowqe_pix_reference_value": {"type": "number"},
                                "pipeline_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                },
                            },
                            "required": [
                                "mean_flat_rate_reference_value", "med_flat_rate_reference_value",
                                "max_std_flat_rate_reference_value", "max_num_lowqe_pix_reference_value",
                                "pipeline_step_status"
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
                                "romancal_flat_step": {"type": "boolean"}
                            },
                            "required": ["romancal_flat_step"]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "romancal_flat_step_status": {"type": "string", "enum": ["Incomplete", "Complete", "N/A"]}
                            },
                            "required": ["romancal_flat_step_status"]
                        }
                    }
                },
                # Summary
                "flat_quality_summary": {"type": "string"}
            },
            "required": ["prep_pipeline", "pipeline", "pre_delivery", "flat_quality_summary"]
        }  
    }
}