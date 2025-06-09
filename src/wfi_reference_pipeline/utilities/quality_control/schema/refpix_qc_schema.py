
QC_CONFIG_SCHEMA_REFPIX = {
    "type": "object",
    "properties": {
        "refpix": {
            "type": "object",
            "properties": {
                # Prep pipeline checks
                "prep_pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "dqinit_step": {"type": "boolean"},
                                "saturation_step": {"type": "boolean"},
                                "prep_pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "dqinit_step", "saturation_step", "prep_pipeline_step"
                            ]
                        },
                        "values": {
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
                                "prep_pipeline_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                },
                            },
                            "required": [
                                "dqinit_step_status", "saturation_step_status", "prep_pipeline_step_status"
                            ]
                        }
                    }
                },
                # Refpix pipeline checks
                "pipeline": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "check_refpix_spectral_slope": {"type": "boolean"},
                                "pipeline_step": {"type": "boolean"},
                            },
                            "required": [
                                "check_refpix_spectral_slope", "pipeline_step"
                            ]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "refpix_spectral_slope": {"type": "number"},
                                "pipeline_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                }
                            },
                            "required": [
                                "refpix_spectral_slope", "pipeline_step_status"
                            ]
                        }
                    }
                },
                # Pre-delivery checks
                "pre_delivery": {
                    "type": "object",
                    "properties": {
                        "checks": {
                            "type": "object",
                            "properties": {
                                "romancal_refpix_step": {"type": "boolean"}
                            },
                            "required": ["romancal_refpix_step"]
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "romancal_refpix_step_status": {
                                    "type": "string",
                                    "enum": ["Incomplete", "Complete", "N/A"]
                                }
                            },
                            "required": ["romancal_refpix_step_status"]
                        }
                    }
                },
                # Summary
                "refpix_quality_summary": {"type": "string"}
            },
            "required": ["prep_pipeline", "pipeline", "pre_delivery", "refpix_quality_summary"]
        }
    }
}