
QC_CONFIG_SCHEMA_REFPIX = {
    "type": "object",
    "properties": {
        "prep_pipeline": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {
                        "dqinit": {"type": "boolean"},
                        "saturation": {"type": "boolean"},
                    },
                    "required": [
                        "dqinit",
                        "saturation",
                    ],
                },
            },
            "required": ["checks"],
        },
        "pipeline": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {
                        "refpix_spectral_slope": {"type": "boolean"},
                    },
                    "required": [
                        "refpix_spectral_slope",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "refpix_spectral_slope": {"type": "number"},
                    },
                    "required": [
                        "refpix_spectral_slope",
                    ],
                },
            },
            "required": ["checks", "values"],
        },
        "pre_delivery": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {"romancal_refpix": {"type": "boolean"}},
                    "required": ["romancal_refpix"],
                }
            },
            "required": ["checks"],
        },
        "delivery": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {"quality_summary": {"type": "boolean"}},
                    "required": ["quality_summary"],
                }
            },
            "required": ["checks"],
        },
    },
    "required": ["prep_pipeline", "pipeline", "pre_delivery", "delivery"],
}