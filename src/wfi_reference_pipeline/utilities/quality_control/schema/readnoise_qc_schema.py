QC_CONFIG_SCHEMA_READNOISE = {
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
                        "refpix": {"type": "boolean"},
                        "linearity": {"type": "boolean"},
                    },
                    "required": [
                        "dqinit",
                        "saturation",
                        "refpix",
                        "linearity",
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
                        "mean_readnoise": {"type": "boolean"},
                        "med_readnoise": {"type": "boolean"},
                        "std_readnoise": {"type": "boolean"},
                    },
                    "required": [
                        "mean_readnoise",
                        "med_readnoise",
                        "std_readnoise",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "max_mean_readnoise": {"type": "number"},
                        "max_med_readnoise": {"type": "number"},
                        "max_std_readnoise": {"type": "number"},
                    },
                    "required": [
                        "max_mean_readnoise",
                        "max_med_readnoise",
                        "max_std_readnoise",
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
                    "properties": {"romancal_rampfit": {"type": "boolean"}},
                    "required": ["romancal_rampfit"],
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