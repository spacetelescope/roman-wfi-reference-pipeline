QC_CONFIG_SCHEMA_FLAT = {
    "type": "object",
    "properties": {
        "prep_pipeline": {
            "type": "object",
            "properties": {
                "checks": {
                    "type": "object",
                    "properties": {
                        "dqinit": {"type": "boolean"},
                        "refpix": {"type": "boolean"},
                        "saturation": {"type": "boolean"},
                        "linearity": {"type": "boolean"},
                        "darkcurrent": {"type": "boolean"},
                        "rampfit": {"type": "boolean"},
                    },
                    "required": [
                        "dqinit",
                        "refpix",
                        "saturation",
                        "linearity",
                        "darkcurrent",
                        "rampfit",
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
                        "mean_flat_rate": {"type": "boolean"},
                        "med_flat_rate": {"type": "boolean"},
                        "std_flat_rate": {"type": "boolean"},
                        "num_lowqe_pix": {"type": "boolean"},
                    },
                    "required": [
                        "mean_flat_rate",
                        "med_flat_rate",
                        "std_flat_rate",
                        "num_lowqe_pix",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "max_mean_flat_rate": {"type": "number"},
                        "max_med_flat_rate": {"type": "number"},
                        "max_std_flat_rate": {"type": "number"},
                        "max_num_lowqe_pix": {"type": "number"},
                    },
                    "required": [
                        "max_mean_flat_rate",
                        "max_med_flat_rate",
                        "max_std_flat_rate",
                        "max_num_lowqe_pix",
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
                    "properties": {"romancal_flat": {"type": "boolean"}},
                    "required": ["romancal_flat"],
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
