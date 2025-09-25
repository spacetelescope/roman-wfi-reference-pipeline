QC_CONFIG_SCHEMA_DARK = {
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
                    },
                    "required": ["dqinit", "saturation", "refpix"],
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
                        "mean_dark_rate": {"type": "boolean"},
                        "med_dark_rate": {"type": "boolean"},
                        "std_dark_rate": {"type": "boolean"},
                        "num_hot_pix": {"type": "boolean"},
                        "num_dead_pix": {"type": "boolean"},
                        "num_warm_pix": {"type": "boolean"},
                    },
                    "required": [
                        "mean_dark_rate",
                        "med_dark_rate",
                        "std_dark_rate",
                        "num_hot_pix",
                        "num_dead_pix",
                        "num_warm_pix",
                    ],
                },
                "values": {
                    "type": "object",
                    "properties": {
                        "max_mean_dark_rate": {"type": "number"},
                        "max_med_dark_rate": {"type": "number"},
                        "max_std_dark_rate": {"type": "number"},
                        "max_num_hot_pix": {"type": "number"},
                        "max_num_dead_pix": {"type": "number"},
                        "max_num_warm_pix": {"type": "number"},
                    },
                    "required": [
                        "max_mean_dark_rate",
                        "max_med_dark_rate",
                        "max_std_dark_rate",
                        "max_num_hot_pix",
                        "max_num_dead_pix",
                        "max_num_warm_pix",
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
                    "properties": {"romancal_dark": {"type": "boolean"}},
                    "required": ["romancal_dark"],
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
