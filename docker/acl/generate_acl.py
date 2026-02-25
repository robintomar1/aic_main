import json5
from pathlib import Path

allow_topics_eval_to_model: list[str] = [
    "/aic_controller/controller_state",
    "/joint_states",
    "/observations",
    "/left_camera/camera_info",
    "/center_camera/camera_info",
    "/right_camera/camera_info",
]

allow_topics_bidi: list[str] = [
    "/aic_controller/change_target_mode",
    "/aic_controller/joint_commands",
    "/aic_controller/pose_commands",
]

allow_services_bidi: list[str] = [
    "/aic_model/change_state",
    "/aic_model/get_state",
    "/cancel_task",
]

allow_actions_bidi: list[str] = [
    "/gripper_action_controller/gripper_cmd",
    "/insert_cable",
]

allow_topics_eval_to_model_key_exprs = [
    expr
    for topic in allow_topics_eval_to_model
    for expr in [
        f"@ros2_lv/**/%{topic.strip('/').replace('/', '%')}/**",
        f"*/{topic.strip('/')}/*/*",
        f"*/{topic.strip('/')}/*/*/@adv/**",
    ]
]

outgoing_publications = {
    "id": "outgoing_publications",
    "flows": ["egress"],
    "key_exprs": allow_topics_eval_to_model_key_exprs,
    "messages": [
        "put",
        "reply",
        "declare_queryable",
        "liveliness_token",
    ],
    "permission": "allow",
}

incoming_subscriptions = {
    "id": "incoming_subscriptions",
    "flows": ["ingress"],
    "key_exprs": allow_topics_eval_to_model_key_exprs,
    "messages": [
        "declare_subscriber",
        "query",
        "declare_liveliness_subscriber",
        "liveliness_query",
    ],
    "permission": "allow",
}

incoming_publications = {
    "id": "incoming_publications",
    "flows": ["ingress"],
    "key_exprs": allow_topics_eval_to_model_key_exprs,
    "messages": [
        "put",
        "reply",
        "declare_queryable",
        "liveliness_token",
    ],
    "permission": "allow",
}

outgoing_subscriptions = {
    "id": "outgoing_subscriptions",
    "flows": ["egress"],
    "key_exprs": allow_topics_eval_to_model_key_exprs,
    "messages": [
        "declare_subscriber",
        "query",
        "declare_liveliness_subscriber",
    ],
    "permission": "allow",
}

bidi_key_exprs = (
    [
        expr
        for topic in allow_topics_bidi
        for expr in [
            f"@ros2_lv/**/%{topic.strip('/').replace('/', '%')}/**",
            f"*/{topic.strip('/')}/*/*",
            f"*/{topic.strip('/')}/*/*/@adv/**",
        ]
    ]
    + [
        expr
        for service in allow_services_bidi
        for expr in [
            f"@ros2_lv/**/%{service.strip('/').replace('/', '%')}/**",
            f"*/{service.strip('/')}/*/*",
            f"*/{service.strip('/')}/@adv/**",
        ]
    ]
    + [
        expr
        for action in allow_actions_bidi
        for sub in ["send_goal", "cancel_goal", "get_result", "feedback", "status"]
        for expr in [
            f"@ros2_lv/**/%{action.strip('/').replace('/', '%')}/**",
            f"*/{action.strip('/')}/_action/{sub}/*/*",
            f"*/{action.strip('/')}/_action/{sub}/@adv/**",
        ]
    ]
)


bidi = {
    "id": "bidi",
    "flows": ["ingress", "egress"],
    "key_exprs": bidi_key_exprs,
    "messages": [
        "declare_subscriber",
        "put",
        "delete",
        "declare_queryable",
        "query",
        "reply",
        "liveliness_token",
        "declare_liveliness_subscriber",
        "liveliness_query",
    ],
    "permission": "allow",
}

base_config_path = Path(
    Path(__file__).parent.resolve(), "zenoh_router_config_base.json5"
)
output_config_path = Path(
    Path(__file__).parent.resolve(), "..", "aic_eval", "zenoh_router_config.json5"
)
with base_config_path.open() as base_config_fp:
    zenoh_config = json5.load(base_config_fp)
    zenoh_config["access_control"] = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            bidi,
            outgoing_publications,
            incoming_publications,
            outgoing_subscriptions,
            incoming_subscriptions,
            {
                "id": "all_incoming_liveliness",
                "flows": ["ingress"],
                "key_exprs": [
                    "@ros2_lv/**",
                ],
                "messages": [
                    "liveliness_token",
                    "declare_liveliness_subscriber",
                    "liveliness_query",
                ],
                "permission": "allow",
            },
        ],
        "subjects": [
            {
                "id": "all",
            },
            {
                "id": "eval",
                "usernames": [
                    "eval",
                ],
            },
        ],
        "policies": [
            {
                "rules": [
                    "bidi",
                    "outgoing_publications",
                    "incoming_subscriptions",
                    "all_incoming_liveliness",
                ],
                "subjects": ["all"],
            },
            {
                "rules": [
                    "incoming_publications",
                    "outgoing_subscriptions",
                ],
                "subjects": ["eval"],
            },
        ],
    }

with output_config_path.open("w") as output_config_fp:
    output_config_fp.write("// generated by docker/acl/generate_acl.py\n")
    json5.dump(zenoh_config, output_config_fp, indent=2)
