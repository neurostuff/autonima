from autonima.coordinates.openai_client import _sanitize_parse_result


def test_sanitize_parse_result_keeps_valid_points_around_malformed_items():
    result = _sanitize_parse_result(
        {
            "analyses": [
                {
                    "name": "A > B",
                    "points": [
                        "not-a-point",
                        {
                            "coordinates": [1, 2, 3],
                            "values": [
                                "not-a-value",
                                {"value": 4.2, "kind": "t-statistic"},
                            ],
                        },
                        {"coordinates": [1, 2]},
                        {"coordinates": [1, 2, 1122]},
                        {"coordinates": [1, float("nan"), 3]},
                    ],
                },
                "not-an-analysis",
            ]
        }
    )

    assert len(result["analyses"]) == 1
    assert result["analyses"][0]["points"] == [
        {
            "coordinates": [1, 2, 3],
            "values": [{"value": 4.2, "kind": "t-statistic"}],
        }
    ]
