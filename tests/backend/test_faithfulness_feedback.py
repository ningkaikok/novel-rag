from scripts.eval_faithfulness_feedback import aggregate


def test_feedback_aggregation_is_grouped_by_method_and_model():
    result = aggregate(
        [
            {"feedback": "helpful", "label": "supported", "method": "shadow_two_step", "model": "m1"},
            {"feedback": "incorrect", "label": "supported", "method": "shadow_two_step", "model": "m1"},
            {"feedback": "incorrect", "label": "unsupported", "method": "shadow_two_step", "model": "m1"},
        ]
    )
    item = result["shadow_two_step|m1"]
    assert item["samples"] == 3
    assert item["agreement"] == 2 / 3
    assert item["matrix"]["supported|supported"] == 1
    assert item["matrix"]["unsupported|unsupported"] == 1
