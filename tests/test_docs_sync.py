from scripts.check_docs_sync import deployment_expectations, documented_endpoints


def test_documented_endpoints_extract_only_contract_headings() -> None:
    assert documented_endpoints("## `GET /health`\n### `POST /ignored`\n") == {("GET", "/health")}


def test_deployment_expectations_detect_drift() -> None:
    template = {
        "Parameters": {
            "FunctionMemory": {"Default": 1024},
            "ReservedConcurrency": {"Default": 2},
        },
        "Globals": {"Function": {"Timeout": 60}},
        "Resources": {"PetCareAiFunctionLogGroup": {"Properties": {"RetentionInDays": 30}}},
    }
    missing = deployment_expectations(template, "defaults to `1024` MB")
    assert missing == ["defaults to `2`", "`Timeout=60`", "for 30 days"]
