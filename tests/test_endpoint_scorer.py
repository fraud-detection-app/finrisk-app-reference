"""The endpoint contract, client side: EndpointScorer against a fake ``sagemaker-runtime``.

What is pinned down: the exact InvokeEndpoint request, validation BEFORE any AWS call, every failure becoming
ModelUnavailable (with "model endpoint is starting, retry" for a cold endpoint), which failures are retried and which are not,
the time budget, and DescribeEndpoint cached for 60 s. The fake runtime answers with a real ModelRunner, so the success path is
the real contract. No AWS.
"""

from __future__ import annotations

import json

import pytest
from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

from artifact_fakes import ACCOUNT, FakeEndpoints, client_error, make_runner
from financial_risk.models.training import FEATURE_COLUMNS
from financial_risk.serving.contract import InvalidRequest, frame_to_instances
from financial_risk.service.scoring import STARTING_MESSAGE, EndpointScorer, ModelUnavailable, score_frame

ENDPOINT = "finrisk-prod-fraud"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def aws(tmp_path, demo_gold):
    fake = FakeEndpoints()
    fake.deploy(ENDPOINT, make_runner(demo_gold, tmp_path / "pkg", 3, threshold=0.42))
    return fake


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def scorer(aws, clock):
    return EndpointScorer(ENDPOINT, 3, "ap-southeast-2", runtime=aws.runtime, sagemaker=aws.sagemaker, clock=clock, sleep=clock.sleep)


@pytest.fixture(scope="module")
def instances(demo_gold):
    import pandas as pd

    return frame_to_instances(pd.read_parquet(demo_gold).head(4))


# ------------------------------------------------------------------------------ the request and the response


def test_the_invoke_endpoint_request_is_the_documented_one(scorer, aws, instances):
    result = scorer.score(instances, explain=True, top_k=2)
    (call,) = aws.runtime.calls
    assert call["EndpointName"] == ENDPOINT and call["ContentType"] == "application/json" and call["Accept"] == "application/json"
    assert json.loads(call["Body"]) == {"instances": instances, "explain": True, "top_k": 2}
    assert result["model_version"] == "3" and result["threshold"] == pytest.approx(0.42)
    assert [p["row_id"] for p in result["predictions"]] == [i["row_id"] for i in instances]
    assert all(len(p["reasons"]) == 2 for p in result["predictions"])
    assert scorer.threshold == pytest.approx(0.42) and scorer.last_error is None


def test_explain_defaults_to_off(scorer, aws, instances):
    result = scorer.score(instances)
    assert json.loads(aws.runtime.calls[0]["Body"])["explain"] is False and all("reasons" not in p for p in result["predictions"])


def test_a_bad_row_is_refused_before_any_aws_call(scorer, aws, instances):
    broken = dict(instances[0])
    del broken["amount"]
    with pytest.raises(InvalidRequest, match="missing"):
        scorer.score([broken])
    with pytest.raises(InvalidRequest):
        scorer.score([])
    with pytest.raises(InvalidRequest, match="at most 500"):
        scorer.score(instances * 200)
    assert aws.runtime.calls == []


def test_score_frame_sends_a_feature_table(scorer, aws, demo_gold):
    import pandas as pd

    frame = pd.read_parquet(demo_gold).head(3)
    result = score_frame(scorer, frame)
    assert [p["row_id"] for p in result["predictions"]] == [str(r) for r in frame["row_id"]]
    assert set(json.loads(aws.runtime.calls[0]["Body"])["instances"][0]) == {"row_id", *FEATURE_COLUMNS}      # nothing else leaves the app


# ------------------------------------------------------------------------------ failures


def test_a_read_timeout_is_a_cold_endpoint_and_is_not_retried(scorer, aws, instances, clock):
    aws.runtime.errors = [ReadTimeoutError(endpoint_url="https://runtime.sagemaker.ap-southeast-2.amazonaws.com/endpoints/x/invocations")]
    with pytest.raises(ModelUnavailable) as caught:
        scorer.score(instances)
    assert str(caught.value) == STARTING_MESSAGE == "model endpoint is starting, retry" and caught.value.starting is True
    assert len(aws.runtime.calls) == 1 and clock.slept == []                    # 40 s already passed: a second wait would outlive a Lambda
    assert scorer.last_error == STARTING_MESSAGE
    assert scorer.score(instances)["model_version"] == "3" and scorer.last_error is None      # the next call works and clears it


def test_a_connection_that_fails_fast_is_retried_twice_then_reported(scorer, aws, instances, clock):
    aws.runtime.errors = [ConnectTimeoutError(endpoint_url="x"), EndpointConnectionError(endpoint_url="x"), ConnectTimeoutError(endpoint_url="x")]
    with pytest.raises(ModelUnavailable, match="cannot reach the model endpoint") as caught:
        scorer.score(instances)
    assert len(aws.runtime.calls) == 3 and len(clock.slept) == 2 and caught.value.starting is False


def test_a_retry_that_works_returns_the_answer(scorer, aws, instances, clock):
    aws.runtime.errors = [client_error("ThrottlingException", "Rate exceeded", "InvokeEndpoint"),
                          client_error("ModelNotReadyException", "model is still loading", "InvokeEndpoint")]
    assert scorer.score(instances)["model_version"] == "3"
    assert len(aws.runtime.calls) == 3 and clock.slept == [0.4, 0.8]


def test_model_not_ready_after_all_retries_is_the_starting_message(scorer, aws, instances):
    aws.runtime.errors = [client_error("ModelNotReadyException", "still loading", "InvokeEndpoint")] * 3
    with pytest.raises(ModelUnavailable) as caught:
        scorer.score(instances)
    assert str(caught.value) == STARTING_MESSAGE and caught.value.starting is True and len(aws.runtime.calls) == 3


def test_retries_stop_when_the_time_budget_is_spent(aws, instances, clock):
    scorer = EndpointScorer(ENDPOINT, 3, runtime=aws.runtime, sagemaker=aws.sagemaker, clock=clock, sleep=clock.sleep, budget_seconds=46.0)
    # one worst-case call is 5 + 40 = 45 s: the first pause (0.4 s) fits in 46, the second (0.4 + 0.8) does not
    aws.runtime.errors = [client_error("ServiceUnavailable", "busy", "InvokeEndpoint")] * 3
    with pytest.raises(ModelUnavailable, match="ServiceUnavailable"):
        scorer.score(instances)
    assert len(aws.runtime.calls) == 2


@pytest.mark.parametrize("code, message", [
    ("AccessDeniedException", f"User is not authorized to perform sagemaker:InvokeEndpoint on arn:aws:sagemaker:ap-southeast-2:{ACCOUNT}:endpoint/x"),
    ("ValidationError", f"Endpoint x of account {ACCOUNT} not found."),
])
def test_errors_that_retrying_cannot_fix_are_not_retried_and_hide_the_account_id(scorer, aws, instances, code, message):
    aws.runtime.errors = [client_error(code, message, "InvokeEndpoint")]
    with pytest.raises(ModelUnavailable) as caught:
        scorer.score(instances)
    assert code in str(caught.value) and ACCOUNT not in str(caught.value) and "<account-id>" in str(caught.value)
    assert len(aws.runtime.calls) == 1 and caught.value.starting is False


def test_a_model_error_from_the_container_is_reported_with_its_status(scorer, aws, instances):
    aws.runtime.errors = [client_error("ModelError", "Received client error (400)", "InvokeEndpoint", OriginalStatusCode=400,
                                       OriginalMessage="instance 0: missing ['amount']")]
    with pytest.raises(ModelUnavailable, match=r"rejected the request \(400\): instance 0: missing"):
        scorer.score(instances)
    assert len(aws.runtime.calls) == 1


@pytest.mark.parametrize("raw", [b"<html>502</html>", b'{"predictions": []}', b'{"threshold": 0.5, "predictions": "x"}',
                                 b'{"threshold": 0.5, "predictions": [{"row_id": "1", "score": 0.1, "flagged": false}]}'])
def test_an_answer_that_is_not_the_contract_is_an_error_not_a_guess(scorer, aws, instances, raw):
    import io

    aws.runtime.invoke_endpoint = lambda **kw: {"Body": io.BytesIO(raw)}
    with pytest.raises(ModelUnavailable, match="contract|scored"):
        scorer.score(instances)
    assert scorer.last_error


def test_the_boto3_client_has_the_documented_timeouts_and_no_botocore_retries(tmp_path, monkeypatch):
    # building a client reads ~/.aws: point it at nothing and give it obviously fake keys, so the test never depends on a laptop's setup
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none-credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    client = EndpointScorer(ENDPOINT, 3, "ap-southeast-2")._runtime_client()
    config = client.meta.config
    assert (config.connect_timeout, config.read_timeout) == (5, 40) and config.retries["total_max_attempts"] == 1
    assert client.meta.region_name == "ap-southeast-2"


# ------------------------------------------------------------------------------ DescribeEndpoint for /ready


def test_the_endpoint_status_is_cached_for_sixty_seconds(scorer, aws, clock):
    assert scorer.status() == ("InService", None)
    aws.sagemaker.statuses[ENDPOINT] = "Updating"
    clock.now += 59
    assert scorer.status() == ("InService", None) and aws.sagemaker.calls == [ENDPOINT]
    clock.now += 2
    assert scorer.status() == ("Updating", None) and aws.sagemaker.calls == [ENDPOINT, ENDPOINT]


def test_a_describe_failure_shows_up_in_facts_without_raising(scorer, aws):
    aws.sagemaker.error = client_error("AccessDeniedException", f"not allowed on arn:aws:sagemaker:ap-southeast-2:{ACCOUNT}:endpoint/x", "DescribeEndpoint")
    facts = scorer.facts()
    assert facts["status"] is None and "cannot describe endpoint" in facts["error"] and ACCOUNT not in facts["error"]
    assert facts["endpoint"] == ENDPOINT and facts["threshold"] is None


def test_facts_report_the_last_scoring_error_and_threshold(scorer, aws, instances):
    scorer.score(instances)
    aws.runtime.errors = [ReadTimeoutError(endpoint_url="x")]
    with pytest.raises(ModelUnavailable):
        scorer.score(instances)
    facts = scorer.facts()
    assert facts["status"] == "InService" and facts["error"] == STARTING_MESSAGE and facts["threshold"] == pytest.approx(0.42)


def test_verify_fails_for_an_endpoint_that_does_not_exist(aws, clock):
    scorer = EndpointScorer("finrisk-prod-fraud-gone", 3, runtime=aws.runtime, sagemaker=aws.sagemaker, clock=clock)
    with pytest.raises(Exception, match="Could not find endpoint"):
        scorer.verify()


# ------------------------------------------------------------------------------ the timeouts are settings


def test_the_budget_derives_from_the_timeouts():
    assert EndpointScorer(ENDPOINT, 3).budget == 50.0                                              # 5 + 40 + 5
    assert EndpointScorer(ENDPOINT, 3, connect_timeout=5, read_timeout=25).budget == 35.0          # Lambda: 20 inline + 35 <= 60
    assert EndpointScorer(ENDPOINT, 3, connect_timeout=2, read_timeout=10, slack_seconds=1).budget == 13.0
    assert EndpointScorer(ENDPOINT, 3, budget_seconds=7).budget == 7


def test_retries_use_the_slack_and_never_push_a_call_past_the_budget(aws, instances, clock):
    def run(slack: float) -> int:
        aws.runtime.calls.clear()
        clock.slept.clear()
        scorer = EndpointScorer(ENDPOINT, 3, runtime=aws.runtime, sagemaker=aws.sagemaker, clock=clock, sleep=clock.sleep,
                                connect_timeout=5, read_timeout=25, slack_seconds=slack)          # one worst-case call = 30 s
        aws.runtime.errors = [client_error("ThrottlingException", "slow down", "InvokeEndpoint")] * 3
        with pytest.raises(ModelUnavailable):
            scorer.score(instances)
        return len(aws.runtime.calls)

    assert run(slack=5) == 3            # budget 35: the pauses (0.4 + 0.8) fit in the 5 s of slack -> both retries happen
    assert run(slack=1) == 2            # budget 31: 0.4 + 0.8 + 30 > 31 -> the second retry would overrun, so it is not started


def test_the_settings_reach_the_boto3_client_config(tmp_path, monkeypatch):
    import dataclasses

    from financial_risk.config import get_settings
    from financial_risk.service.artifacts import ArtifactStore

    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none-credentials"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("ENDPOINT_READ_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ENDPOINT_CONNECT_TIMEOUT_SECONDS", raising=False)
    defaults = get_settings()
    assert (defaults.endpoint_connect_timeout_seconds, defaults.endpoint_read_timeout_seconds) == (5.0, 40.0)

    monkeypatch.setenv("ENDPOINT_READ_TIMEOUT_SECONDS", "25")                                      # what Terraform sets on Lambda
    monkeypatch.setenv("ENDPOINT_CONNECT_TIMEOUT_SECONDS", "3")
    settings = get_settings()
    assert (settings.endpoint_connect_timeout_seconds, settings.endpoint_read_timeout_seconds) == (3.0, 25.0)
    scorer = ArtifactStore(dataclasses.replace(settings, pointer_prefix="/finrisk/dev"), ssm_client=object())._scorer_factory(
        ENDPOINT, 3, "ap-southeast-2")
    config = scorer._runtime_client().meta.config
    assert (config.connect_timeout, config.read_timeout) == (3, 25) and scorer.budget == 33.0
