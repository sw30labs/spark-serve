"""Offline validation of the GLM pilot's objective acceptance checks."""
import base64
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import glm53_acceptance as acceptance


def args(tmp_path, **overrides):
    options = dict(output=str(tmp_path / "receipt.json"), model="glm-5.3-flash",
                   url="http://localhost:8000/v1", tokenizer_url="http://localhost:8000/v1", timeout=30, max_tokens=2048,
                   context_limit=131072, reasoning_effort="low", tool_choice="named", api_key_env="TEST_GLM_KEY",
                   long_context=False, context_targets=[8192, 32768], only=list(acceptance.CASES))
    options.update(overrides)
    return SimpleNamespace(**options)


def response(content, *, calls=None, finish="stop", prompt_tokens=100):
    return {"model": "glm-5.3-flash", "content": content, "reasoning": "A short private trace.",
            "tool_calls": calls or [], "finish_reason": finish,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 32, "total_tokens": prompt_tokens + 32},
            "metrics": {"elapsed_seconds": 1.5, "first_model_output_seconds": .1, "ttft_seconds": .5}}


def test_stream_requires_real_model_and_counts_reasoning_separately():
    stream = acceptance.ModelStream(10, "glm-5.3-flash")
    stream.feed(json.dumps({"model": "glm-5.3-flash", "choices": [{"delta": {"reasoning_content": "thinking"}}]}), 11)
    stream.feed(json.dumps({"model": "glm-5.3-flash", "choices": [{"delta": {"content": "391"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}), 12)
    stream.feed("[DONE]", 13)
    actual = stream.finish(14)
    assert actual["model"] == "glm-5.3-flash"
    assert actual["metrics"]["first_model_output_seconds"] == 1
    assert actual["metrics"]["ttft_seconds"] == 2
    assert actual["usage"]["completion_tokens"] == 20
    assert actual["metrics"]["nonempty_delta_count_not_tokens"] == 2


@pytest.mark.parametrize("model", [None, "qwen3.8-flash-next"])
def test_other_or_missing_stream_model_cannot_pass(model):
    stream = acceptance.ModelStream(10, "glm-5.3-flash")
    stream.feed(json.dumps({"model": model, "choices": [{"delta": {"content": "391"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}), 11)
    stream.feed("[DONE]", 12)
    with pytest.raises(AssertionError, match="identity mismatch"):
        stream.finish(13)


@pytest.mark.parametrize("body", ["min(max(value, low), high)", "max(low, min(high, value))"])
def test_safe_code_interpreter_verifies_known_outputs(body):
    result = acceptance.validate_clamp("def clamp(value, low, high):\n    return " + body)
    assert result["known_output_cases"] == 9


@pytest.mark.parametrize("code,match", [
    ("def clamp(value, low, high):\n return value", "Clamp failed"),
    ("def clamp(value, low, high):\n return __import__('os').system('touch /tmp/never')", "Unsupported code"),
    ("def clamp(value, low, high):\n return value.__class__", "Unsupported code"),
    ("def clamp(value, low, high):\n while True: pass", "one return"),
    ("@dangerous\ndef clamp(value, low, high):\n return value", "decorators"),
    ("def clamp(value, low, high=evil()):\n return value", "signature"),
    ("def clamp(value, low, high):\n return min(max(value, low), high)\nattack()", "one function"),
])
def test_generated_code_never_reaches_arbitrary_execution(code, match):
    with pytest.raises(AssertionError, match=match):
        acceptance.validate_clamp(code)


def test_code_size_is_bounded_before_parse():
    with pytest.raises(AssertionError, match="4096"):
        acceptance.validate_clamp("x" * 4097)


@pytest.mark.parametrize("tool_choice,finish", [("named", "stop"), ("named", "tool_calls"), ("auto", "tool_calls")])
def test_tools_validate_schema_and_preserve_reasoning_call_id_roundtrip(tmp_path, monkeypatch, tool_choice, finish):
    monkeypatch.setattr(acceptance.secrets, "token_hex", lambda size: "ab" * size)
    call = {"id": "call-42", "type": "function", "function": {"name": "lookup_archive", "arguments": '{"record":"spark-pilot"}'}}
    mock = Mock(side_effect=[response("", calls=[call], finish=finish), response("READY-" + "AB" * 8)])
    monkeypatch.setattr(acceptance, "http_request", mock)
    runner = acceptance.Qualification(args(tmp_path, tool_choice=tool_choice))
    result = runner.tools()
    first, second = [entry.args[2] for entry in mock.call_args_list]
    assert first["tool_choice"] == ("auto" if tool_choice == "auto" else
                                    {"type": "function", "function": {"name": "lookup_archive"}})
    assert second["messages"][1]["tool_calls"][0]["id"] == "call-42"
    assert second["messages"][1]["reasoning_content"] == "A short private trace."
    assert second["messages"][2]["tool_call_id"] == "call-42"
    assert second["tool_choice"] == "none"
    assert all(payload["reasoning_effort"] == "low" and "chat_template_kwargs" not in payload for payload in (first, second))
    assert result["real_tool_actions_executed"] == 0
    assert result["tool_choice"] == tool_choice
    assert "reasoning" not in runner.requests[0]


@pytest.mark.parametrize("tool_choice,finish", [("auto", "stop"), ("auto", "length"),
                                               ("named", "length"), ("named", "error")])
def test_tool_finish_convention_does_not_accept_auto_stop_or_truncation(tmp_path, monkeypatch, tool_choice, finish):
    call = {"id": "call-1", "type": "function", "function": {"name": "lookup_archive", "arguments": '{"record":"spark-pilot"}'}}
    mock = Mock(return_value=response("", calls=[call], finish=finish))
    monkeypatch.setattr(acceptance, "http_request", mock)
    case = acceptance.Qualification(args(tmp_path, tool_choice=tool_choice)).run_case("tools")
    assert not case["passed"]
    assert "Expected one completed tool call" in case["error"]
    assert mock.call_count == 1


@pytest.mark.parametrize("count", [0, 2])
def test_named_stop_still_requires_exactly_one_call(tmp_path, monkeypatch, count):
    call = {"id": "call-1", "type": "function", "function": {"name": "lookup_archive", "arguments": '{"record":"spark-pilot"}'}}
    mock = Mock(return_value=response("", calls=[call] * count, finish="stop"))
    monkeypatch.setattr(acceptance, "http_request", mock)
    case = acceptance.Qualification(args(tmp_path)).run_case("tools")
    assert not case["passed"]
    assert "Expected one completed tool call" in case["error"]
    assert mock.call_count == 1


@pytest.mark.parametrize("call_id,call_type,name", [("", "function", "lookup_archive"),
                                                   ("call-1", "other", "lookup_archive"),
                                                   ("call-1", "function", "wrong_tool")])
def test_named_stop_still_validates_tool_identity(tmp_path, monkeypatch, call_id, call_type, name):
    call = {"id": call_id, "type": call_type, "function": {"name": name, "arguments": '{"record":"spark-pilot"}'}}
    mock = Mock(return_value=response("", calls=[call], finish="stop"))
    monkeypatch.setattr(acceptance, "http_request", mock)
    case = acceptance.Qualification(args(tmp_path)).run_case("tools")
    assert not case["passed"]
    assert "Invalid tool identity/type/name" in case["error"]
    assert mock.call_count == 1


def test_named_stop_still_requires_exact_tool_result_replay(tmp_path, monkeypatch):
    call = {"id": "call-1", "type": "function", "function": {"name": "lookup_archive", "arguments": '{"record":"spark-pilot"}'}}
    mock = Mock(side_effect=[response("", calls=[call], finish="stop"), response("invented marker")])
    monkeypatch.setattr(acceptance, "http_request", mock)
    case = acceptance.Qualification(args(tmp_path)).run_case("tools")
    assert not case["passed"]
    assert "Tool result was not used exactly" in case["error"]
    assert mock.call_count == 2


@pytest.mark.parametrize("arguments", ['{"record":"spark-pilot","extra":true}', '{"record":true}', 'not JSON'])
@pytest.mark.parametrize("finish", ["tool_calls", "stop"])
def test_invalid_tool_arguments_do_not_reach_roundtrip(tmp_path, monkeypatch, arguments, finish):
    call = {"id": "call-1", "type": "function", "function": {"name": "lookup_archive", "arguments": arguments}}
    mock = Mock(return_value=response("", calls=[call], finish=finish))
    monkeypatch.setattr(acceptance, "http_request", mock)
    case = acceptance.Qualification(args(tmp_path)).run_case("tools")
    assert not case["passed"]
    assert mock.call_count == 1
    assert case["requests"][0]["tool_calls"] == [call]


def test_vision_uses_generated_png_data_uri_and_checks_ground_truth(tmp_path, monkeypatch):
    mock = Mock(return_value=response('{"text":"SPARK 73","square":"red","circle":"blue"}'))
    monkeypatch.setattr(acceptance, "http_request", mock)
    result = acceptance.Qualification(args(tmp_path)).vision()
    uri = mock.call_args.args[2]["messages"][0]["content"][1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
    assert len(result["fixture_sha256"]) == 64
    schema = mock.call_args.args[2]["response_format"]
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["additionalProperties"] is False
    assert set(schema["json_schema"]["schema"]["required"]) == {"text", "square", "circle"}
    assert result["structured_output"] == "strict json_schema"


def fixture_markers(prompt):
    return dict(re.findall(r"Archive entry (early|middle|late): the recovery marker is (REC-[A-F0-9]+)\.", prompt))


def test_documents_smoke_is_small_no_tokenizer_or_long_requests(tmp_path, monkeypatch):
    def respond(url, endpoint, payload, *unused, **kwargs):
        assert endpoint == "chat/completions"
        prompt = payload["messages"][0]["content"]
        assert prompt.count("Routine inspection") == 64
        schema = payload["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["properties"] == {key: {"type": "string"} for key in ("early", "middle", "late")}
        assert schema["schema"]["additionalProperties"] is False
        assert "REC-" not in json.dumps(schema)
        return response(json.dumps(fixture_markers(prompt)), prompt_tokens=2048)
    monkeypatch.setattr(acceptance, "http_request", respond)
    result = acceptance.Qualification(args(tmp_path)).documents()
    assert len(result["stages"]) == 1
    assert result["stages"][0]["actual_prompt_tokens"] == 2048
    assert result["full_context_window_validated"] is False


def test_long_documents_size_with_tokenizer_and_verify_measured_progress(tmp_path, monkeypatch):
    endpoints = []
    def respond(url, endpoint, payload, *unused, **kwargs):
        endpoints.append(endpoint)
        prompt = payload.get("prompt") or payload["messages"][0]["content"]
        count = prompt.count("Routine inspection") * 24 + 128
        if endpoint == "tokenize":
            assert kwargs["root"] is True
            return {"count": count}
        return response(json.dumps(fixture_markers(prompt)), prompt_tokens=count + 32)
    monkeypatch.setattr(acceptance, "http_request", respond)
    runner = acceptance.Qualification(args(tmp_path, long_context=True))
    result = runner.documents()
    assert endpoints == ["tokenize", "tokenize", "chat/completions"] * 2
    counts = [stage["actual_prompt_tokens"] for stage in result["stages"]]
    assert 8192 * .9 <= counts[0] <= 8192 * 1.1
    assert 32768 * .9 <= counts[1] <= 32768 * 1.1
    assert all("document" in request for request in runner.requests)


@pytest.mark.parametrize("fits_reserve", [True, False])
def test_explicit_800k_document_is_not_clipped_and_preserves_budget_guard(tmp_path, monkeypatch, fits_reserve):
    tokenized_rows, endpoints = [], []
    context_limit, max_tokens = 850000, 2048

    def respond(url, endpoint, payload, *unused, **kwargs):
        endpoints.append(endpoint)
        prompt = payload.get("prompt") or payload["messages"][0]["content"]
        rows = prompt.count("Routine inspection")
        count = rows * 24
        if endpoint == "tokenize":
            tokenized_rows.append(rows)
            if len(tokenized_rows) == 2 and not fits_reserve:
                # The actual tokenizer can disagree with the small-fixture
                # estimate. Reject before sending a completion in that case.
                count = context_limit - max_tokens - 1024 + 1
            return {"count": count}
        assert fits_reserve, "Over-budget prompt must not reach completion"
        return response(json.dumps(fixture_markers(prompt)), prompt_tokens=count + 32)

    monkeypatch.setattr(acceptance, "http_request", respond)
    runner = acceptance.Qualification(args(tmp_path, long_context=True,
                                          context_targets=[800000], context_limit=context_limit,
                                          max_tokens=max_tokens))
    if fits_reserve:
        stage = runner.documents()["stages"][0]
        assert stage["target_prompt_tokens"] == 800000
        assert 790000 <= stage["actual_prompt_tokens"] <= 810000
        assert endpoints == ["tokenize", "tokenize", "chat/completions"]
    else:
        with pytest.raises(AssertionError, match="output/template reserve exceeds"):
            runner.documents()
        assert endpoints == ["tokenize", "tokenize"]
        assert not runner.requests
    assert tokenized_rows[0] == 64
    assert 12000 < tokenized_rows[1] <= acceptance.MAX_DOCUMENT_ROWS


def test_inaccurate_token_measurements_or_missing_marker_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "http_request", Mock(return_value=response('{}', prompt_tokens=500)))
    case = acceptance.Qualification(args(tmp_path)).run_case("documents")
    assert not case["passed"]
    assert "not retrieved" in case["error"]
    assert case["requests"][0]["document"]["actual_prompt_tokens"] == 500


def test_tokenizer_override_checks_backend_identity_once_and_keeps_completions_on_proxy(tmp_path, monkeypatch):
    captured = []
    def respond(url, endpoint, payload, *unused, **kwargs):
        captured.append((url, endpoint))
        if endpoint == "models":
            assert url == "http://localhost:8001/v1"
            return {"data": [{"id": "glm-5.3-flash"}]}
        prompt = payload.get("prompt") or payload["messages"][0]["content"]
        count = prompt.count("Routine inspection") * 24 + 128
        if endpoint == "tokenize":
            assert url == "http://localhost:8001"
            return {"count": count}
        assert url == "http://localhost:8000/v1"
        return response(json.dumps(fixture_markers(prompt)), prompt_tokens=count + 32)
    monkeypatch.setattr(acceptance, "http_request", respond)
    runner = acceptance.Qualification(args(tmp_path, long_context=True, tokenizer_url="http://localhost:8001"))
    runner.documents()
    assert sum(endpoint == "models" for _, endpoint in captured) == 1
    assert runner.tokenizer_identity["advertised_models"] == ["glm-5.3-flash"]
    assert len(runner.tokenizer_requests) == 4


def test_wrong_tokenizer_model_fails_before_sizing_or_completion(tmp_path, monkeypatch):
    mock = Mock(return_value={"data": [{"id": "other-model"}]})
    monkeypatch.setattr(acceptance, "http_request", mock)
    runner = acceptance.Qualification(args(tmp_path, long_context=True, tokenizer_url="http://localhost:8001"))
    case = runner.run_case("documents")
    assert not case["passed"]
    assert "identity mismatch" in case["error"]
    assert mock.call_count == 1
    assert not runner.requests and not runner.tokenizer_requests


def test_receipt_keeps_failure_evidence_tokens_actual_model_and_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GLM_KEY", "secret-test-token")
    mock = Mock(side_effect=[{"data": [{"id": "glm-5.3-flash"}]}, response("392")])
    monkeypatch.setattr(acceptance, "http_request", mock)
    assert acceptance.Qualification(args(tmp_path, only=["streaming"])).run() == 1
    raw = (tmp_path / "receipt.json").read_text()
    report = json.loads(raw)
    assert not report["passed"]
    assert report["actual_served_models"] == ["glm-5.3-flash"]
    assert report["tool_choice"] == "named"
    assert report["cases"][1]["requests"][0]["usage"]["completion_tokens"] == 32
    assert report["cases"][1]["requests"][0]["content"] == "392"
    assert "secret-test-token" not in raw
    assert "private trace" not in raw


def test_identity_mismatch_stops_completion_spend(tmp_path, monkeypatch):
    mock = Mock(return_value={"data": [{"id": "deepseek-v4-flash"}]})
    monkeypatch.setattr(acceptance, "http_request", mock)
    assert acceptance.Qualification(args(tmp_path)).run() == 1
    assert mock.call_count == 1
    report = json.loads((tmp_path / "receipt.json").read_text())
    assert len(report["cases"]) == 1 and not report["passed"]


@pytest.mark.parametrize("flags", [["--no-thinking"], ["--thinking"],
                                   ["--context-targets", "8192"],
                                   ["--long-context", "--context-targets", "32768", "8192"],
                                   ["--long-context", "--context-targets", "131072"],
                                   ["--url", "http://user:password@localhost:8000/v1"],
                                   ["--url", "http://localhost:8000/v1?api_key=secret"],
                                   ["--tokenizer-url", "http://user:secret@localhost:8001"],
                                   ["--max-tokens", "0"]])
def test_cli_rejects_unsafe_or_inapplicable_configuration(tmp_path, monkeypatch, flags):
    runner = Mock()
    monkeypatch.setattr(acceptance, "Qualification", runner)
    with pytest.raises(SystemExit) as exc:
        acceptance.main(["--output", str(tmp_path / "receipt.json"), *flags])
    assert exc.value.code == 2
    runner.assert_not_called()


def test_cli_defaults_are_bounded_always_thinking_smoke(tmp_path, monkeypatch):
    runner = Mock()
    runner.return_value.run.return_value = 0
    monkeypatch.setattr(acceptance, "Qualification", runner)
    assert acceptance.main(["--output", str(tmp_path / "receipt.json")]) == 0
    options = runner.call_args.args[0]
    assert options.model == "glm-5.3-flash"
    assert options.context_limit == 131072
    assert options.reasoning_effort == "low"
    assert options.tool_choice == "named"
    assert not options.long_context
    assert options.context_targets == [8192, 32768, 98304]
    assert options.tokenizer_url == options.url
