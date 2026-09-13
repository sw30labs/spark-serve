"""Offline checks for stream accounting and completion validation."""
import json
import struct
import zlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.qwen_acceptance import ProtocolError, StreamResult, png_fixture, sse_events
from tools import qwen_acceptance as acceptance


def packet(content=None, reasoning=None, finish=None, tools=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tools is not None:
        delta["tool_calls"] = tools
    result = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        result["usage"] = usage
    return json.dumps(result)


def usage(completion=13):
    return {"prompt_tokens": 100, "completion_tokens": completion, "total_tokens": 100 + completion}


def test_sse_comments_crlf_multiline_and_final_unterminated_event():
    lines = [b": heartbeat\r\n", b"event: message\r\n", b'data: {"choices":\r\n',
             b"data: []}\r\n", b"\r\n", b"data: [DONE]\n"]
    events = list(sse_events(lines))
    assert json.loads(events[0]) == {"choices": []}
    assert events[1] == "[DONE]"


def test_mtp_packed_deltas_use_usage_not_chunk_count_and_distinguish_reasoning_ttft():
    result = StreamResult(100)
    result.feed(packet(reasoning="think carefully"), 101)
    result.feed(packet(content="Several tokens arrive together"), 102)
    result.feed(packet(content=" in one delta.", finish="stop"), 103)
    result.feed(json.dumps({"choices": [], "usage": usage()}), 104)
    result.feed("[DONE]", 105)
    finished = result.finish(110)
    assert finished["usage"]["completion_tokens"] == 13
    assert finished["metrics"]["nonempty_delta_count_not_tokens"] == 3
    assert finished["metrics"]["completion_tokens_per_second_end_to_end"] == 1.3
    assert finished["metrics"]["ttft_seconds"] == 2
    assert finished["metrics"]["first_model_output_seconds"] == 1


def test_tool_name_arguments_are_assembled_across_chunks_and_round_trip_id_preserved():
    result = StreamResult(100)
    result.feed(packet(tools=[{"index": 0, "id": "call-1", "function": {"name": "lookup_", "arguments": '{"cluster":'}}]), 101)
    result.feed(packet(tools=[{"index": 0, "function": {"name": "cluster_status", "arguments": '"spark"}'}}], finish="tool_calls", usage=usage()), 102)
    result.feed("[DONE]", 103)
    finished = result.finish(104)
    tool = finished["tool_calls"][0]
    assert tool["id"] == "call-1"
    assert tool["function"]["name"] == "lookup_cluster_status"
    assert json.loads(tool["function"]["arguments"]) == {"cluster": "spark"}
    assert finished["metrics"]["ttft_seconds"] is None
    assert finished["metrics"]["first_model_output_seconds"] == 1


@pytest.mark.parametrize("finish,token_usage,done,match", [
    ("stop", usage(), False, "without"),
    ("length", usage(), True, "Incomplete"),
    ("stop", None, True, "usage"),
    ("stop", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}, True, "total"),
    ("stop", {"prompt_tokens": 1, "completion_tokens": True, "total_tokens": 2}, True, "Invalid"),
])
def test_incomplete_or_unaccounted_completions_fail(finish, token_usage, done, match):
    result = StreamResult(100)
    result.feed(packet(content="391", finish=finish, usage=token_usage), 101)
    if done:
        result.feed("[DONE]", 102)
    with pytest.raises(ProtocolError, match=match):
        result.finish(103)


def test_in_stream_error_is_not_mistaken_for_success():
    result = StreamResult(100)
    with pytest.raises(ProtocolError, match="out of memory"):
        result.feed(json.dumps({"error": {"message": "out of memory"}}), 101)


def test_generated_fixture_is_valid_rgb_png_with_expected_shapes():
    data = png_fixture()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed = 8, b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        if kind == b"IHDR":
            width, height, depth, color, *_ = struct.unpack(">IIBBBBB", payload)
            assert (width, height, depth, color) == (640, 360, 8, 2)
        elif kind == b"IDAT":
            compressed += payload
        offset += 12 + length
    raw = zlib.decompress(compressed)
    assert len(raw) == 360 * (1 + 640 * 3)
    def pixel(x, y):
        position = y * (1 + 640 * 3) + 1 + x * 3
        return tuple(raw[position:position + 3])
    assert pixel(140, 110) == (230, 25, 25)
    assert pixel(430, 110) == (20, 70, 235)
    assert pixel(0, 0) == (255, 255, 255)


@pytest.mark.parametrize("thinking,budget", [(None, 512), (True, 512), (False, 128)])
def test_thinking_modes_preserve_server_default_and_allow_reasoning_for_retrieval(tmp_path, monkeypatch, thinking, budget):
    monkeypatch.setattr(acceptance.secrets, "token_hex", lambda size: "ab" * 12)
    result = {"content": "SPARK-" + "AB" * 12, "usage": {"prompt_tokens": 8351}}
    request = Mock(return_value=result)
    monkeypatch.setattr(acceptance, "stream_request", request)
    runner = acceptance.Qualification(SimpleNamespace(
        output=str(tmp_path), model="qwen-test", url="http://localhost:8000/v1",
        timeout=240, api_key_env="SPARK_API_KEY", thinking=thinking,
        context_rows=1, min_context_tokens=6000,
    ))
    runner.retrieval()
    payload = request.call_args.args[1]
    assert payload["max_tokens"] == budget
    if thinking is None:
        assert "chat_template_kwargs" not in payload
    else:
        assert payload["chat_template_kwargs"] == {"enable_thinking": thinking}


@pytest.mark.parametrize("flags,expected", [([], None), (["--thinking"], True), (["--no-thinking"], False)])
def test_cli_thinking_default_and_overrides(monkeypatch, flags, expected):
    qualification = Mock()
    qualification.return_value.run.return_value = 0
    monkeypatch.setattr(acceptance, "Qualification", qualification)
    monkeypatch.setattr("sys.argv", ["qwen_acceptance.py", "--model", "qwen-test", *flags])
    assert acceptance.main() == 0
    assert qualification.call_args.args[0].thinking is expected


def test_cli_rejects_conflicting_thinking_flags(monkeypatch):
    qualification = Mock()
    monkeypatch.setattr(acceptance, "Qualification", qualification)
    monkeypatch.setattr("sys.argv", ["qwen_acceptance.py", "--model", "qwen-test", "--thinking", "--no-thinking"])
    with pytest.raises(SystemExit) as error:
        acceptance.main()
    assert error.value.code == 2
    qualification.assert_not_called()
