#!/usr/bin/env python3
"""Bounded OpenAI-compatible model qualification, using only Python's stdlib.

Example:
  python3 tools/qwen_acceptance.py --url http://sparkone.local:8000/v1 \
      --model qwen3.8-flash-next --output diagnostics/qwen-acceptance

This sends synthetic test data only. The tool round trip executes no model-supplied
code or external actions. It does not start, stop, or modify serving processes.
Thinking follows the server's default unless --thinking or --no-thinking is set.
Server-reported token usage is authoritative: a streamed delta may contain several
tokens, particularly with multi-token prediction. Throughput includes prefill;
the separately labelled post-first-output estimate includes the first delta's
tokens, whose count is not available from the streaming protocol.
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import datetime
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import threading
import time
import urllib.parse
import zlib


MAX_LINE = 1024 * 1024
MAX_RESPONSE = 16 * 1024 * 1024
CASES = ("text", "json", "tools", "vision", "summary", "coding", "reasoning", "retrieval")


class ProtocolError(RuntimeError):
    pass


def sse_events(lines):
    """Read SSE events, including comments, CRLF, and multiline data fields."""
    data = []
    for raw in lines:
        if len(raw) > MAX_LINE:
            raise ProtocolError("SSE line exceeds the response bound")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


class StreamResult:
    def __init__(self, start):
        self.start = start
        self.content = ""
        self.reasoning = ""
        self.tools = {}
        self.usage = None
        self.finish_reason = None
        self.first_content = None
        self.first_output = None
        self.delta_count = 0
        self.done = False

    def feed(self, data, now):
        if data == "[DONE]":
            self.done = True
            return
        packet = json.loads(data)
        if packet.get("error"):
            raise ProtocolError("Server stream error: " + json.dumps(packet["error"]))
        if packet.get("usage") is not None:
            self.usage = packet["usage"]
        for choice in packet.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ProtocolError("Unexpected multiple completion choices")
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            tool_deltas = delta.get("tool_calls") or []
            if content or reasoning or tool_deltas:
                self.delta_count += 1
                if self.first_output is None:
                    self.first_output = now
            if content and self.first_content is None:
                self.first_content = now
            self.content += content
            self.reasoning += reasoning
            for item in tool_deltas:
                index = item.get("index", 0)
                tool = self.tools.setdefault(index, {"id": "", "type": "function",
                                                     "function": {"name": "", "arguments": ""}})
                if item.get("id"):
                    tool["id"] += item["id"]
                function = item.get("function", {})
                for field in ("name", "arguments"):
                    tool["function"][field] += function.get(field) or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def finish(self, end):
        if not self.done:
            raise ProtocolError("Stream ended without [DONE]")
        if self.finish_reason not in ("stop", "tool_calls"):
            raise ProtocolError("Incomplete completion: finish_reason=" + repr(self.finish_reason))
        if not self.usage:
            raise ProtocolError("Server did not return requested token usage")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = self.usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError("Invalid token usage: " + name)
        if self.usage["total_tokens"] != self.usage["prompt_tokens"] + self.usage["completion_tokens"]:
            raise ProtocolError("Token usage total does not equal prompt plus completion")
        elapsed = end - self.start
        output_tokens = self.usage["completion_tokens"]
        decode = end - self.first_output if self.first_output is not None else None
        return {
            "content": self.content, "reasoning": self.reasoning,
            "tool_calls": [self.tools[i] for i in sorted(self.tools)],
            "finish_reason": self.finish_reason, "usage": self.usage,
            "metrics": {
                "elapsed_seconds": round(elapsed, 4),
                "ttft_seconds": round(self.first_content - self.start, 4) if self.first_content is not None else None,
                "first_model_output_seconds": round(self.first_output - self.start, 4) if self.first_output is not None else None,
                "completion_tokens_per_second_end_to_end": round(output_tokens / elapsed, 3) if elapsed else None,
                "post_first_output_tokens_per_second_estimate": round(output_tokens / decode, 3) if decode and decode > 0.01 else None,
                "nonempty_delta_count_not_tokens": self.delta_count,
            },
        }


def stream_request(url, payload, timeout, api_key=None):
    endpoint = urllib.parse.urlsplit(url.rstrip("/") + "/chat/completions")
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname or endpoint.username:
        raise ValueError("URL must be an HTTP(S) base URL without embedded credentials")
    connection_cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    conn = connection_cls(endpoint.hostname, endpoint.port, timeout=min(timeout, 15))
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    started = time.monotonic()
    result = StreamResult(started)
    transport = None
    timer = None
    expired = threading.Event()

    def expire():
        expired.set()
        if transport:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    try:
        conn.connect()
        transport = conn.sock
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Connection exhausted the request deadline")
        transport.settimeout(remaining)
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        path = urllib.parse.urlunsplit(("", "", endpoint.path, endpoint.query, ""))
        conn.request("POST", path, body=json.dumps(payload).encode(), headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            body = response.read(8192).decode("utf-8", errors="replace")
            raise ProtocolError(f"HTTP {response.status}: {body}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise ProtocolError("Expected an SSE response")

        def lines():
            size = 0
            while True:
                line = response.readline(MAX_LINE + 1)
                if expired.is_set():
                    raise TimeoutError(f"Request exceeded {timeout:g} seconds")
                if not line:
                    break
                size += len(line)
                if size > MAX_RESPONSE:
                    raise ProtocolError("Response exceeds 16 MiB bound")
                yield line

        for event in sse_events(lines()):
            result.feed(event, time.monotonic())
            if result.done:
                break
        return result.finish(time.monotonic())
    except (OSError, http.client.HTTPException) as exc:
        if expired.is_set():
            raise TimeoutError(f"Request exceeded {timeout:g} seconds") from exc
        raise
    finally:
        if timer:
            timer.cancel()
        conn.close()


def png_fixture():
    """Create an original RGB PNG: red square, blue circle, large SPARK 73 text."""
    width, height = 640, 360
    pixels = bytearray(b"\xff\xff\xff" * width * height)

    def pixel(x, y, color):
        offset = (y * width + x) * 3
        pixels[offset:offset + 3] = bytes(color)

    for y in range(40, 180):
        for x in range(70, 210):
            pixel(x, y, (230, 25, 25))
    for y in range(35, 185):
        for x in range(355, 505):
            if (x - 430) ** 2 + (y - 110) ** 2 <= 70 ** 2:
                pixel(x, y, (20, 70, 235))
    glyphs = {
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    }
    scale = 10
    for i, char in enumerate("SPARK 73"):
        for y, row in enumerate(glyphs.get(char, [])):
            for x, bit in enumerate(row):
                if bit == "1":
                    for yy in range(scale):
                        for xx in range(scale):
                            pixel(75 + i * 60 + x * scale + xx, 240 + y * scale + yy, (0, 0, 0))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    raw = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Qualification:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fixture = png_fixture()
        (self.directory / "vision-fixture.png").write_bytes(self.fixture)
        self.lock = threading.Lock()

    def request(self, name, messages, **options):
        payload = {
            "model": self.args.model, "messages": messages, "temperature": 0,
            "max_tokens": 512, "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.args.thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.args.thinking}
        payload.update(options)
        # These are synthetic fixtures. Never record the API key or request headers.
        (self.directory / (name + ".request.json")).write_text(json.dumps(payload, indent=2) + "\n")
        result = stream_request(self.args.url, payload, self.args.timeout, os.getenv(self.args.api_key_env))
        (self.directory / (name + ".response.json")).write_text(json.dumps(result, indent=2) + "\n")
        return result

    def prompt(self, name, prompt, **options):
        return self.request(name, [{"role": "user", "content": prompt}], **options)

    def text(self, name="text"):
        result = self.prompt(name, "Calculate 17 multiplied by 23. Reply with only the integer, no explanation.", max_tokens=128)
        require(result["content"].strip() == "391", "Expected exactly 391")
        return [result]

    def json(self):
        schema = {"type": "object", "properties": {"ready": {"type": "boolean"}, "workers": {"type": "integer"}},
                  "required": ["ready", "workers"], "additionalProperties": False}
        result = self.prompt("json", "Return an object indicating ready is true and workers is 2.",
                             response_format={"type": "json_schema", "json_schema": {"name": "status", "strict": True, "schema": schema}})
        value = json.loads(result["content"])
        require(value == {"ready": True, "workers": 2} and value["ready"] is True and type(value["workers"]) is int, "Wrong structured JSON result")
        return [result]

    def tools(self):
        tool = {"type": "function", "function": {
            "name": "lookup_cluster_status", "description": "Read a simulated cluster status fixture.",
            "parameters": {"type": "object", "properties": {"cluster": {"type": "string", "enum": ["spark"]}},
                           "required": ["cluster"], "additionalProperties": False}}}
        messages = [{"role": "user", "content": "Use lookup_cluster_status for cluster spark. Then reply only READY:<ready_workers>, using the integer from the tool result."}]
        first = self.request("tools-call", messages, tools=[tool], tool_choice="auto")
        calls = first["tool_calls"]
        require(first["finish_reason"] == "tool_calls" and len(calls) == 1, "Expected one completed tool call")
        call = calls[0]
        require(bool(call["id"]) and call["function"]["name"] == "lookup_cluster_status", "Wrong tool name or missing ID")
        require(json.loads(call["function"]["arguments"]) == {"cluster": "spark"}, "Wrong tool arguments")
        messages += [{"role": "assistant", "content": first["content"] or None, "tool_calls": calls},
                     {"role": "tool", "tool_call_id": call["id"], "content": '{"cluster":"spark","ready_workers":2}'}]
        second = self.request("tools-result", messages, tools=[tool], tool_choice="none")
        require(second["content"].strip() == "READY:2", "Tool result was not used correctly")
        return [first, second]

    def vision(self):
        content = [{"type": "text", "text": "Read the large black text and identify the colors of the square and circle. Reply only a JSON object with string keys text, square, circle. Use lowercase English color names; preserve the text as shown."},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(self.fixture).decode()}}]
        result = self.request("vision", [{"role": "user", "content": content}], response_format={"type": "json_object"})
        require(json.loads(result["content"]) == {"text": "SPARK 73", "square": "red", "circle": "blue"}, "Vision colors or OCR did not match the fixture")
        return [result]

    def summary(self):
        result = self.prompt("summary", "Summarize this incident in one sentence, preserving the cause and recovery time: At 09:00 the API stopped responding because its certificate had expired. The team renewed the certificate and service recovered at 09:12. No records were lost.")
        text = result["content"].lower()
        require("certificate" in text and ("expired" in text or "expiration" in text) and ("09:12" in text or "12 minutes" in text or "12-minute" in text), "Summary omitted the cause or recovery time")
        return [result]

    def coding(self):
        result = self.prompt("coding", "Write a Python function clamp(value, low, high) that returns value clamped to the inclusive interval [low, high]. Return only the function, without Markdown, imports, tests, or explanation.")
        tree = ast.parse(result["content"].strip())
        require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef) and tree.body[0].name == "clamp", "Expected one Python function named clamp")
        require([a.arg for a in tree.body[0].args.args] == ["value", "low", "high"], "Wrong function signature")
        require(any(isinstance(node, ast.Return) for node in ast.walk(tree)), "Function has no return")
        # Syntax/shape smoke only. Model-generated code is deliberately never executed.
        return [result]

    def reasoning(self):
        result = self.prompt("reasoning", "A machine makes 24 parts per minute for 7 minutes, then 18 parts per minute for 2 minutes. How many parts does it make in total? Reply with only the integer.")
        require(result["content"].strip() == "204", "Expected 204 parts")
        return [result]

    def retrieval(self):
        marker = "SPARK-" + secrets.token_hex(12).upper()
        # About 8K tokens by default; the server's actual input count is verified.
        rows = [f"Record {i:04d}: Routine inspection found stable temperatures and normal network traffic. No action was requested."
                for i in range(self.args.context_rows)]
        position = len(rows) // 2
        rows.insert(position, "Record special: The unique recovery marker is " + marker + ".")
        prompt = "Read these records and remember the unique recovery marker.\n\n" + "\n".join(rows) + "\n\nReply with only the unique recovery marker from Record special."
        result = self.prompt("retrieval", prompt, max_tokens=128 if self.args.thinking is False else 512)
        (self.directory / "retrieval-fixture.json").write_text(json.dumps({"expected": marker, "position": position, "rows": len(rows), "characters": len(prompt)}, indent=2) + "\n")
        require(result["usage"]["prompt_tokens"] >= self.args.min_context_tokens,
                f"Retrieval request had fewer than {self.args.min_context_tokens} actual input tokens")
        require(result["content"].strip() == marker, "Long-context marker was not retrieved exactly")
        return [result]

    def run_case(self, name, function=None):
        started = time.monotonic()
        try:
            responses = (function or getattr(self, name))()
            case = {"name": name, "passed": True, "requests": [{"usage": r["usage"], "metrics": r["metrics"]} for r in responses]}
        except Exception as exc:
            case = {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        case["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with self.lock:
            print(json.dumps(case), flush=True)
        return case

    def run(self):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cases = [self.run_case(name) for name in self.args.only]
        if self.args.concurrency:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.run_case, f"concurrency-{i}", lambda i=i: self.text(f"concurrency-{i}")) for i in range(2)]
                cases.extend(f.result() for f in futures)
        report = {
            "model": self.args.model, "url": self.args.url, "started_at": started,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "thinking_enabled": self.args.thinking, "passed": all(c["passed"] for c in cases), "cases": cases,
            "thinking_mode": "server_default" if self.args.thinking is None else ("enabled" if self.args.thinking else "disabled"),
            "limitations": ["Synthetic smoke tests, not a general model quality evaluation.",
                            "Coding checks syntax and signature; generated code is never executed.",
                            "Token counts are server-reported; streamed delta counts are not token counts.",
                            "Throughput includes prefill. The post-first-output estimate includes the first delta's unknown token count.",
                            "Retrieval uses the measured prompt length; it does not validate the full advertised context window."],
        }
        (self.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{'PASS' if report['passed'] else 'FAIL'}: {sum(c['passed'] for c in cases)}/{len(cases)} cases; report: {self.directory / 'report.json'}", flush=True)
        return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://sparkone.local:8000/v1", help="OpenAI API base URL including /v1")
    parser.add_argument("--model", required=True, help="Exact served model ID")
    parser.add_argument("--output", default="diagnostics/qwen-acceptance-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout", type=float, default=240, help="Wall-clock deadline per request, seconds")
    parser.add_argument("--api-key-env", default="SPARK_API_KEY", help="Environment variable holding an optional Spark API key")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", action="store_true", dest="thinking", help="Explicitly enable thinking")
    thinking.add_argument("--no-thinking", action="store_false", dest="thinking", help="Explicitly disable thinking for fast-mode qualification")
    parser.set_defaults(thinking=None)
    parser.add_argument("--context-rows", type=int, default=360, help="Synthetic retrieval record count, about 8K tokens by default")
    parser.add_argument("--min-context-tokens", type=int, default=6000, help="Minimum actual server-reported retrieval input tokens")
    parser.add_argument("--concurrency", action="store_true", help="Also run two simultaneous short requests")
    parser.add_argument("--only", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800 or not 1 <= args.context_rows <= 10000 or args.min_context_tokens < 1:
        parser.error("timeout must be 1–1800, context-rows 1–10000, and min-context-tokens positive")
    return Qualification(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
