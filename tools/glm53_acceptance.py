#!/usr/bin/env python3
"""Synthetic acceptance checks for the always-thinking GLM-5.3-Flash pilot.

  python3 tools/glm53_acceptance.py --output diagnostics/glm53-smoke.json
  python3 tools/glm53_acceptance.py --tool-choice auto --only tools --output diagnostics/glm53-auto-tools.json
  python3 tools/glm53_acceptance.py --long-context --output diagnostics/glm53-long.json

Uses stdlib only, sends synthetic fixtures, and never executes generated Python
or real tools. All completions stream. Long-context checks use /tokenize to size
progressive documents. For a NIM proxy that blocks it, provide the same model's
backend URL with --tokenizer-url http://sparkone.local:8001. No serving state is changed.
These checks establish bounded correctness and API compatibility, not superiority
over another model or full Hermes end-to-end compatibility.
"""
from __future__ import annotations

import argparse
import ast
import base64
import datetime
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time
import urllib.parse

try:  # Both direct script execution and package import by offline tests.
    from .qwen_acceptance import MAX_LINE, MAX_RESPONSE, ProtocolError, StreamResult, png_fixture, sse_events
except ImportError:
    from qwen_acceptance import MAX_LINE, MAX_RESPONSE, ProtocolError, StreamResult, png_fixture, sse_events


CASES = ("streaming", "tools", "vision", "coding", "documents")
# Supports explicitly requested near-million-token probes while keeping the
# fixture bounded; default smoke and long-context targets remain unchanged.
MAX_DOCUMENT_ROWS = 50000


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_url(base_url):
    endpoint = urllib.parse.urlsplit(base_url)
    if (endpoint.scheme not in ("http", "https") or not endpoint.hostname or
            endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("URL must be an HTTP(S) base URL without credentials, query, or fragment")
    endpoint.port  # Validate malformed ports before writing any receipt.
    return endpoint


class ModelStream(StreamResult):
    """Use server token accounting and require the actual served model identity."""
    def __init__(self, start, expected_model):
        super().__init__(start)
        self.expected_model = expected_model
        self.models = set()

    def feed(self, data, now):
        if data != "[DONE]":
            packet = json.loads(data)
            if packet.get("model"):
                self.models.add(packet["model"])
        super().feed(data, now)

    def finish(self, end):
        result = super().finish(end)
        require(self.models == {self.expected_model},
                f"Stream model identity mismatch: expected {self.expected_model!r}, received {sorted(self.models)!r}")
        result["model"] = self.expected_model
        return result


def http_request(base_url, suffix, payload, timeout, api_key=None, *, stream=False, root=False):
    endpoint = validate_url(base_url)
    # NIM /tokenize is outside /v1; preserve an optional reverse-proxy prefix.
    base_path = endpoint.path.rstrip("/")
    if root and base_path.endswith("/v1"):
        base_path = base_path[:-3]
    path = base_path + "/" + suffix.lstrip("/")
    connection = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    conn = connection(endpoint.hostname, endpoint.port, timeout=min(timeout, 15))
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    started, expired = time.monotonic(), threading.Event()
    transport = timer = None

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
            raise TimeoutError("Connection exhausted request deadline")
        transport.settimeout(remaining)
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        conn.request("POST" if payload is not None else "GET", path,
                     body=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise ProtocolError(f"HTTP {response.status}: " + response.read(8192).decode("utf-8", errors="replace"))
        if not stream:
            raw = response.read(MAX_RESPONSE + 1)
            if expired.is_set():
                raise TimeoutError("Request deadline exceeded")
            require(len(raw) <= MAX_RESPONSE, "JSON response exceeds 16 MiB")
            return json.loads(raw)
        require("text/event-stream" in response.getheader("Content-Type", ""), "Expected SSE response")
        result = ModelStream(started, payload["model"])

        def lines():
            size = 0
            while True:
                line = response.readline(MAX_LINE + 1)
                if expired.is_set():
                    raise TimeoutError("Request deadline exceeded")
                if not line:
                    break
                size += len(line)
                require(size <= MAX_RESPONSE, "Stream exceeds 16 MiB")
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


def validate_clamp(source):
    """Interpret only an intentionally tiny expression grammar; never eval/exec."""
    require(len(source) <= 4096, "Code exceeds 4096 characters")
    tree = ast.parse(source.strip())
    require(sum(1 for _ in ast.walk(tree)) <= 80, "Code AST exceeds 80 nodes")
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), "Expected one function")
    fn = tree.body[0]
    require(fn.name == "clamp" and not fn.decorator_list and fn.returns is None, "Wrong function or decorators/annotations")
    args = fn.args
    require([a.arg for a in args.args] == ["value", "low", "high"] and
            not (args.posonlyargs or args.kwonlyargs or args.defaults or args.kw_defaults or args.vararg or args.kwarg) and
            all(a.annotation is None for a in args.args), "Wrong clamp signature")
    require(len(fn.body) == 1 and isinstance(fn.body[0], ast.Return), "Expected exactly one return expression")

    def interpret(node, values, depth=0):
        require(depth <= 12, "Expression exceeds depth bound")
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                node.func.id in ("min", "max") and len(node.args) == 2 and not node.keywords):
            pair = [interpret(arg, values, depth + 1) for arg in node.args]
            return min(pair) if node.func.id == "min" else max(pair)
        raise AssertionError("Unsupported code: only argument names and two-argument min/max calls are permitted")

    samples = [(-5, 0, 10, 0), (0, 0, 10, 0), (4, 0, 10, 4), (10, 0, 10, 10),
               (99, 0, 10, 10), (-9, -7, -2, -7), (-4, -7, -2, -4), (2, -7, -2, -2), (9, 3, 3, 3)]
    for value, low, high, expected in samples:
        actual = interpret(fn.body[0].value, {"value": value, "low": low, "high": high})
        require(actual == expected, f"Clamp failed ({value}, {low}, {high}): expected {expected}, received {actual}")
    return {"known_output_cases": len(samples), "evaluation": "restricted AST interpretation; no generated code execution"}


def document_fixture(rows, markers=None):
    require(1 <= rows <= MAX_DOCUMENT_ROWS, "Document row count outside bound")
    markers = markers or {key: "REC-" + secrets.token_hex(8).upper() for key in ("early", "middle", "late")}
    positions = dict(zip((rows // 10, rows // 2, rows * 9 // 10), markers))
    require(len(positions) == 3, "Document must allow three separate retrieval positions")
    records = []
    for index in range(rows):
        records.append(f"Record {index:05d}: Routine inspection found stable temperatures and normal network traffic; no action was requested.")
        if index in positions:
            label = positions[index]
            records.append(f"Archive entry {label}: the recovery marker is {markers[label]}.")
    prompt = ("Read this synthetic archive. Retrieve the recovery markers from the early, middle, and late archive entries.\n" +
              "\n".join(records) + "\nReturn only a JSON object with keys early, middle, late and their exact marker strings.")
    return prompt, markers


class Qualification:
    def __init__(self, args):
        self.args = args
        self.requests = []
        self.fixture = png_fixture()
        self.api_key = os.getenv(args.api_key_env)
        self.tokenizer_identity = None
        self.tokenizer_requests = []

    def request(self, name, messages, **options):
        payload = {"model": self.args.model, "messages": messages, "temperature": 0,
                   "reasoning_effort": self.args.reasoning_effort, "max_tokens": self.args.max_tokens,
                   "stream": True, "stream_options": {"include_usage": True}}
        payload.update(options)
        started = time.monotonic()
        receipt = {"name": name, "requested_model": self.args.model}
        self.requests.append(receipt)
        try:
            result = http_request(self.args.url, "chat/completions", payload, self.args.timeout, self.api_key, stream=True)
            receipt.update({key: result[key] for key in ("model", "content", "tool_calls", "finish_reason", "usage", "metrics")})
            receipt["reasoning_characters"] = len(result["reasoning"])
            require(result["usage"]["total_tokens"] <= self.args.context_limit,
                    "Measured prompt plus completion exceeds configured context limit")
            return result
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 4)

    def prompt(self, name, text, **options):
        return self.request(name, [{"role": "user", "content": text}], **options)

    def identity(self):
        models = http_request(self.args.url, "models", None, self.args.timeout, self.api_key)
        ids = [item.get("id") for item in models.get("data", [])]
        require(self.args.model in ids, f"Requested model absent from /models: {ids!r}")
        return {"advertised_models": ids}

    def streaming(self):
        result = self.prompt("streaming", "Calculate 17 times 23. Return only the integer.")
        require(result["content"].strip() == "391", "Expected final answer 391")
        require(not result["tool_calls"], "Unexpected tool call")
        return {"separate_reasoning_observed": bool(result["reasoning"])}

    def tools(self):
        tool = {"type": "function", "function": {"name": "lookup_archive", "description": "Read a synthetic archive fixture.",
                "parameters": {"type": "object", "properties": {"record": {"type": "string", "enum": ["spark-pilot"]}},
                               "required": ["record"], "additionalProperties": False}}}
        messages = [{"role": "user", "content": "Call lookup_archive for record spark-pilot. Then return only the recovery marker supplied by the tool."}]
        # A named call verifies the forced-call contract; auto also exercises
        # the model's tool syntax and the serving runtime's tool parser.
        choice = ("auto" if self.args.tool_choice == "auto" else
                  {"type": "function", "function": {"name": "lookup_archive"}})
        first = self.request("tools-call", messages, tools=[tool], tool_choice=choice)
        calls = first["tool_calls"]
        # The pinned native vLLM serving path intentionally reports "stop" for
        # named calls and "tool_calls" for auto/required calls. Accommodate that
        # named-call convention without claiming API equivalence or relaxing
        # the automatic-call protocol, exact payload, or result-replay checks.
        tool_finishes = {"tool_calls"} if choice == "auto" else {"stop", "tool_calls"}
        require(first["finish_reason"] in tool_finishes and len(calls) == 1, "Expected one completed tool call")
        call = calls[0]
        require(isinstance(call["id"], str) and bool(call["id"]) and call["type"] == "function" and
                call["function"]["name"] == "lookup_archive", "Invalid tool identity/type/name")
        require(json.loads(call["function"]["arguments"]) == {"record": "spark-pilot"}, "Tool arguments violate fixture schema")
        marker = "READY-" + secrets.token_hex(8).upper()
        assistant = {"role": "assistant", "content": first["content"] or None, "tool_calls": calls}
        if first["reasoning"]:
            assistant["reasoning_content"] = first["reasoning"]
        messages += [assistant, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"recovery_marker": marker})}]
        second = self.request("tools-result", messages, tools=[tool], tool_choice="none")
        require(not second["tool_calls"] and second["content"].strip() == marker, "Tool result was not used exactly")
        return {"tool_choice": self.args.tool_choice, "validated_calls": 1,
                "roundtrip_marker": marker, "real_tool_actions_executed": 0}

    def vision(self):
        content = [{"type": "text", "text": "Read the large black text and identify the colors of the square and circle. Return only JSON with keys text, square, circle; use lowercase English color names and preserve the visible text."},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(self.fixture).decode()}}]
        response_format = {"type": "json_schema", "json_schema": {
            "name": "image_observation", "strict": True,
            "schema": {"type": "object", "properties": {
                "text": {"type": "string"}, "square": {"type": "string"},
                "circle": {"type": "string"}},
                "required": ["text", "square", "circle"], "additionalProperties": False}}}
        result = self.request("vision", [{"role": "user", "content": content}],
                              response_format=response_format)
        require(json.loads(result["content"]) == {"text": "SPARK 73", "square": "red", "circle": "blue"}, "Image OCR or shape colors incorrect")
        return {"fixture_sha256": hashlib.sha256(self.fixture).hexdigest(), "input": "locally generated PNG data URI",
                "structured_output": "strict json_schema"}

    def coding(self):
        result = self.prompt("coding", "Write Python function clamp(value, low, high) for inclusive clamping, assuming low <= high. Use exactly one return expression made only of the arguments and two-argument calls to min/max. Return only the function: no Markdown, comments, decorators, annotations, imports, or explanation.")
        return validate_clamp(result["content"])

    def token_count(self, prompt):
        base = self.args.tokenizer_url.rstrip("/")
        if base != self.args.url.rstrip("/") and self.tokenizer_identity is None:
            models_base = base if urllib.parse.urlsplit(base).path.endswith("/v1") else base + "/v1"
            models = http_request(models_base, "models", None, self.args.timeout, self.api_key)
            ids = [item.get("id") for item in models.get("data", [])]
            require(self.args.model in ids, f"Tokenizer backend model identity mismatch: {ids!r}")
            self.tokenizer_identity = {"models_url": models_base + "/models", "advertised_models": ids}
        started = time.monotonic()
        receipt = {"url": base, "requested_model": self.args.model}
        self.tokenizer_requests.append(receipt)
        try:
            result = http_request(base, "tokenize", {"model": self.args.model, "prompt": prompt},
                                  self.args.timeout, self.api_key, root=True)
            count = result.get("count")
            require(type(count) is int and count > 0, "Tokenizer did not return a positive integer count")
            receipt["count"] = count
            return count
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 4)

    def documents(self):
        stages, previous = [], 0
        targets = self.args.context_targets if self.args.long_context else [None]
        for target in targets:
            rows = 64
            prompt, markers = document_fixture(rows)
            if target:
                count = self.token_count(prompt)
                rows = max(64, min(MAX_DOCUMENT_ROWS, int(rows * target / count)))
                prompt, markers = document_fixture(rows, markers)
                count = self.token_count(prompt)
                require(count + self.args.max_tokens + 1024 <= self.args.context_limit,
                        "Document plus output/template reserve exceeds context limit; reduce --context-targets")
            response_format = {"type": "json_schema", "json_schema": {
                "name": "archive_retrieval", "strict": True,
                "schema": {"type": "object", "properties": {
                    key: {"type": "string"} for key in ("early", "middle", "late")},
                    "required": ["early", "middle", "late"], "additionalProperties": False}}}
            result = self.prompt(f"documents-{target or 'smoke'}", prompt,
                                 response_format=response_format)
            measured = result["usage"]["prompt_tokens"]
            stage = {"target_prompt_tokens": target, "actual_prompt_tokens": measured, "rows": rows,
                     "expected_markers": markers, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            self.requests[-1]["document"] = stage
            require(json.loads(result["content"]) == markers, "Document markers were not retrieved exactly")
            require(measured > previous, "Document stages did not increase actual prompt length")
            if target:
                require(target * .9 <= measured <= target * 1.1, "Actual prompt length differs from requested target by more than 10%")
            previous = measured
            stages.append(stage)
        return {"stages": stages, "full_context_window_validated": False}

    def run_case(self, name):
        started, first = time.monotonic(), len(self.requests)
        case = {"name": name}
        try:
            case["checks"] = getattr(self, name)()
            case["passed"] = True
        except Exception as exc:
            case.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        case.update(elapsed_seconds=round(time.monotonic() - started, 4), requests=self.requests[first:])
        print(json.dumps({key: case[key] for key in ("name", "passed", "elapsed_seconds")}), flush=True)
        return case

    def run(self):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cases = [self.run_case("identity")]
        if cases[0]["passed"]:
            cases.extend(self.run_case(name) for name in self.args.only)
        report = {"schema_version": 1, "requested_model": self.args.model,
                  "actual_served_models": sorted({r["model"] for r in self.requests if "model" in r}),
                  "url": self.args.url, "started_at": started,
                  "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "reasoning_mode": "always-thinking", "reasoning_effort": self.args.reasoning_effort,
                  "tool_choice": self.args.tool_choice,
                  "context_limit": self.args.context_limit, "long_context": self.args.long_context,
                  "tokenizer_url": self.args.tokenizer_url, "tokenizer_backend_identity": self.tokenizer_identity,
                  "tokenization_requests": self.tokenizer_requests,
                  "requested_cases": self.args.only, "passed": all(c["passed"] for c in cases), "cases": cases,
                  "limitations": ["Synthetic correctness/API smoke tests; no claim of general model quality or Hermes end-to-end compatibility.",
                                  "Code checked by a bounded min/max AST interpreter on nine cases; no generated code is executed.",
                                  "Server-reported tokens are used; streaming chunks are not token counts.",
                                  "Retrieval verifies three markers at measured lengths, not the entire advertised context window.",
                                  "No subjective scoring, video, concurrency, cancellation, or sustained-load qualification."]}
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{'PASS' if report['passed'] else 'FAIL'}: {output}", flush=True)
        return 0 if report["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://sparkone.local:8000/v1")
    parser.add_argument("--tokenizer-url", help="Same model's tokenizer backend URL; defaults to --url (NIM Spark backend commonly uses port8001)")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--output", required=True, help="JSON receipt file (not a directory)")
    parser.add_argument("--timeout", type=float, default=300, help="Wall-clock seconds per HTTP request")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Includes reasoning and final output")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--tool-choice", choices=("named", "auto"), default="named",
                        help="Named verifies forced calls; auto also exercises automatic tool parsing")
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--long-context", action="store_true", help="Expand retrieval beyond the inexpensive 64-record smoke")
    parser.add_argument("--context-targets", type=int, nargs="+", help="Increasing prompt-token targets (default: 8192 32768 98304); requires --long-context")
    parser.add_argument("--api-key-env", default="SPARK_API_KEY")
    parser.add_argument("--only", choices=CASES, nargs="+", default=list(CASES))
    args = parser.parse_args(argv)
    args.tokenizer_url = args.tokenizer_url or args.url
    try:
        validate_url(args.url)
        validate_url(args.tokenizer_url)
    except ValueError:
        parser.error("url must be HTTP(S) with a valid port and no credentials, query, or fragment")
    if args.context_targets is not None and not args.long_context:
        parser.error("--context-targets requires --long-context")
    if args.context_targets is None:
        args.context_targets = [8192, 32768, 98304]
    if not 1 <= args.timeout <= 3600 or not 128 <= args.max_tokens <= 16384 or not 4096 <= args.context_limit <= 1048576:
        parser.error("timeout must be 1–3600, max-tokens 128–16384, context-limit 4096–1048576")
    if args.max_tokens + 2048 >= args.context_limit:
        parser.error("context-limit must leave at least 2048 prompt tokens beyond max-tokens")
    if args.long_context and (args.context_targets != sorted(set(args.context_targets)) or
                             any(t < 4096 or t + args.max_tokens + 2048 > args.context_limit for t in args.context_targets)):
        parser.error("context-targets must be unique, increasing, >=4096, and fit context-limit with output/template reserve")
    args.only = list(dict.fromkeys(args.only))
    return Qualification(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
