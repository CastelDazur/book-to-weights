import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from b2w import teachers


class _Stub:
    """A tiny OpenAI-compatible chat endpoint that records requests and replays canned answers."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"headers": dict(self.headers), "body": body})
                answer = stub.answers.pop(0) if stub.answers else ""
                data = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(teachers.time, "sleep", lambda s: None)
    made = []

    def make(answers, api_key=""):
        s = _Stub(answers)
        made.append(s)
        monkeypatch.setattr(teachers, "ENDPOINT", s.url)
        monkeypatch.setattr(teachers, "API_KEY", api_key)
        monkeypatch.setitem(teachers.ROLES, "teacher", "model-t")
        return s

    yield make
    for s in made:
        s.close()


def test_chat_sends_the_role_model_and_returns_the_text(stub):
    s = stub(["hello"])
    assert teachers.chat("teacher", "hi") == "hello"
    assert s.requests[0]["body"]["model"] == "model-t"
    assert s.requests[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "Authorization" not in s.requests[0]["headers"]


def test_chat_sends_the_api_key_as_a_bearer_token(stub):
    s = stub(["ok"], api_key="secret")
    teachers.chat("teacher", "hi")
    assert s.requests[0]["headers"]["Authorization"] == "Bearer secret"


def test_chat_retries_after_an_empty_answer(stub):
    s = stub(["", "second try"])
    assert teachers.chat("teacher", "hi") == "second try"
    assert len(s.requests) == 2


def test_chat_fails_clearly_when_the_role_has_no_model(monkeypatch):
    monkeypatch.setitem(teachers.ROLES, "judge", "")
    with pytest.raises(RuntimeError, match="B2W_JUDGE"):
        teachers.chat("judge", "hi")


def test_chat_json_reads_a_fenced_block(stub):
    stub(['Here you go:\n```json\n{"q": "why", "a": 3}\n```'])
    assert teachers.chat_json("teacher", "give json") == {"q": "why", "a": 3}


def test_chat_json_asks_again_after_garbage(stub):
    s = stub(["not json at all", '[1, 2, 3]'])
    assert teachers.chat_json("teacher", "give json") == [1, 2, 3]
    assert "ONLY valid JSON" in s.requests[1]["body"]["messages"][0]["content"]


def test_chat_json_gives_up_after_two_bad_answers(stub):
    stub(["nope", "still nope"])
    with pytest.raises(ValueError):
        teachers.chat_json("teacher", "give json")
