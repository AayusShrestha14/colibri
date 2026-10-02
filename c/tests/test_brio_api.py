"""The three forms of POST /v1/brio, driven through the gateway with a scoring
engine we control.

`options` is one closed question. `questions` is many questions on one state,
each with its own options, and the state must be photographed ONCE: that is
where the measured 5.7x lives, and a handler that re-reads the state per
question would still answer correctly while throwing the saving away, so the
pin order is asserted, not just the answers. `schema` fills a JSON object one
field at a time; the skeleton is data the server writes, so the field after
must see the value chosen for the field before, and nothing is ever generated.

The fake engine tokenises on whitespace and scores an option by a table, so
every answer here is chosen by the test, not by chance. It records every
call: prompt, max_tokens and pin, which is what the assertions read.

MimoBrioEndToEnd puts the real MiMo engine behind the same server, on the tiny
fixture (`make mimo mimo-tiny-generate`; skipped without them): the endpoint
reaches it, every option after the photo reads only its own tokens, the scores
equal a cold engine's, and chat logprobs pass the gateway's gate.
"""
import json
import math
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimo_serve_fixture  # noqa: E402
import openai_server  # noqa: E402
from family_registry import family_by_id  # noqa: E402
from openai_server import APIServer  # noqa: E402

STATE = "340 lines across 8 files, no tests. CI is green but nothing covers that path."


class ScoringEngine:
    """Scores like the real channel, deterministically.

    `table` maps an option string to the mean log-probability its tokens get;
    anything not in the table scores -5.0. A pinned call answers with ACCEPT
    (prompt_tokens) and one ECHO per position, like an engine with the channel."""

    def __init__(self, table):
        self.table = table
        self.calls = []
        self.kv_slots = 1

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, audio=None,
                 on_tool=None, image=None, logprobs=0, pin=False, on_echo=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "pin": bool(pin),
                           "logprobs": logprobs})
        words = prompt.split()
        if on_accept:
            on_accept({"prompt_tokens": len(words)})
        if on_echo and logprobs:
            # The option is whatever follows the last "Answer:" or the last
            # opening quote of a skeleton cell; score its words from the table.
            option = None
            for key in self.table:
                if prompt.endswith(" " + key):
                    option = key
            per_token = self.table.get(option, -5.0)
            for pos, _word in enumerate(words):
                on_echo({"pos": pos, "logprob": per_token if option and pos >= len(words) - len(option.split()) else -1.0})

    def close(self):
        pass


class BrioApi(unittest.TestCase):
    def serve(self, table):
        self.engine = ScoringEngine(table)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        import threading
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, body):
        request = Request(self.base + "/v1/brio", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def post_error(self, body):
        try:
            self.post(body)
        except HTTPError as error:
            return error.code, json.loads(error.read())
        self.fail("expected an error")

    def pins(self):
        return [c["prompt"] for c in self.engine.calls if c["pin"]]

    # ---- options: the single form is unchanged ------------------------------
    def test_single_form_still_answers_and_never_generates(self):
        self.serve({"merge": -3.0, "request changes": -0.2, "close": -4.0})
        out = self.post({"model": "test-model", "state": STATE,
                         "question": "What should the reviewer do?",
                         "options": ["merge", "request changes", "close"]})
        self.assertEqual(out["object"], "brio.choice")
        self.assertEqual(out["answer"], "request changes")
        self.assertEqual(out["usage"]["completion_tokens"], 0)
        self.assertTrue(all(c["max_tokens"] == 0 for c in self.engine.calls))
        # Two photographs and no more: the shared state on its own, then the
        # whole prefix. The options are never pinned. The state photograph is
        # the return point that lets a later question on the same document
        # reuse the snapshot instead of re-reading it (asserted below).
        pins = self.pins()
        self.assertEqual(len(pins), 2, pins)
        self.assertEqual(pins[0], f"Context:\n{STATE}\n\n")
        self.assertTrue(pins[1].startswith(pins[0]))
        self.assertTrue(pins[1].endswith("Answer:"))

    def test_options_form_pins_the_shared_state_for_reuse_across_requests(self):
        # The web asks one question per request on the same document. The state
        # must be photographed on its own each time, so the engine has a strict
        # prefix to restore and every question after the first pays only its own
        # tokens instead of re-reading the whole document -- the "read once" the
        # mode exists for. A handler that folded the state into the question
        # prefix only would still answer correctly while re-reading it, so the
        # pins are asserted, not just the answers.
        self.serve({"merge": -3.0, "request changes": -0.2, "close": -4.0,
                    "yes": -0.1, "no": -2.0})
        self.post({"model": "test-model", "state": STATE,
                   "question": "What should the reviewer do?",
                   "options": ["merge", "request changes", "close"]})
        self.assertEqual([p for p in self.pins() if p == f"Context:\n{STATE}\n\n"],
                         [f"Context:\n{STATE}\n\n"])
        self.post({"model": "test-model", "state": STATE,
                   "question": "Does it need tests?", "options": ["yes", "no"]})
        # both requests photographed the same shared state prefix: the return
        # point exists for the second question, not only the first
        state_pins = [p for p in self.pins() if p == f"Context:\n{STATE}\n\n"]
        self.assertEqual(len(state_pins), 2, self.pins())
        # and every question prefix extends that shared state
        question_pins = [p for p in self.pins() if p.endswith("Answer:")]
        self.assertEqual(len(question_pins), 2, self.pins())
        for pin in question_pins:
            self.assertTrue(pin.startswith(f"Context:\n{STATE}\n\n"))

    # ---- questions: many on one state ----------------------------------------
    def test_questions_share_one_state_photograph(self):
        self.serve({"merge": -3.0, "request changes": -0.2, "close": -4.0,
                    "yes": -0.1, "no": -2.0, "high": -0.3, "low": -1.5})
        out = self.post({"model": "test-model", "state": STATE, "questions": [
            {"question": "What should the reviewer do?",
             "options": ["merge", "request changes", "close"]},
            {"question": "Does it need tests?", "options": ["yes", "no"]},
            {"question": "How risky is it?", "options": ["high", "low"], "normalize": "sum"},
        ]})
        self.assertEqual(out["object"], "brio.answers")
        self.assertEqual([a["answer"] for a in out["answers"]],
                         ["request changes", "yes", "high"])
        self.assertEqual([a["normalize"] for a in out["answers"]], ["sum", "sum", "sum"])
        for answer in out["answers"]:
            self.assertAlmostEqual(sum(c["p"] for c in answer["choices"]), 1.0, places=6)
            self.assertGreaterEqual(answer["entropy"], 0.0)
            self.assertLessEqual(answer["entropy"], 1.0)
        # The state is photographed exactly once, first, on its own; then each
        # question once. Options are never pinned. This is the two-level order
        # that makes the second question cost only its own words.
        pins = self.pins()
        self.assertEqual(len(pins), 4, pins)
        self.assertEqual(pins[0], f"Context:\n{STATE}\n\n")
        for pin, entry in zip(pins[1:], out["answers"]):
            self.assertTrue(pin.startswith(pins[0]), "a question prefix must extend the state")
            self.assertTrue(pin.endswith(f"Question: {entry['question']}\nAnswer:"))
        self.assertEqual(out["usage"]["completion_tokens"], 0)
        self.assertEqual(out["usage"]["read_tokens"],
                         sum(c["tokens"] for a in out["answers"] for c in a["choices"]))

    def test_default_normalize_is_sum(self):
        # The default was "mean", whose per-token averages silently favor
        # multi-token options on any menu that mixes token counts; "sum" is
        # the joint log-probability and the only safe default.
        self.serve({"merge": -3.0, "close": -4.0})
        out = self.post({"model": "test-model", "state": STATE,
                         "question": "What should the reviewer do?",
                         "options": ["merge", "close"]})
        self.assertEqual(out["normalize"], "sum")

    def test_mean_with_unequal_option_token_counts_warns(self):
        import contextlib
        import io
        self.serve({"merge": -3.0, "request changes": -0.2})
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.post({"model": "test-model", "state": STATE,
                       "question": "What should the reviewer do?",
                       "options": ["merge", "request changes"],
                       "normalize": "mean"})
        self.assertIn("unequal option token counts", stderr.getvalue())
        self.assertIn("merge=1", stderr.getvalue())
        # Equal token counts stay quiet — mean is valid there. (The [api]
        # request log also lands on stderr; only the bias warning matters.)
        self.serve({"yes": -0.1, "no": -2.0})
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.post({"model": "test-model", "state": STATE,
                       "question": "Ship it?", "options": ["yes", "no"],
                       "normalize": "mean"})
        self.assertNotIn("unequal option token counts", stderr.getvalue())

    # ---- schema: a JSON object filled one cell at a time --------------------
    def test_schema_fills_cells_in_order_and_each_cell_sees_the_ones_before(self):
        self.serve({"merge": -3.0, "request changes": -0.2, "close": -4.0,
                    "yes": -0.1, "no": -2.0, "engine": -0.05, "docs": -3.0})
        out = self.post({"model": "test-model", "state": STATE,
                         "task": "Review this pull request.",
                         "schema": {"decision": ["merge", "request changes", "close"],
                                    "needs_tests": ["yes", "no"],
                                    "area": ["engine", "docs"]}})
        self.assertEqual(out["object"], "brio.schema")
        self.assertEqual(out["json"], {"decision": "request changes",
                                       "needs_tests": "yes", "area": "engine"})
        self.assertEqual([f["field"] for f in out["fields"]],
                         ["decision", "needs_tests", "area"])
        # the JSON is valid by construction and round-trips
        self.assertEqual(json.loads(json.dumps(out["json"])), out["json"])
        pins = self.pins()
        self.assertEqual(len(pins), 4, pins)             # state, then one per cell
        self.assertTrue(pins[1].endswith('Task: Review this pull request.\n{"decision": "'))
        # the second cell's skeleton carries the first cell's chosen value
        self.assertTrue(pins[2].endswith('{"decision": "request changes", "needs_tests": "'))
        self.assertTrue(pins[3].endswith(
            '{"decision": "request changes", "needs_tests": "yes", "area": "'))
        self.assertEqual(out["usage"]["completion_tokens"], 0)
        self.assertTrue(all(c["max_tokens"] == 0 for c in self.engine.calls))
        for field in out["fields"]:
            self.assertIn("p", field)
            self.assertIn("entropy", field)

    def test_schema_value_with_a_quote_is_written_as_valid_json(self):
        self.serve({'say "hi"': -0.1, "stay quiet": -3.0})
        out = self.post({"model": "test-model", "state": STATE,
                         "schema": {"reply": ['say "hi"', "stay quiet"],
                                    "again": ['say "hi"', "stay quiet"]}})
        self.assertEqual(out["json"]["reply"], 'say "hi"')
        # the skeleton for the second cell must escape the first cell's value,
        # or the JSON the server claims to build would not parse
        self.assertIn('"reply": "say \\"hi\\"", "again": "', self.pins()[2])

    # ---- validation ----------------------------------------------------------
    def test_exactly_one_form(self):
        self.serve({})
        code, body = self.post_error({"model": "test-model", "state": STATE,
                                      "options": ["a", "b"], "questions": []})
        self.assertEqual(code, 400)
        self.assertIn("exactly one", body["error"]["message"])
        code, _ = self.post_error({"model": "test-model", "state": STATE})
        self.assertEqual(code, 400)

    def test_every_option_set_needs_two_entries(self):
        self.serve({})
        code, body = self.post_error({"model": "test-model", "state": STATE,
                                      "schema": {"decision": ["merge"]}})
        self.assertEqual(code, 400)
        self.assertIn("schema.decision", body["error"]["message"])
        code, body = self.post_error({"model": "test-model", "state": STATE,
                                      "questions": [{"question": "q?", "options": ["only"]}]})
        self.assertEqual(code, 400)
        self.assertIn("questions[0].options", body["error"]["message"])

    def test_questions_and_schema_need_a_state(self):
        self.serve({})
        code, body = self.post_error({"model": "test-model",
                                      "questions": [{"question": "q?", "options": ["a", "b"]}]})
        self.assertEqual(code, 400)
        self.assertIn("state", body["error"]["message"])

    def test_non_string_message_text_is_a_client_error(self):
        self.serve({})
        code, body = self.post_error({
            "model": "test-model", "options": ["yes", "no"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": 7}]}],
        })
        self.assertEqual(code, 400)
        self.assertEqual(body["error"]["param"], "messages.0.content.0.text")
        self.assertEqual(self.engine.calls, [])
        out = self.post({
            "model": "test-model", "options": ["yes", "no"],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Ship it?"}]}],
        })
        self.assertEqual(out["object"], "brio.choice")
        self.assertEqual(self.pins()[0], "Context:\nuser: Ship it?\n\n")

    def test_schema_field_names_that_would_break_the_skeleton_are_refused(self):
        self.serve({})
        code, body = self.post_error({"model": "test-model", "state": STATE,
                                      "schema": {'bad"name': ["a", "b"]}})
        self.assertEqual(code, 400)
        self.assertIn("quotes", body["error"]["message"])


class _Recorder:
    """The real engine, with every call it is given written down: the prompt, the
    pin, the ACCEPT count and the ECHO positions -- what the saving is read from."""

    def __init__(self, engine):
        self.engine = engine
        self.kv_slots = engine.kv_slots
        self.calls = []

    def generate(self, prompt, *args, on_echo=None, on_accept=None, pin=False, **kwargs):
        call = {"prompt": prompt, "pin": bool(pin), "echo": [], "accept": None}
        self.calls.append(call)

        def echo(record):
            call["echo"].append(record["pos"])
            if on_echo:
                on_echo(record)

        def accept(value):
            call["accept"] = value.get("prompt_tokens")
            if on_accept:
                on_accept(value)

        return self.engine.generate(prompt, *args, on_echo=echo, on_accept=accept,
                                    pin=pin, **kwargs)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def _cold_option_score(prefix, option):
    """What one option scores on a fresh engine that reads the whole prompt: the sum
    of the ECHO log-probabilities past the prefix, from the frames themselves."""
    env = dict(os.environ, **mimo_serve_fixture.engine_env())
    p = subprocess.Popen([str(mimo_serve_fixture.ENGINE), "8"], env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    try:
        while b"READY" not in p.stdout.readline():
            pass
        n_prefix, values = None, []
        for rid, text in ((1, prefix), (2, prefix + " " + option)):
            data = text.encode()
            p.stdin.write(f"SUBMIT {rid} 0 {len(data)} 0 0 1 logprobs=1\n".encode()
                          + data + b"\n")
            p.stdin.flush()
            while True:
                fields = p.stdout.readline().split()
                if fields[0] == b"ECHO":
                    p.stdout.read(int(fields[2]) + 1)
                    if rid == 2 and int(fields[3]) >= n_prefix and fields[4] != b"nan":
                        values.append(float(fields[4]))
                elif fields[0] == b"ACCEPT" and rid == 1:
                    n_prefix = int(fields[2])
                elif fields[0] in (b"DONE", b"ERROR"):
                    break
        # sum(), as the endpoint adds them: on Python 3.12 it is compensated, and a
        # plain running total differs from it in the last bits
        return sum(values), len(values)
    finally:
        p.stdin.close()
        p.wait(timeout=30)
        p.stdout.close()


@unittest.skipUnless(mimo_serve_fixture.available(),
                     "mimo is not built or the tiny MiMo fixture is absent "
                     "(make mimo mimo-tiny-generate)")
class MimoBrioEndToEnd(unittest.TestCase):
    STATE = ("The release is late, the tests are red and the changelog is empty. "
             "Two reviewers approved it last week.")
    OPTIONS = ["ship it", "wait", "wait until the tests pass and the changelog is written"]

    def setUp(self):
        engine = openai_server.Engine(
            mimo_serve_fixture.ENGINE, mimo_serve_fixture.served_fixture(), cap=8,
            env=dict(os.environ, **mimo_serve_fixture.engine_env()),
            family=family_by_id("mimo"))
        self.addCleanup(engine.process.stdout.close)
        self.addCleanup(engine.close)
        self.engine = _Recorder(engine)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "mimo-tiny")
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def test_options_read_only_their_own_tokens_and_score_like_a_cold_engine(self):
        out = self.post("/v1/brio", {"model": "mimo-tiny", "state": self.STATE,
                                     "question": "What should we do?",
                                     "options": self.OPTIONS})
        self.assertEqual(out["object"], "brio.choice")
        self.assertIn(out["answer"], self.OPTIONS)
        self.assertEqual(out["usage"]["completion_tokens"], 0)
        self.assertAlmostEqual(sum(c["p"] for c in out["choices"]), 1.0, places=9)

        calls = self.engine.calls
        state, prefix, options = calls[0], calls[1], calls[2:]
        self.assertTrue(state["pin"] and prefix["pin"])
        self.assertEqual(len(options), len(self.OPTIONS))
        # The question resumes from the state's photo, every option from the
        # question's: each read-out starts exactly where the photo ends and covers
        # nothing before it. A fallback to a full recompute would read from 0.
        self.assertEqual(min(prefix["echo"]), state["accept"])
        for call in options:
            self.assertEqual(sorted(call["echo"]),
                             list(range(prefix["accept"], call["accept"])),
                             f"{call['prompt']!r} did not resume from the photo")
        prompt = prefix["prompt"]
        by_option = {c["option"]: c for c in out["choices"]}
        for option in self.OPTIONS:
            total, rows = _cold_option_score(prompt, option)
            self.assertEqual(by_option[option]["tokens"], rows)
            self.assertEqual(by_option[option]["logprob"], total,
                             f"{option!r}: the photo scored differently from a cold engine")

    def test_chat_logprobs_reach_the_engine(self):
        with patch("openai_server.ARCH", "mimo"):
            out = self.post("/v1/chat/completions", {
                "model": "mimo-tiny", "max_tokens": 6, "logprobs": True,
                "top_logprobs": 2, "enable_thinking": False,
                "messages": [{"role": "user", "content": "Ship it?"}]})
        choice = out["choices"][0]
        entries = choice["logprobs"]["content"]
        self.assertTrue(entries, "no per-token logprobs came back")
        for entry in entries:
            self.assertLessEqual(entry["logprob"], 0.0)
            self.assertEqual(len(entry["top_logprobs"]), 2)


if __name__ == "__main__":
    unittest.main()
