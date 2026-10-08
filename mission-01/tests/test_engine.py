"""Offline tests for the Mission 01 dialogic engine (no API key needed)."""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine", ROOT / "04_dialogic_engine.py")
engine = importlib.util.module_from_spec(spec)
sys.modules["engine"] = engine
spec.loader.exec_module(engine)


def _mock():
    return engine.LLMClient("mock"), engine.RpcLog(), engine.Transcript()


def test_socratic_fsm_reaches_a_halting_state():
    llm, rpc, tx = _mock()
    res = engine.SocraticStateEngine(llm, engine.DEFAULT_CLAIM, rpc, tx).run()
    assert res.path[0] == "S0_ELICIT"
    assert res.path[-1] in ("GROUNDED", "APORIA")
    assert res.refinements <= engine.MAX_REFINEMENTS
    assert len(res.hypotheses) == res.refinements + 1


def test_every_transition_follows_a_defined_edge():
    llm, rpc, tx = _mock()
    res = engine.SocraticStateEngine(llm, engine.DEFAULT_CLAIM, rpc, tx).run()
    S = engine.SocraticState
    for a, b in zip(res.path, res.path[1:]):
        assert S(b) in engine.TRANSITIONS[S(a)]


def test_jsonrpc_messages_carry_required_markers():
    llm, rpc, tx = _mock()
    engine.SocraticStateEngine(llm, engine.DEFAULT_CLAIM, rpc, tx).run()
    markers = {m["params"]["marker"] for m in rpc.messages if "params" in m}
    assert {"PROMPT_PREMISE", "INJECT_COUNTEREXAMPLE", "CONTRADICTION_TRIGGERED"} <= markers
    assert all(m["jsonrpc"] == "2.0" for m in rpc.messages)


def test_loop_limit_forces_aporia():
    llm, rpc, tx = _mock()
    res = engine.SocraticStateEngine(llm, engine.DEFAULT_CLAIM, rpc, tx,
                                     max_refinements=1).run()
    assert res.path[-1] == "APORIA"
    assert any(m.get("error", {}).get("code") == engine.RpcError.LOOP_LIMIT
               for m in rpc.messages)


def test_galilean_debate_has_three_personas_and_synthesis():
    llm, rpc, tx = _mock()
    res = engine.GalileanEthicsSandbox(llm, engine.DEFAULT_TOPIC, rpc, tx).run()
    speakers = {s for s, _ in res.turns}
    assert speakers == {"Salviati", "Simplicio", "Sagredo"}
    assert res.turns[0][0] == "Sagredo" and res.turns[-1][0] == "Sagredo"
    assert res.synthesis


def test_parse_json_handles_code_fences():
    assert engine.parse_json('```json\n{"contradiction": false}\n```') == {"contradiction": False}
    assert engine.parse_json("no json here") is None


def test_full_run_writes_transcript_and_log(tmp_path):
    out, log = tmp_path / "t.txt", tmp_path / "l.json"
    summary = engine.run_all("mock", out=str(out), json_log=str(log))
    assert out.read_text().count("PROTOCOL") >= 3
    assert json.loads(log.read_text())["socratic"]["path"][-1] in ("GROUNDED", "APORIA")
    assert summary["sycophancy"]["fixtures"] == len(engine.SYCOPHANCY_FIXTURES)
