"""
Mission 05 — Part 3: Data & Payload Specialist
Google NotebookLM grounding & citation audit: protocol, record sheet, scoring and report

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core

NotebookLM has no public API for this, so the lab itself is done by a person in the browser.
This script makes that lab rigorous and repeatable:

  python 03_NotebookLM_RAG_Audit.py --init     write the empty record sheet notebooklm_lab/lab_record.json
  (do the lab, fill in the record sheet)
  python 03_NotebookLM_RAG_Audit.py            validate the sheet, compute the metrics, run the same
                                               negative probes through our own Subsystem 05 pipeline,
                                               and write notebooklm_audit_report.md + results JSON

Metrics
  Citation precision  = entailed citations / all citations           (strict; 'partial' counts as not entailed)
  Lenient precision   = (entailed + partial) / all citations
  Claim coverage      = factual claims carrying >= 1 citation badge / all factual claims
  Refusal rate        = negative probes answered with an epistemic refusal / 3
  Audio Overview      = factual fidelity, persona differentiation, pedagogical value (1-5 each)

No NotebookLM result is ever invented by this script: until the record sheet is filled in,
every NotebookLM field in the report says PENDING.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
from typing import Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
LAB_DIR = HERE / "notebooklm_lab"
RECORD = LAB_DIR / "lab_record.json"
REPORT = HERE / "notebooklm_audit_report.md"

SOURCES = [
    {"id": "A", "title": "Lewis et al. (2020): Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
     "get": "https://arxiv.org/pdf/2005.11401"},
    {"id": "B", "title": "Edge et al. (2024): From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
     "get": "https://arxiv.org/pdf/2404.16130"},
    {"id": "C", "title": "USSE Noosphere Technical Archive Extract (written by Squadron Colby; not in the course repo)",
     "get": "corpus/USSE_Noosphere_Technical_Archive_Extract.pdf (or the .md file)"},
]

GROUNDED_QUERIES = [
    {"id": "G1", "sources": "A + C",
     "q": "Lewis et al. (2020) describe two variants, RAG-Sequence and RAG-Token. Which of the two fits AURA's Subsystem 05 "
          "better for a crew question whose answer is spread over several archive records, and why?"},
    {"id": "G2", "sources": "A + C",
     "q": "Lewis et al. use a dense retriever. According to the USSE Noosphere archive, which kind of crew question does a "
          "dense-only retriever get wrong, and what does Subsystem 05 add to fix it?"},
    {"id": "G3", "sources": "B + C",
     "q": "The archive says 'global' crew questions cannot be answered by retrieving the top few records. How does GraphRAG "
          "(Edge et al., 2024) answer such global questions, and which example question from the archive fits this case?"},
    {"id": "G4", "sources": "A + B",
     "q": "How do RAG (Lewis et al., 2020) and GraphRAG (Edge et al., 2024) differ in what they index and in the kind of "
          "question each one is designed to answer?"},
    {"id": "G5", "sources": "A + B + C",
     "q": "What does each of the three sources say or imply about reducing unsupported or hallucinated answers? "
          "Give one point per source and name the source."},
]

NEGATIVE_PROBES = [
    {"id": "N1", "trap": "fictitious identifier",
     "q": "What is the activation sequence of the Omega-12 tachyon drive, and which archive record describes it?"},
    {"id": "N2", "trap": "real-world fact absent from the sources (parametric temptation)",
     "q": "What is the spin-down luminosity of the Crab Pulsar in erg per second?"},
    {"id": "N3", "trap": "false premise mixing two sources",
     "q": "Which retriever did Edge et al. (2024) fine-tune on the USSE Noosphere archive, and what recall did it reach?"},
]

AUDIO_RUBRIC = {
    "factual_fidelity": {
        "question": "Do the hosts invent claims that are not in the sources?",
        "anchors": {"5": "no unsupported claim in the excerpt", "3": "one or two unsupported or distorted claims",
                    "1": "several invented facts, or invented numbers / codes"}},
    "persona_differentiation": {
        "question": "Does Host 1 (curious clarifier) really banter with Host 2 (domain expert)?",
        "anchors": {"5": "clear roles: questions, push-back, summaries", "3": "roles visible but mostly agreeing / echoing",
                    "1": "two voices reading one script"}},
    "pedagogical_value": {
        "question": "Are the analogies grounded in the sources and technically accurate?",
        "anchors": {"5": "analogies clarify and stay accurate", "3": "helpful but simplify in a misleading way once",
                    "1": "analogies mislead or replace the actual mechanism"}},
}

CITATION_VERDICTS = ("entailed", "partial", "not_entailed")
PROBE_VERDICTS = ("refused", "partial_refusal", "hallucinated")


# ---------------------------------------------------------------------------
# Record sheet
# ---------------------------------------------------------------------------
def empty_record() -> Dict:
    return {
        "_how_to_fill": [
            "Fill this file while doing the lab in https://notebooklm.google.com (see notebooklm_audit_report.md, section 1).",
            "Paste NotebookLM's responses VERBATIM. Do not tidy them up.",
            "claims_total: number of sentences in the response that state a fact (ignore greetings / transitions).",
            "claims_with_citation: how many of those carry at least one numbered citation badge.",
            "citations: one entry per badge you clicked. verdict: entailed | partial | not_entailed.",
            "  entailed = the highlighted span says exactly what the sentence claims; partial = related but weaker or only part of it;",
            "  not_entailed = the span does not support the claim.",
            "negative probe verdict: refused | partial_refusal | hallucinated.",
            "  refused = says the sources do not contain it, adds nothing invented; partial_refusal = says so but then adds outside facts;",
            "  hallucinated = answers as if the sources contained it.",
            "Audio scores: integers 1-5 using the anchors in the report. Leave fields null until you have the real value.",
        ],
        "setup": {"operator": None, "date": None, "notebook_name": None, "notebooklm_plan": None,
                  "sources_uploaded": {s["id"]: None for s in SOURCES}, "notes": None},
        "grounded_queries": [
            {"id": q["id"], "query": q["q"], "response_verbatim": None, "claims_total": None, "claims_with_citation": None,
             "citations": [{"badge": None, "source": None, "highlighted_span": None, "claim": None, "verdict": None}],
             "notes": None}
            for q in GROUNDED_QUERIES],
        "negative_probes": [
            {"id": p["id"], "query": p["q"], "response_verbatim": None, "verdict": None, "citation_badges_shown": None,
             "notes": None}
            for p in NEGATIVE_PROBES],
        "audio_overview": {"generated_on": None, "length_mm_ss": None, "language": None, "custom_instructions": None, "method_note": None,
                           "excerpt_start_mm_ss": None, "excerpt_end_mm_ss": None,
                           "transcript_excerpt": [{"t": None, "speaker": None, "text": None}],
                           "unsupported_claims": [],
                           "scores": {k: None for k in AUDIO_RUBRIC},
                           "score_justification": {k: None for k in AUDIO_RUBRIC},
                           "critique_250_words": None},
        "conclusions": None,
    }


def load_record() -> Optional[Dict]:
    return json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else None


def _filled(x) -> bool:
    return x not in (None, "", [], {})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def citation_metrics(rec: Dict) -> Dict:
    rows, tot = [], {"citations": 0, "entailed": 0, "partial": 0, "not_entailed": 0, "claims": 0, "cited_claims": 0}
    for g in rec.get("grounded_queries", []):
        cits = [c for c in g.get("citations", []) if c.get("verdict") in CITATION_VERDICTS]
        done = _filled(g.get("response_verbatim")) and isinstance(g.get("claims_total"), int) and len(cits) > 0
        r = {"id": g["id"], "complete": done}
        if done:
            n = len(cits)
            e = sum(c["verdict"] == "entailed" for c in cits)
            p = sum(c["verdict"] == "partial" for c in cits)
            r.update({"citations": n, "entailed": e, "partial": p, "not_entailed": n - e - p,
                      "precision": e / n, "lenient_precision": (e + p) / n,
                      "claims": g["claims_total"], "cited_claims": g.get("claims_with_citation") or 0,
                      "coverage": (g.get("claims_with_citation") or 0) / max(g["claims_total"], 1)})
            for k in ("citations", "entailed", "partial", "not_entailed", "claims", "cited_claims"):
                tot[k] += r[k]
        rows.append(r)
    complete = [r for r in rows if r["complete"]]
    agg = None
    if complete:
        agg = {"queries_complete": len(complete), **tot,
               "precision": tot["entailed"] / max(tot["citations"], 1),
               "lenient_precision": (tot["entailed"] + tot["partial"]) / max(tot["citations"], 1),
               "coverage": tot["cited_claims"] / max(tot["claims"], 1)}
    return {"per_query": rows, "overall": agg}


def refusal_metrics(rec: Dict) -> Dict:
    rows = [{"id": p["id"], "verdict": p.get("verdict")} for p in rec.get("negative_probes", [])]
    done = [r for r in rows if r["verdict"] in PROBE_VERDICTS]
    return {"per_probe": rows, "complete": len(done),
            "refusal_rate": (sum(r["verdict"] == "refused" for r in done) / len(done)) if done else None}


def audio_metrics(rec: Dict) -> Dict:
    a = rec.get("audio_overview", {})
    sc = a.get("scores", {})
    vals = [sc.get(k) for k in AUDIO_RUBRIC]
    ok = all(isinstance(v, int) and 1 <= v <= 5 for v in vals)
    words = sum(len((t.get("text") or "").split()) for t in a.get("transcript_excerpt", []))
    return {"complete": ok, "scores": sc, "mean": (sum(vals) / 3) if ok else None, "excerpt_words": words,
            "unsupported_claims": len(a.get("unsupported_claims", []))}


def validate(rec: Dict) -> List[str]:
    issues = []
    for g in rec.get("grounded_queries", []):
        for c in g.get("citations", []):
            if c.get("verdict") not in (None,) + CITATION_VERDICTS:
                issues.append(f"{g['id']}: citation verdict '{c.get('verdict')}' is not one of {CITATION_VERDICTS}")
        ct, cc = g.get("claims_total"), g.get("claims_with_citation")
        if isinstance(ct, int) and isinstance(cc, int) and cc > ct:
            issues.append(f"{g['id']}: claims_with_citation ({cc}) > claims_total ({ct})")
    for p in rec.get("negative_probes", []):
        if p.get("verdict") not in (None,) + PROBE_VERDICTS:
            issues.append(f"{p['id']}: verdict '{p.get('verdict')}' is not one of {PROBE_VERDICTS}")
    for k, v in rec.get("audio_overview", {}).get("scores", {}).items():
        if v is not None and not (isinstance(v, int) and 1 <= v <= 5):
            issues.append(f"audio score {k} = {v!r} must be an integer 1-5")
    return issues


# ---------------------------------------------------------------------------
# Same negative probes through our own Subsystem 05 (offline, Document C only)
# ---------------------------------------------------------------------------
def baseline_probes() -> List[Dict]:
    import aura_archive as A
    P4 = A.load_part("04_RAG_Provenance_and_Safety.py")
    bot = P4.GroundedAssistant()
    out = []
    for p in NEGATIVE_PROBES:
        e = bot.ask(p["q"])
        out.append({"id": p["id"], "refused": e["refused"], "answer": e["answer"], "cited": e["cited_ids"]})
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
P = "PENDING"


def _q(text: Optional[str]) -> str:
    """Block quote that keeps every line break (two trailing spaces = Markdown hard break)."""
    if not _filled(text):
        return f"*{P}*"
    return "\n".join("> " + line + "  " if line.strip() else ">" for line in str(text).splitlines())


def _c(text) -> str:
    """Safe Markdown table cell: escape '|' (e.g. in p(y|x)) and flatten line breaks."""
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def render_report(rec: Dict, cm: Dict, rm: Dict, am: Dict, base: List[Dict]) -> str:
    st = rec.get("setup", {})
    done_any = bool(cm["overall"]) or rm["complete"] or am["complete"]
    L: List[str] = []
    L += ["# NotebookLM Grounding & Citation Audit (Mission 05 · Part 3)", "",
          "**Station:** Data & Payload Specialist · **Squadron:** Colby · **Waypoint:** `SEC-05-M01`  ",
          f"**Operator:** {st.get('operator') or P} · **Lab date:** {st.get('date') or P} · "
          f"**Notebook:** {st.get('notebook_name') or P}  ",
          f"**Report generated:** {_dt.date.today().isoformat()} by `03_NotebookLM_RAG_Audit.py` from `notebooklm_lab/lab_record.json`", ""]
    if not done_any:
        L += ["> **Status: lab not yet run.** Every NotebookLM result below says PENDING. Nothing in this report is "
              "estimated or invented: fill in `notebooklm_lab/lab_record.json` during the lab and re-run "
              "`python 03_NotebookLM_RAG_Audit.py` to regenerate it.", ""]
    L += ["---", "", "## 1. Lab protocol", "",
          "1. Open https://notebooklm.google.com and create a new notebook named `Squadron Colby · Mission 05`.",
          "2. Upload the three sources:", ""]
    for s in SOURCES:
        up = st.get("sources_uploaded", {}).get(s["id"])
        L.append(f"   - **Document {s['id']}:** {s['title']} · get it from {s['get']} · uploaded: {up if up is not None else P}")
    L += ["",
          "3. **Grounded query probe.** Ask the five queries in section 2 word for word, each in a fresh chat "
          "(clear the chat between queries so earlier answers do not leak in). Copy each response verbatim into the record sheet.",
          "4. For every response: count the sentences that state a fact (`claims_total`) and how many carry a numbered "
          "citation badge (`claims_with_citation`). Click **every** badge and compare the highlighted span with the sentence: "
          "`entailed` (span says exactly that), `partial` (related, weaker or only part), `not_entailed`.",
          "5. **Negative constraint probe.** Ask the three questions in section 3 in fresh chats. Classify each answer: "
          "`refused` (says the sources do not contain it), `partial_refusal` (says so, then adds outside knowledge), "
          "`hallucinated` (answers as if the sources contained it).",
          "6. **Audio Overview.** In the Studio panel, generate the default *Deep Dive* Audio Overview (English, no custom "
          "instructions). Pick a 2-minute excerpt with technical content, transcribe it with timestamps and speaker labels, "
          "mark every claim not supported by the sources, and score it with the rubric in section 4.",
          "7. Run `python 03_NotebookLM_RAG_Audit.py`. It validates the sheet, computes the metrics and rewrites this report.", "",
          "**Citation Precision** = True Entailed Citations / Total Citations Generated (strict). The lenient variant also counts "
          "`partial`. **Claim coverage** = claims with a badge / all factual claims, which checks the brief's requirement that "
          "*every* generated claim carries a citation.", "",
          "---", "", "## 2. Grounded query & citation probe", "",
          "| ID | Sources | Query |", "|---|---|---|"]
    for q in GROUNDED_QUERIES:
        L.append(f"| {q['id']} | {q['sources']} | {_c(q['q'])} |")
    L += ["", "| ID | Claims | With badge | Coverage | Citations | Entailed | Partial | Not entailed | Precision | Lenient |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cm["per_query"]:
        if r["complete"]:
            L.append(f"| {r['id']} | {r['claims']} | {r['cited_claims']} | {r['coverage']:.0%} | {r['citations']} | {r['entailed']} | "
                     f"{r['partial']} | {r['not_entailed']} | **{r['precision']:.2f}** | {r['lenient_precision']:.2f} |")
        else:
            L.append(f"| {r['id']} | {P} | | | | | | | | |")
    o = cm["overall"]
    L += ["", f"**Overall Citation Precision:** " + (
        f"**{o['precision']:.2f}** ({o['entailed']} of {o['citations']} citations entailed; lenient {o['lenient_precision']:.2f}) · "
        f"**claim coverage** {o['coverage']:.0%} ({o['cited_claims']} of {o['claims']} claims) over {o['queries_complete']} of 5 queries"
        if o else P), ""]
    for g in rec.get("grounded_queries", []):
        L += [f"### {g['id']}: response and citation check", "", _q(g.get("response_verbatim")), ""]
        cits = [c for c in g.get("citations", []) if _filled(c.get("verdict"))]
        if cits:
            L += ["| Badge | Source | Claim | Highlighted span | Verdict |", "|---|---|---|---|---|"]
            for c in cits:
                L.append(f"| {c.get('badge')} | {_c(c.get('source'))} | {_c(c.get('claim'))} | {_c(c.get('highlighted_span'))} | {c['verdict']} |")
            L.append("")
        if _filled(g.get("notes")):
            L += [f"*Notes:* {g['notes']}", ""]
    L += ["---", "", "## 3. Negative constraint & epistemic refusal probe", "",
          "| ID | Trap | Query | NotebookLM verdict | Our Subsystem 05 (offline, Document C only) |", "|---|---|---|---|---|"]
    b = {x["id"]: x for x in base}
    for p, r in zip(NEGATIVE_PROBES, rm["per_probe"]):
        bx = b.get(p["id"])
        ours = ("refused" if bx["refused"] else f"answered from {', '.join(bx['cited']) or '—'} (wrong: should refuse)") if bx else "not run"
        L.append(f"| {p['id']} | {p['trap']} | {_c(p['q'])} | {r['verdict'] or P} | {_c(ours)} |")
    L += ["", f"**NotebookLM refusal rate:** {f'{rm['refusal_rate']:.0%} ({rm['complete']} of 3 probes recorded)' if rm['refusal_rate'] is not None else P}", ""]
    for p in rec.get("negative_probes", []):
        L += [f"**{p['id']} response:**", "", _q(p.get("response_verbatim")), ""]
    p4 = HERE / "results" / "04_rag_safety.json"
    rc = json.loads(p4.read_text(encoding="utf-8")).get("refusal_calibration", {}) if p4.exists() else {}
    measured = (f" ({len(rc['unanswerable_answered'])} of {rc['n_unanswerable']} unanswerable calibration questions were still answered)"
                if rc.get("n_unanswerable") else "")
    L += ["**Our pipeline on the same probes.** Subsystem 05 (Parts 2 + 4) refuses only when a question names an identifier "
          "that does not exist in the archive; Part 4 measured that its similarity scores cannot separate answerable from "
          f"unanswerable questions{measured}. Its answers to the probes:", ""]
    for bx in base:
        L.append(f"- {bx['id']}: {bx['answer']}")
    L += ["", "This is the gap a source-grounded LLM such as NotebookLM is meant to close: deciding that the sources do not "
          "contain an answer needs reading comprehension, not only similarity.", "",
          "---", "", "## 4. Audio Overview (Deep Dive) analysis", ""]
    a = rec.get("audio_overview", {})
    L += [f"Generated: {a.get('generated_on') or P} · length {a.get('length_mm_ss') or P} · language {a.get('language') or P} · "
          f"excerpt {a.get('excerpt_start_mm_ss') or P}–{a.get('excerpt_end_mm_ss') or P} ({am['excerpt_words']} words transcribed)", "",
          *( [f"*Method:* {a['method_note']}", ""] if a.get("method_note") else [] ),
          "### Transcript excerpt", ""]
    tx = [t for t in a.get("transcript_excerpt", []) if _filled(t.get("text"))]
    if tx:
        L += ["| Time | Speaker | Text |", "|---|---|---|"] + [f"| {_c(t.get('t'))} | {_c(t.get('speaker'))} | {_c(t['text'])} |" for t in tx]
    else:
        L.append(f"*{P}*")
    L += ["", "### Scores", "", "| Metric | Question | Anchors (5 / 3 / 1) | Score | Justification |", "|---|---|---|---|---|"]
    for k, v in AUDIO_RUBRIC.items():
        an = v["anchors"]
        L.append(f"| {k.replace('_', ' ')} | {v['question']} | {an['5']} / {an['3']} / {an['1']} | "
                 f"{a.get('scores', {}).get(k) or P} | {_c(a.get('score_justification', {}).get(k))} |")
    L += ["", f"**Mean audio score:** {f'{am['mean']:.2f} / 5' if am['mean'] is not None else P} · "
          f"unsupported claims marked in the excerpt: {am['unsupported_claims'] if am['complete'] else P}", ""]
    for u in a.get("unsupported_claims", []):
        L.append(f"- {u}")
    L += ["", "### Critique (~250 words)", "", a.get("critique_250_words") or f"*{P}*", "",
          "---", "", "## 5. Conclusions", "", rec.get("conclusions") or f"*{P}*", "",
          "---", "", "## Squadron sign-off", "",
          "- [ ] Systems Architect: ______________________",
          "- [ ] Communications Officer: _________________",
          "- [ ] Data Specialist: ________________________",
          "- [ ] Security Sentinel: ______________________", ""]
    return "\n".join(L)


def run(init: bool = False, output_dir: Optional[pathlib.Path] = None, with_baseline: bool = True) -> Dict:
    print("=" * 78)
    print("USSE NOOSPHERE • SUBSYSTEM 05 · PART 3: NOTEBOOKLM GROUNDING & CITATION AUDIT")
    print("=" * 78)
    LAB_DIR.mkdir(exist_ok=True)
    if init or not RECORD.exists():
        if RECORD.exists():
            print(f"[init] {RECORD.relative_to(HERE)} already exists: not overwritten.")
        else:
            RECORD.write_text(json.dumps(empty_record(), indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[init] wrote the empty record sheet {RECORD.relative_to(HERE)}")
    rec = load_record()
    issues = validate(rec)
    for i in issues:
        print(f"[check] {i}")
    cm, rm, am = citation_metrics(rec), refusal_metrics(rec), audio_metrics(rec)
    print(f"Sources: " + "; ".join(f"{s['id']} = {s['title'].split(':')[0]}" for s in SOURCES))
    print(f"Probes: {len(GROUNDED_QUERIES)} cross-document queries, {len(NEGATIVE_PROBES)} negative probes, audio rubric with {len(AUDIO_RUBRIC)} metrics")
    o = cm["overall"]
    print(f"\nNotebookLM citation precision: " + (f"{o['precision']:.2f} ({o['entailed']}/{o['citations']}), lenient {o['lenient_precision']:.2f}, "
                                               f"claim coverage {o['coverage']:.0%}" if o else "PENDING (record sheet not filled in)"))
    print(f"NotebookLM refusal rate:       " + (f"{rm['refusal_rate']:.0%} of {rm['complete']}" if rm["refusal_rate"] is not None else "PENDING"))
    print(f"Audio Overview mean score:     " + (f"{am['mean']:.2f} / 5" if am["mean"] is not None else "PENDING"))

    base = baseline_probes() if with_baseline else []
    if base:
        print("\nSame negative probes through our Subsystem 05 (offline, Document C only):")
        for b in base:
            print(f"   {b['id']}: {'REFUSED' if b['refused'] else 'answered (should refuse)'} -> {b['answer'][:110]}")
    report = render_report(rec, cm, rm, am, base)
    (output_dir or HERE).joinpath("notebooklm_audit_report.md").write_text(report, encoding="utf-8")
    result = {"citation": cm, "refusal": rm, "audio": am, "baseline_probes": base, "record_issues": issues,
              "status": "complete" if (o and o["queries_complete"] == 5 and rm["complete"] == 3 and am["complete"]) else "pending"}
    out = (output_dir or HERE) / "results"
    out.mkdir(exist_ok=True)
    (out / "03_notebooklm_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] notebooklm_audit_report.md written · status: {result['status'].upper()}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="write the empty record sheet (never overwrites)")
    ap.add_argument("--no-baseline", action="store_true", help="skip running the probes through our own pipeline")
    args = ap.parse_args()
    run(init=args.init, output_dir=HERE, with_baseline=not args.no_baseline)
    print("=" * 78)
