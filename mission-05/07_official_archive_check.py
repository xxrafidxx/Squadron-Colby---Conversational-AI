from pathlib import Path
import re, json
import aura_archive as A

root = Path(__file__).resolve().parent
raw = (root / 'corpus/Document_C_official.md').read_text(encoding='utf-8')
parts = re.split(r'(?m)(?=^#{2,3} )', raw)
chunks = []
for part in parts:
    lines = part.strip().splitlines()
    if lines and lines[0].startswith('## '):
        chunks.append(A.Chunk(f'O-{len(chunks)+1:02d}', lines[0].lstrip('# ').strip(), '\n'.join(lines[1:]).strip()))
    elif lines and lines[0].startswith('### '):
        chunks.append(A.Chunk(f'O-{len(chunks)+1:02d}', lines[0].lstrip('# ').strip(), '\n'.join(lines[1:]).strip()))
P2 = A.load_part('02_BM25_and_RRF_Combiner.py')
P4 = A.load_part('04_RAG_Provenance_and_Safety.py')
bm = P2.BM25Okapi([c.full_text for c in chunks])
checks = {}
for code in ('PR-882-HEX', 'ALPHA-9', 'VALVE-CRYO-404-Z', 'ZK-LUHMANN-89'):
    ranked = bm.rank(code, top_k=5)
    matches = [chunks[i].id for i, _ in ranked if code.lower() in chunks[i].full_text.lower()]
    checks[code] = {'matching_chunks_in_top5': matches, 'pass': bool(matches)}
packet = re.search(r'(?s)```[^\n]*\n(.*?)```', raw)
if not packet:
    raise ValueError('Security packet not found')
scan = P4.HeuristicScanner().scan(packet.group(1))
out = {'source': 'corpus/Document_C_official.md', 'chunking': 'Markdown sections and subsections', 'chunk_count': len(chunks), 'bm25_exact_code_smoke_tests': checks, 'security_packet_flagged': bool(scan.flagged), 'security_categories': scan.categories}
print(json.dumps(out, indent=2))
if not all(v['pass'] for v in checks.values()) or not scan.flagged:
    raise SystemExit(1)
