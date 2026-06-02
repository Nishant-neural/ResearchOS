# Persistent Encoder Memory Experiment Log

This log tracks observed behavior of the hidden-state memory pipeline across
different test conditions. It is meant to preserve research signal, not just
software status.

## Current System Status

- The API pipeline works end to end.
- PDFs can be uploaded, chunked, embedded, and stored in Qdrant.
- FLAN-T5 encoder hidden states are precomputed and stored for chunks.
- Hidden-state retrieval can load cached encoder states and attention masks.
- The decoder can generate from cached hidden states using framed latent memory.
- Source filename filtering is available for per-paper evaluation.
- Retrieval previews are recorded in evaluation outputs.

## Core Observations So Far

### 1. Pipeline Works Mechanically

Evidence:

- `hidden_states_used=true`
- `hidden_chunks_used > 0`
- hidden-state endpoint returns generated answers without server errors

Interpretation:

The stored encoder states are usable by the FLAN-T5 decoder. The decoder is not
receiving random noise; it can condition on the cached latent memory.

### 2. Hidden States Preserve Source Information

Evidence:

- On real papers, hidden-state answers often reproduce phrases close to the
  retrieved paper text.
- Examples include source-like phrases such as Transformer/RAG/BERT paper
  definitions and method descriptions.

Interpretation:

Persistent encoder memory preserves enough lexical and semantic information for
the decoder to reconstruct source-like text from latent states.

Current label:

- Latent source reconstruction
- Latent extractive QA behavior

### 3. README Toy Test Showed Stronger Semantic Behavior

Condition:

- The uploaded "paper" was the project README.
- Text was clean, coherent, sectioned, and conceptually focused.
- Chunks were less noisy than real PDF chunks.

Observed behavior:

- Hidden-state answers captured the main thesis:
  - semantic understanding should become persistent
  - contextual semantic computation can be stored
  - the encoder runs once during ingestion instead of every query

Interpretation:

With clean chunks, cached hidden states showed more semantic, thesis-level
answering. This suggests chunk quality strongly affects latent-memory quality.

### 4. Real-Paper Tests Exposed Retrieval And Chunking Weaknesses

Condition:

- Three real PDFs were uploaded:
  - `paper1.pdf`: Attention Is All You Need
  - `paper2.pdf`: BERT
  - `paper3.pdf`: RAG

Observed behavior:

- Retrieval initially mixed papers until `source_filename` filtering was added.
- Retrieval pulled references, acknowledgements, tables, and unrelated fragments.
- A retrieval-time noise filter improved reference pollution.
- Even after filtering, many retrieved chunks were related but not answer-specific.

Interpretation:

The hidden-state method is currently bottlenecked by ingestion and retrieval
quality. Bad chunks produce bad latent memory.

### 5. Query-Conditioned Abstraction Is Still Weak

Evidence:

- Hidden-state answers sometimes repeat broad source statements instead of
  answering the exact question.
- Similar retrieved chunks can produce similar answers for different queries.
- The decoder often extracts or reconstructs rather than synthesizes.

Interpretation:

The model can read cached latent states, but the current conditioning method does
not yet reliably force query-specific abstraction.

Current label:

- Query signal is weaker than latent memory signal.

### 6. Framed Latent Memory Improved Behavior

Change:

The hidden-state generation path was changed from:

```text
[query instruction states]
[cached chunk states]
```

to:

```text
[question + "Relevant latent memory begins"]
[cached chunk states]
["Relevant latent memory ended" + repeated question + answer instruction]
```

Observed behavior:

- Reduced generic thesis repetition on the README test.
- Improved directness for some answers.
- Did not fully solve extractive behavior on noisy real-paper chunks.

Interpretation:

Explicit latent separators help, but do not replace the need for cleaner chunks
and better retrieval.

## Current Research Interpretation

The strongest current claim is:

> Cached encoder hidden states can act as reusable latent source memory.

The project has not yet proven:

> Cached encoder hidden states reliably support high-quality query-conditioned
> reasoning over noisy research PDFs.

Current progress summary:

- Pipeline works.
- Hidden states preserve source information.
- Decoder can reconstruct from latent states.
- README test showed promising semantic behavior with clean chunks.
- Real-paper tests show output is often extractive/copy-like.
- Query-conditioned abstraction is still weak.
- Retrieval and chunk quality are the main bottlenecks.

## Working Hypotheses

### H1: Chunk Quality Controls Latent Memory Quality

Clean coherent chunks produce more useful latent memories. Noisy PDF chunks
produce noisy latent memories and extractive/citation-like outputs.

### H2: Dense Retrieval Alone Is Not Enough

Technical paper QA needs exact term matching and section targeting. Dense vector
retrieval alone often retrieves related but non-answer chunks.

Likely fix:

- hybrid dense + keyword retrieval
- section-aware filtering
- reranking

### H3: Hidden-State Generation Needs Strong Query Framing

The decoder needs explicit latent boundaries and repeated query signal to avoid
generic reconstruction.

Tested strategy:

- `framed_memory`

Future strategies:

- query repeated before and after memory
- answer-type-specific instructions
- learned separator embeddings
- memory compression with query-aware adapter

## Next Experiments

### Experiment A: Clean Chunk Benchmark

Create a small hand-cleaned corpus from the three papers:

- 5-10 short clean chunks per paper
- no references
- no tables
- no broken equations
- one idea per chunk

Goal:

Test whether hidden-state memory behaves more semantically when chunk quality is
controlled.

### Experiment B: Retrieval Quality Audit

For every query, record:

- retrieved chunk ids
- retrieved text previews
- whether the answer is present in retrieved chunks
- whether the answer came from references/noise

Goal:

Separate retrieval failure from hidden-state generation failure.

### Experiment C: Copy/Extractiveness Measurement

Compute overlap between generated answer and retrieved text.

Possible metric:

- longest common substring
- n-gram overlap
- answer tokens present in retrieved chunks

Goal:

Distinguish:

- extractive latent reconstruction
- paraphrased semantic answering
- unsupported hallucination

### Experiment D: Hybrid Retrieval

Combine:

- dense vector retrieval
- BM25/keyword scoring
- source filename filtering
- reference/noise filtering

Goal:

Improve answer-specific evidence selection before judging hidden-state memory.

## Notes For Future Interpretation

Do not evaluate the hidden-state hypothesis from raw answer quality alone unless
retrieval previews show that the correct evidence was retrieved.

A bad answer can come from:

- bad PDF extraction
- bad chunking
- wrong retrieval
- noisy references
- weak query conditioning
- FLAN-T5-base limitations
- hidden-state memory limitations

The experiment must isolate these failure modes before making claims about the
architecture.
