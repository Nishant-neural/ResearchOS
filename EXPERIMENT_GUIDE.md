# Hidden State Pipeline Experiment Guide

## Quick Start

### 1. Start Services

**Terminal 1 - Start Qdrant (vector database):**
```powershell
docker run -p 6333:6333 qdrant/qdrant
```

**Terminal 2 - Start Backend:**
```powershell
cd c:\programming\ResearchOS
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`

---

## Reset Qdrant Vector Database

### Option 1: Delete Collection Only (Recommended)
Keeps Qdrant running, just clears the data:

```powershell
cd c:\programming\ResearchOS\backend
.\.venv\Scripts\Activate.ps1
python ..\scripts\reset_qdrant.py --action delete-collection
```

✅ Fast, keeps Qdrant container running
⏱️ ~2 seconds

### Option 2: Full Reset
Stops and removes the Qdrant container completely:

```powershell
cd c:\programming\ResearchOS\backend
.\.venv\Scripts\Activate.ps1
python ..\scripts\reset_qdrant.py --action full-reset
```

⚠️ Slower but guaranteed clean state
⏱️ ~10 seconds

### Option 3: Check Status
See what collections exist and their point counts:

```powershell
python ..\scripts\reset_qdrant.py --action check
```

---

## Run Experiment

### Option A: Automated Test Script (Recommended)

```powershell
cd c:\programming\ResearchOS\backend
.\.venv\Scripts\Activate.ps1
python ..\experiments\test_hidden_state_pipeline.py
```

This will:
1. ✅ Check services are running
2. 🔄 Preprocess mock chunks with hidden states
3. 🎯 Run 3 test queries through both pipelines
4. 📊 Display timing comparisons
5. 💾 Save detailed results to `experiment_YYYYMMDD_HHMMSS.json`

**Output example:**
```
[1/3] Query: What is the main contribution of this paper?...
  → Baseline RAG... ✅ (450.23ms)
  → Hidden-state (concatenate)... ✅ (120.45ms)
  → Hidden-state (mean)... ✅ (118.32ms)
  → Hidden-state (weighted)... ✅ (122.67ms)
  → Hidden-state (stacking)... ✅ (125.89ms)

📊 TIMING COMPARISON
═════════════════════════════════
Method               Time (ms)      Speedup   
─────────────────────────────────
Baseline RAG             450.23        1.0x
concatenate              120.45        3.74x
mean                     118.32        3.80x
weighted                 122.67        3.67x
stacking                 125.89        3.57x
```

### Option B: Manual Testing with cURL

**1. Upload PDF:**
```bash
curl -X POST "http://localhost:8000/upload-paper" `
  -F "file=@C:\path\to\paper.pdf"
```

**2. Preprocess hidden states:**
```bash
curl -X POST "http://localhost:8000/preprocess-hidden-states" `
  -H "Content-Type: application/json" `
  -d '{
    "texts": ["chunk1", "chunk2", "chunk3"],
    "chunk_metadata": [
      {"source_filename": "paper.pdf", "chunk_index": 0},
      {"source_filename": "paper.pdf", "chunk_index": 1},
      {"source_filename": "paper.pdf", "chunk_index": 2}
    ]
  }'
```

**3. Query baseline RAG:**
```bash
curl -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"query": "What is the main contribution?", "limit": 5}'
```

**4. Query hidden-state pipeline:**
```bash
curl -X POST "http://localhost:8000/ask-with-hidden-states" `
  -H "Content-Type: application/json" `
  -d '{
    "query": "What is the main contribution?",
    "limit": 5,
    "aggregation_method": "concatenate"
  }'
```

Try different methods: `"mean"`, `"weighted"`, `"stacking"`

---

## Understanding the Results

### Timing Breakdown

```
embedding_ms     : Time to embed the query
retrieval_ms     : Time to retrieve from Qdrant
aggregation_ms   : Time to combine hidden states
generation_ms    : Time for decoder to generate answer
total_ms         : Total end-to-end latency
```

### What to Look For

| Metric | Expected Result |
|--------|-----------------|
| **Speedup** | Hidden-state should be 2-5x faster than baseline |
| **Generation time** | Should be similar across all methods (decoder usage is same) |
| **Aggregation time** | Concatenate > Stacking > Mean > Weighted (in terms of computation) |
| **Answer quality** | All methods should produce reasonable answers (may differ) |

---

## Aggregation Methods Explained

### Concatenate (Default) 🎯
```
[chunk1: 768] + [chunk2: 768] + [chunk3: 768] = [2304]
```
- **Pros**: Preserves all information
- **Cons**: Longer sequence for decoder
- **Best for**: Comparing information from multiple chunks

### Mean Pooling
```
Average([chunk1, chunk2, chunk3]) = [768]
```
- **Pros**: Compact representation, fast
- **Cons**: Information loss from averaging
- **Best for**: Quick queries

### Stacking
```
Sequence of [chunk1, chunk2, chunk3]
```
- **Pros**: Treats chunks as tokens
- **Cons**: May lose context between chunks
- **Best for**: Traditional seq2seq models

### Weighted by Relevance
```
(weight1 * chunk1) + (weight2 * chunk2) + (weight3 * chunk3)
```
- **Pros**: Prioritizes most relevant chunks
- **Cons**: Slightly more computation
- **Best for**: Quality over speed

---

## Troubleshooting

### Backend connection refused
```
❌ Backend is not running. Start it with:
   uvicorn app.main:app --reload
```

### Qdrant connection refused
```
❌ Qdrant is not running. Start with:
   docker run -p 6333:6333 qdrant/qdrant
```

### "No pre-computed hidden states available"
```
⚠️  Run /preprocess-hidden-states first
```

### ModuleNotFoundError
```
pip install -r requirements.txt
```

### Qdrant has stale data
```
# Option 1: Delete collection only
python ..\scripts\reset_qdrant.py --action delete-collection

# Option 2: Full container reset
python ..\scripts\reset_qdrant.py --action full-reset
```

---

## Next Steps

1. **Run the automated test script** to see baseline results
2. **Analyze timing**: Which aggregation method is fastest?
3. **Evaluate quality**: Do answers differ between methods?
4. **Optimize**: Based on your use case (speed vs quality)
5. **Real-world test**: Upload your own PDFs and test with domain-specific queries

Good luck with your experiments! 🚀
