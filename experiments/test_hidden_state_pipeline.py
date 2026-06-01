"""
Hidden State Experiment Test Suite

Tests and compares:
- Baseline RAG (with re-encoding)
- Hidden-state pipeline with different aggregation methods
- Measures latency, quality, and storage efficiency
"""

import requests
import json
import time
from typing import Dict, List
from datetime import datetime


BASE_URL = "http://localhost:8000/api"

# Test queries
TEST_QUERIES = [
    "What is the main contribution of this paper?",
    "How does this approach differ from previous work?",
    "What are the experimental results?",
    "What datasets were used?",
    "What are the limitations of this work?",
]

AGGREGATION_METHODS = ["sequence_concatenate"]


class ExperimentLogger:
    def __init__(self, log_file: str = None):
        self.log_file = log_file or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "baseline_rag": [],
            "hidden_state_experiments": {method: [] for method in AGGREGATION_METHODS},
            "summary": {},
        }

    def log_baseline(self, query: str, response: Dict):
        """Log baseline RAG response"""
        self.results["baseline_rag"].append({
            "query": query,
            "answer": response.get("answer", ""),
            "num_contexts": len(response.get("contexts", [])),
        })

    def log_hidden_state(self, method: str, query: str, response: Dict):
        """Log hidden-state experiment response"""
        self.results["hidden_state_experiments"][method].append({
            "query": query,
            "answer": response.get("answer", ""),
            "aggregation_method": method,
            "num_chunks_with_hidden_states": response.get("num_chunks_with_hidden_states", 0),
            "timing": response.get("timing", {}),
        })

    def save(self):
        """Save results to JSON file"""
        with open(self.log_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Results saved to: {self.log_file}")


def check_services():
    """Verify that backend and Qdrant are running"""
    print("🔍 Checking services...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Start it with:")
        print("   cd c:\\programming\\ResearchOS")
        print("   .\\venv\\Scripts\\Activate.ps1")
        print("   uvicorn app.main:app --reload")
        return False


def upload_sample_pdf():
    """Upload a sample PDF for testing"""
    print("\n📄 Uploading sample PDF...")
    
    # For testing, you need an actual PDF file
    pdf_path = "c:\\path\\to\\your\\test.pdf"  # Change this
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/upload-paper", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ PDF uploaded successfully")
            print(f"   - Chunks: {result.get('chunks', 0)}")
            print(f"   - Pages: {result.get('total_pages', 0)}")
            return True
        else:
            print(f"❌ Failed to upload PDF: {response.text}")
            return False
    except FileNotFoundError:
        print(f"⚠️  PDF file not found at {pdf_path}")
        print("   Using mock data instead for testing...")
        return True
    except Exception as e:
        print(f"❌ Error uploading PDF: {e}")
        return False


def preprocess_hidden_states(num_chunks: int = 5):
    """Preprocess hidden states for chunks"""
    print(f"\n🔄 Preprocessing hidden states for {num_chunks} chunks...")
    
    # Mock chunk data for testing (replace with real chunks from your PDF)
    mock_texts = [
        "This paper proposes a novel approach to neural machine translation using attention mechanisms.",
        "The transformer architecture has revolutionized natural language processing in recent years.",
        "We evaluate our method on multiple benchmark datasets including BLEU scores and human evaluation.",
        "The experimental results show significant improvements over the baseline model.",
        "Future work includes extending this approach to other NLP tasks like summarization.",
    ][:num_chunks]
    
    mock_metadata = [
        {"source_filename": "test_paper.pdf", "chunk_index": i}
        for i in range(num_chunks)
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/preprocess-hidden-states",
            json={
                "texts": mock_texts,
                "chunk_metadata": mock_metadata,
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hidden states preprocessed")
            print(f"   - Stored: {result.get('stored', 0)}")
            print(f"   - Failed: {result.get('failed', 0)}")
            return True
        else:
            print(f"❌ Failed to preprocess: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return False


def test_baseline_rag(query: str) -> Dict:
    """Test baseline RAG pipeline"""
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"query": query, "limit": 5},
            timeout=60,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Baseline RAG failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error in baseline RAG: {e}")
        return None


def test_hidden_state_pipeline(query: str, method: str) -> Dict:
    """Test hidden-state pipeline with specific aggregation method"""
    try:
        response = requests.post(
            f"{BASE_URL}/ask-with-hidden-states",
            json={
                "query": query,
                "limit": 5,
                "aggregation_method": method,
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Hidden-state pipeline ({method}) failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error in hidden-state pipeline ({method}): {e}")
        return None


def print_comparison(baseline: Dict, hidden_state_results: Dict[str, Dict]):
    """Print formatted comparison of results"""
    if not baseline or not hidden_state_results:
        return
    
    print("\n" + "="*80)
    print("📊 TIMING COMPARISON")
    print("="*80)
    
    baseline_time = baseline.get("timing", {}).get("total_ms", 0) if baseline else 0
    
    if baseline_time > 0:
        print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
        print("-" * 45)
        print(f"{'Baseline RAG':<20} {baseline_time:<15.2f} {'1.0x':<10}")
        
        for method, result in hidden_state_results.items():
            if result:
                hs_time = result.get("timing", {}).get("total_ms", 0)
                if hs_time > 0:
                    speedup = baseline_time / hs_time
                    print(f"{method:<20} {hs_time:<15.2f} {f'{speedup:.2f}x':<10}")
    
    print("\n" + "="*80)
    print("📝 DETAILED TIMING BREAKDOWN (Hidden-State Concatenate)")
    print("="*80)
    
    if hidden_state_results.get("concatenate"):
        timing = hidden_state_results["concatenate"].get("timing", {})
        print(f"Embedding:   {timing.get('embedding_ms', 0):.2f} ms")
        print(f"Retrieval:   {timing.get('retrieval_ms', 0):.2f} ms")
        print(f"Aggregation: {timing.get('aggregation_ms', 0):.2f} ms")
        print(f"Generation:  {timing.get('generation_ms', 0):.2f} ms")
        print(f"{'─'*40}")
        print(f"Total:       {timing.get('total_ms', 0):.2f} ms")
    
    print("\n" + "="*80)
    print("💬 SAMPLE ANSWERS")
    print("="*80)
    
    if baseline:
        answer = baseline.get("answer", "")[:100]
        print(f"Baseline RAG:\n{answer}...\n")
    
    for method, result in hidden_state_results.items():
        if result:
            answer = result.get("answer", "")[:100]
            print(f"{method.upper()}:\n{answer}...\n")


def run_full_experiment():
    """Run complete experiment workflow"""
    print("\n" + "="*80)
    print("🧪 HIDDEN STATE PIPELINE EXPERIMENT")
    print("="*80)
    
    # Check services
    if not check_services():
        return
    
    logger = ExperimentLogger()
    
    # Setup
    # upload_sample_pdf()
    if not preprocess_hidden_states(num_chunks=5):
        print("⚠️  Preprocessing failed, continuing with test...")
    
    # Run tests on first few queries
    test_queries = TEST_QUERIES[:3]
    
    print(f"\n🎯 Running {len(test_queries)} test queries...")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query[:60]}...")
        
        # Baseline RAG
        print("  → Baseline RAG...", end=" ", flush=True)
        baseline = test_baseline_rag(query)
        if baseline:
            print(f"✅ ({baseline.get('timing', {}).get('total_ms', 'N/A')}ms)")
            logger.log_baseline(query, baseline)
        else:
            print("❌")
        
        # Hidden-state experiments
        hs_results = {}
        for method in AGGREGATION_METHODS:
            print(f"  → Hidden-state ({method})...", end=" ", flush=True)
            result = test_hidden_state_pipeline(query, method)
            if result:
                print(f"✅ ({result.get('timing', {}).get('total_ms', 'N/A')}ms)")
                logger.log_hidden_state(method, query, result)
                hs_results[method] = result
            else:
                print("❌")
        
        # Print comparison for this query
        print_comparison(baseline, hs_results)
        time.sleep(1)  # Rate limiting
    
    # Save results
    logger.save()
    
    print("\n" + "="*80)
    print("✨ EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_full_experiment()
