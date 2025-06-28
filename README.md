# LogStreamingLLM
use SLM on cpu to tail log files and alert accordingly

Performance Expectations

On a modern CPU, TinyLlama 1.1B in transformers will run each log line classification in ~300ms–600ms

Good enough for ~100–200 lines per minute, depending on CPU threads
