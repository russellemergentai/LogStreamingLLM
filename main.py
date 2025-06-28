from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
import time
import os

# === 1. Load TinyLlama Model (Transformers, CPU) ===
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.2,
    device=-1  # CPU
)

llm = HuggingFacePipeline(pipeline=pipe)

# === 2. Prompt Template for Summarizing and Suggesting Fix ===
prompt = PromptTemplate.from_template("""
The following log line contains an error. Summarize the issue and suggest a possible fix.

Log Line: {log_line}
""")

# === 3. LangChain Chain ===
chain = prompt | llm | RunnableLambda(lambda result: print(f"\n🛠️  ERROR Summary & Fix:\n{result.strip()}\n"))

# === 4. File Tailer ===
def tail_file(filepath: str, chain):
    with open(filepath, 'r') as f:
        f.seek(0, 2)  # move to end of file
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            line = line.strip()
            if "ERROR" in line.upper():  # case-insensitive check
                chain.invoke({"log_line": line})

# === 5. Start Monitoring ===
if __name__ == "__main__":
    log_path = "//path-to-file//system.log"
    if not os.path.exists(log_path):
        print(f"❌ File does not exist: {log_path}")
    else:
        print(f"🔍 Monitoring {log_path} for ERROR entries...")
        tail_file(log_path, chain)
