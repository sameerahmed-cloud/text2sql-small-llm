# text2sql-small-llm

Teaching a small open LLM (~1.5B parameters) to turn plain-English questions into SQL,
using supervised fine-tuning (SFT) and reinforcement learning (GRPO) on free GPUs,
then making it fast and cheap to serve — even on a CPU.

**Status:** 🚧 In progress: setting up.

## Results
_Coming soon: execution accuracy, latency and cost for every model version._

## Tech
Unsloth · Hugging Face TRL/PEFT · vLLM · llama.cpp · Weights & Biases · Gradio

## Benchmarks
BIRD (main) and Spider, evaluated on databases never seen in training.