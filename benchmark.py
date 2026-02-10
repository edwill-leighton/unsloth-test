#!/usr/bin/env python3
"""
Unsloth GPU Benchmark — Synthetic Data Fine-Tuning
===================================================
Plug-and-play benchmark for comparing GPU performance (RTX 5090, DGX Station GB300, RTX PRO 6000).
Generates synthetic instruction-tuning data in-memory — no downloads needed beyond the model.

Usage:
    python benchmark.py                        # Default: 8B model, auto-detect VRAM
    python benchmark.py --profile small         # ~8 GB VRAM  (3B model)
    python benchmark.py --profile medium        # ~16 GB VRAM (8B model)
    python benchmark.py --profile large         # ~24 GB VRAM (8B model, long ctx)
    python benchmark.py --profile xlarge        # ~40 GB VRAM (32B model)
    python benchmark.py --profile stress        # ~60 GB VRAM (70B model)
    python benchmark.py --profile monster       # ~150 GB VRAM (70B model, max settings)
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation — no external dataset required
# ──────────────────────────────────────────────────────────────────────────────

TOPICS = [
    "quantum computing", "machine learning", "climate change", "space exploration",
    "renewable energy", "blockchain technology", "gene editing", "autonomous vehicles",
    "cybersecurity", "natural language processing", "robotics", "virtual reality",
    "sustainable agriculture", "nuclear fusion", "ocean conservation",
    "artificial general intelligence", "neuroscience", "cryptography",
    "distributed systems", "compiler optimization", "operating systems",
    "database internals", "network protocols", "signal processing",
    "computer vision", "reinforcement learning", "evolutionary algorithms",
    "bioinformatics", "materials science", "astrophysics",
]

TASK_TEMPLATES = [
    ("Explain the concept of {topic} to a beginner.", "Here is a beginner-friendly explanation of {topic}:\n\n{topic} is a fascinating field that involves understanding and applying principles to solve real-world problems. At its core, {topic} deals with {detail}. The key concepts include foundational theories, practical applications, and ongoing research that continues to push boundaries. For beginners, the most important thing to understand is that {topic} represents a convergence of multiple disciplines working together to advance human knowledge and capability. Researchers in this field work on developing new methodologies, testing hypotheses, and building systems that can be applied in various domains. The future of {topic} looks promising with continued investment and innovation driving progress forward."),
    ("Write a detailed technical overview of {topic}.", "Technical Overview of {topic}:\n\n1. Introduction: {topic} encompasses a broad range of techniques and methodologies aimed at {detail}. 2. Core Principles: The fundamental principles underlying {topic} include mathematical frameworks, empirical observations, and computational models. 3. Current State of the Art: Recent advances in {topic} have led to significant breakthroughs in performance, efficiency, and scalability. 4. Challenges: Despite progress, {topic} faces challenges including computational complexity, data requirements, and theoretical limitations. 5. Future Directions: The field is moving toward more integrated approaches that combine {topic} with complementary disciplines to achieve greater impact."),
    ("What are the main challenges in {topic} and how can they be addressed?", "The main challenges in {topic} include:\n\n1. Scalability: As systems grow in complexity, {detail} becomes increasingly difficult to manage. Solutions include distributed architectures and hierarchical approaches. 2. Efficiency: Current methods in {topic} often require significant computational resources. Optimization techniques and hardware acceleration can help address this. 3. Reliability: Ensuring consistent and dependable results in {topic} remains an ongoing challenge. Rigorous testing, formal verification, and redundancy are key strategies. 4. Integration: Combining {topic} with existing systems and workflows requires careful design and standardization efforts. 5. Ethics and Safety: As {topic} becomes more prevalent, addressing ethical considerations and safety concerns is paramount."),
    ("Describe a practical application of {topic} in industry.", "A practical application of {topic} in industry:\n\n{topic} has found significant practical applications across multiple industries. In manufacturing, {detail} enables more efficient production processes and quality control. In healthcare, {topic} contributes to improved diagnostics, treatment planning, and patient outcomes. The financial sector leverages {topic} for risk assessment, fraud detection, and algorithmic trading. Transportation and logistics benefit from {topic} through route optimization, predictive maintenance, and autonomous systems. Energy companies apply {topic} to optimize grid management, predict demand, and improve renewable energy integration. These applications demonstrate the versatility and value of {topic} across diverse industrial contexts."),
    ("Compare and contrast different approaches to {topic}.", "Comparison of approaches to {topic}:\n\nThere are several distinct approaches to {topic}, each with unique strengths and limitations. The classical approach focuses on {detail} using well-established theoretical frameworks. Modern approaches leverage computational power and data-driven methods to achieve results that were previously impossible. Hybrid approaches combine elements of both classical and modern methods, often achieving the best of both worlds. When comparing these approaches, key factors include accuracy, computational cost, scalability, interpretability, and ease of implementation. The choice of approach depends heavily on the specific use case, available resources, and desired outcomes. Recent trends show a convergence of approaches, with practitioners increasingly combining multiple methodologies."),
    ("Write a research proposal outline for advancing {topic}.", "Research Proposal: Advancing {topic}\n\nAbstract: This proposal outlines a comprehensive research program aimed at advancing the state of the art in {topic}. Background: {topic} has seen significant progress in recent years, with {detail} emerging as a key area of focus. The proposed research builds on existing work while addressing critical gaps in current understanding. Objectives: 1) Develop novel algorithms and methodologies for {topic}. 2) Create benchmark datasets and evaluation frameworks. 3) Demonstrate practical applications in real-world settings. Methodology: The research will employ a combination of theoretical analysis, computational experiments, and empirical validation. Expected Outcomes: We anticipate significant improvements in performance metrics, new theoretical insights, and practical tools that can be adopted by the broader community. Timeline: The proposed research spans 36 months with clearly defined milestones and deliverables."),
    ("Create a tutorial on getting started with {topic}.", "Getting Started with {topic} — A Step-by-Step Tutorial\n\nPrerequisites: Before diving into {topic}, you should have a basic understanding of {detail} and related concepts. Step 1: Set up your environment by installing the necessary tools and libraries. Step 2: Familiarize yourself with the fundamental concepts and terminology used in {topic}. Step 3: Work through basic examples to build intuition and practical skills. Step 4: Explore intermediate topics including optimization, best practices, and common pitfalls. Step 5: Apply your knowledge to a real-world project. Step 6: Engage with the community through forums, conferences, and open-source contributions. Resources: Recommended textbooks, online courses, and documentation for further learning. Remember that mastering {topic} is a journey that requires patience, practice, and continuous learning."),
    ("Discuss the future trends and predictions for {topic}.", "Future Trends and Predictions for {topic}:\n\nThe landscape of {topic} is evolving rapidly, with several key trends shaping its future. First, {detail} is expected to play an increasingly important role in driving innovation. Second, the democratization of tools and resources is making {topic} accessible to a broader audience. Third, interdisciplinary collaboration is accelerating progress by bringing diverse perspectives and expertise. Fourth, ethical considerations and responsible development practices are becoming central to the field. Fifth, commercial applications are expanding, creating new markets and opportunities. Looking ahead 5-10 years, we can expect {topic} to become more integrated into everyday life, with significant implications for society, economy, and governance. The pace of advancement shows no signs of slowing, making this an exciting time for practitioners and researchers alike."),
]

DETAILS = [
    "the systematic analysis and optimization of complex systems",
    "developing novel algorithms that can process information more efficiently",
    "understanding the fundamental principles that govern natural phenomena",
    "creating scalable solutions for increasingly complex problems",
    "bridging the gap between theoretical models and practical implementations",
    "leveraging computational resources to simulate and predict outcomes",
    "integrating multiple data sources to derive actionable insights",
    "building robust frameworks that can adapt to changing requirements",
    "establishing standardized methodologies for reproducible results",
    "pushing the boundaries of what is computationally feasible",
]


def generate_synthetic_dataset(num_samples: int, max_seq_length: int) -> list[dict]:
    """Generate synthetic instruction-following data in Alpaca format."""
    random.seed(42)
    dataset = []

    for i in range(num_samples):
        topic = random.choice(TOPICS)
        template_instruction, template_output = random.choice(TASK_TEMPLATES)
        detail = random.choice(DETAILS)

        instruction = template_instruction.format(topic=topic)
        output = template_output.format(topic=topic, detail=detail)

        # Pad output to push sequence length / VRAM usage
        # Repeat the output to fill up to roughly max_seq_length tokens (~4 chars per token)
        target_chars = max_seq_length * 3  # conservative estimate
        while len(output) < target_chars:
            output += f"\n\nFurthermore, regarding {topic}, {random.choice(DETAILS)}. "
            output += f"This aspect of {topic} is particularly relevant because it connects to broader themes in the field. "
            output += f"Continued research and development in {topic} will be essential for addressing future challenges and opportunities."

        dataset.append({
            "instruction": instruction,
            "input": "",
            "output": output[:target_chars],
        })

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Training profiles — tuned to push VRAM on different GPU tiers
# ──────────────────────────────────────────────────────────────────────────────

PROFILES = {
    "small": {
        "description": "~8 GB VRAM (RTX 3070/4060 class)",
        "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lora_rank": 32,
        "num_samples": 1000,
        "max_steps": 60,
        "learning_rate": 2e-4,
    },
    "medium": {
        "description": "~16 GB VRAM (RTX 4080/5070 Ti class)",
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "max_seq_length": 4096,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lora_rank": 64,
        "num_samples": 2000,
        "max_steps": 60,
        "learning_rate": 2e-4,
    },
    "large": {
        "description": "~24 GB VRAM (RTX 4090/5090 class)",
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "max_seq_length": 8192,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lora_rank": 128,
        "num_samples": 3000,
        "max_steps": 60,
        "learning_rate": 2e-4,
    },
    "xlarge": {
        "description": "~40 GB VRAM (RTX PRO 6000 / DGX Station class)",
        "model_name": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "max_seq_length": 8192,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lora_rank": 128,
        "num_samples": 3000,
        "max_steps": 60,
        "learning_rate": 1e-4,
    },
    "stress": {
        "description": "~60 GB VRAM (A100 80GB / multi-GPU class)",
        "model_name": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "max_seq_length": 4096,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_rank": 64,
        "num_samples": 2000,
        "max_steps": 60,
        "learning_rate": 1e-4,
    },
    "monster": {
        "description": "~150 GB VRAM (DGX Station GB300 / 288 GB HBM3e class)",
        "model_name": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "max_seq_length": 8192,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lora_rank": 128,
        "num_samples": 5000,
        "max_steps": 60,
        "learning_rate": 1e-4,
    },
}


def format_alpaca_prompt(sample: dict) -> str:
    """Format a sample into the Alpaca prompt template."""
    if sample["input"]:
        return (
            "Below is an instruction that describes a task, paired with further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Input:\n{sample['input']}\n\n"
            f"### Response:\n{sample['output']}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['output']}"
        )


EOS_TOKEN = "</s>"  # Will be replaced with actual tokenizer EOS


def formatting_prompts_func(examples):
    """Batch formatting function for the SFT trainer."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        sample = {"instruction": instruction, "input": inp, "output": output}
        text = format_alpaca_prompt(sample) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# ──────────────────────────────────────────────────────────────────────────────
# GPU info & monitoring
# ──────────────────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Collect GPU information for the benchmark report."""
    info = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "driver_version": "N/A",
    }
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return info


def get_vram_usage() -> dict:
    """Get current VRAM usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(profile_name: str):
    profile = PROFILES[profile_name]
    print("=" * 70)
    print(f"  UNSLOTH GPU BENCHMARK — Profile: {profile_name.upper()}")
    print(f"  {profile['description']}")
    print("=" * 70)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\n  GPU:           {gpu_info['gpu_name']}")
    print(f"  GPU Count:     {gpu_info['gpu_count']}")
    print(f"  VRAM Total:    {gpu_info['vram_total_gb']} GB")
    print(f"  CUDA:          {gpu_info['cuda_version']}")
    print(f"  PyTorch:       {gpu_info['torch_version']}")
    print(f"  Driver:        {gpu_info['driver_version']}")
    print(f"  Model:         {profile['model_name']}")
    print(f"  Seq Length:    {profile['max_seq_length']}")
    print(f"  Batch Size:    {profile['batch_size']}")
    print(f"  Grad Accum:    {profile['gradient_accumulation_steps']}")
    print(f"  LoRA Rank:     {profile['lora_rank']}")
    print(f"  Max Steps:     {profile['max_steps']}")
    print(f"  Num Samples:   {profile['num_samples']}")
    print("=" * 70)

    # ── Step 1: Generate synthetic data ──────────────────────────────────
    print("\n[1/5] Generating synthetic dataset...")
    t0 = time.time()
    raw_data = generate_synthetic_dataset(profile["num_samples"], profile["max_seq_length"])
    data_gen_time = time.time() - t0
    print(f"       Generated {len(raw_data)} samples in {data_gen_time:.1f}s")

    # ── Step 2: Load model ───────────────────────────────────────────────
    print("\n[2/5] Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=profile["model_name"],
        max_seq_length=profile["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )
    model_load_time = time.time() - t0
    vram_after_load = get_vram_usage()
    print(f"       Model loaded in {model_load_time:.1f}s")
    print(f"       VRAM after load: {vram_after_load['allocated_gb']} GB allocated, {vram_after_load['reserved_gb']} GB reserved")

    # ── Step 3: Apply LoRA ───────────────────────────────────────────────
    print("\n[3/5] Applying LoRA adapters...")
    t0 = time.time()
    model = FastLanguageModel.get_peft_model(
        model,
        r=profile["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=profile["lora_rank"],  # alpha = rank is a common default
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # long context support
        random_state=42,
    )
    lora_time = time.time() - t0
    vram_after_lora = get_vram_usage()
    print(f"       LoRA applied in {lora_time:.1f}s")
    print(f"       VRAM after LoRA: {vram_after_lora['allocated_gb']} GB allocated")

    # ── Step 4: Prepare dataset ──────────────────────────────────────────
    print("\n[4/5] Preparing dataset for training...")
    from datasets import Dataset

    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token or "</s>"

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print(f"       Dataset ready: {len(dataset)} samples")

    # ── Step 5: Train ────────────────────────────────────────────────────
    print("\n[5/5] Starting training...")
    print("-" * 70)

    from trl import SFTTrainer
    from transformers import TrainingArguments

    torch.cuda.reset_peak_memory_stats()
    vram_before_train = get_vram_usage()

    training_args = TrainingArguments(
        per_device_train_batch_size=profile["batch_size"],
        gradient_accumulation_steps=profile["gradient_accumulation_steps"],
        max_steps=profile["max_steps"],
        learning_rate=profile["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="/tmp/unsloth_benchmark_output",
        warmup_steps=5,
        report_to="none",  # no wandb/tensorboard needed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=profile["max_seq_length"],
        dataset_num_proc=2,
        packing=True,  # pack short sequences together for efficiency
        args=training_args,
    )

    t_train_start = time.time()
    train_result = trainer.train()
    t_train_end = time.time()

    training_time = t_train_end - t_train_start
    vram_after_train = get_vram_usage()

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    tokens_per_second = 0
    total_tokens = 0
    if hasattr(train_result, "metrics"):
        metrics = train_result.metrics
        total_tokens = metrics.get("train_total_flos", 0)
        if "train_samples_per_second" in metrics:
            samples_per_sec = metrics["train_samples_per_second"]
        else:
            samples_per_sec = profile["max_steps"] * profile["batch_size"] / training_time

        # Estimate tokens/sec from training throughput
        effective_batch = profile["batch_size"] * profile["gradient_accumulation_steps"]
        tokens_per_step = effective_batch * profile["max_seq_length"]
        tokens_per_second = (profile["max_steps"] * tokens_per_step) / training_time
    else:
        samples_per_sec = profile["max_steps"] * profile["batch_size"] / training_time
        effective_batch = profile["batch_size"] * profile["gradient_accumulation_steps"]
        tokens_per_step = effective_batch * profile["max_seq_length"]
        tokens_per_second = (profile["max_steps"] * tokens_per_step) / training_time

    train_loss = metrics.get("train_loss", "N/A") if hasattr(train_result, "metrics") else "N/A"

    results = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile_name,
        "gpu": gpu_info,
        "config": {
            "model": profile["model_name"],
            "max_seq_length": profile["max_seq_length"],
            "batch_size": profile["batch_size"],
            "gradient_accumulation_steps": profile["gradient_accumulation_steps"],
            "lora_rank": profile["lora_rank"],
            "max_steps": profile["max_steps"],
            "num_samples": profile["num_samples"],
        },
        "timing": {
            "data_generation_sec": round(data_gen_time, 2),
            "model_load_sec": round(model_load_time, 2),
            "lora_setup_sec": round(lora_time, 2),
            "training_sec": round(training_time, 2),
            "total_sec": round(data_gen_time + model_load_time + lora_time + training_time, 2),
        },
        "throughput": {
            "samples_per_second": round(samples_per_sec, 2),
            "tokens_per_second": round(tokens_per_second, 0),
            "steps_per_second": round(profile["max_steps"] / training_time, 3),
        },
        "vram": {
            "after_model_load_gb": vram_after_load["allocated_gb"],
            "after_lora_gb": vram_after_lora["allocated_gb"],
            "peak_training_gb": vram_after_train["peak_gb"],
            "total_gpu_vram_gb": gpu_info["vram_total_gb"],
            "utilization_pct": round(vram_after_train["peak_gb"] / gpu_info["vram_total_gb"] * 100, 1) if gpu_info["vram_total_gb"] > 0 else 0,
        },
        "training": {
            "final_loss": train_loss,
        },
    }

    print(f"\n  GPU:                  {gpu_info['gpu_name']}")
    print(f"  Profile:              {profile_name}")
    print(f"  Model:                {profile['model_name']}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Model Load Time:      {model_load_time:.1f}s")
    print(f"  Training Time:        {training_time:.1f}s")
    print(f"  Total Time:           {results['timing']['total_sec']:.1f}s")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Samples/sec:          {results['throughput']['samples_per_second']:.2f}")
    print(f"  Tokens/sec:           {results['throughput']['tokens_per_second']:.0f}")
    print(f"  Steps/sec:            {results['throughput']['steps_per_second']:.3f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  VRAM after load:      {vram_after_load['allocated_gb']} GB")
    print(f"  VRAM peak training:   {vram_after_train['peak_gb']} GB")
    print(f"  VRAM utilization:     {results['vram']['utilization_pct']}%")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Final Loss:           {train_loss}")
    print("=" * 70)

    # Save results to JSON for comparison
    results_dir = "/workspace/work" if os.path.isdir("/workspace/work") else "."
    gpu_slug = gpu_info["gpu_name"].replace(" ", "_").replace("/", "-")
    results_file = os.path.join(results_dir, f"benchmark_{gpu_slug}_{profile_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")
    print(f"  Copy this file to compare across GPUs!\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unsloth GPU Benchmark — Compare GPU training performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  small    ~8 GB VRAM    Llama 3.2 3B,  seq=2048, batch=2, rank=32
  medium   ~16 GB VRAM   Llama 3.1 8B,  seq=4096, batch=2, rank=64
  large    ~24 GB VRAM   Llama 3.1 8B,  seq=8192, batch=4, rank=128
  xlarge   ~40 GB VRAM   Qwen2.5 32B,   seq=8192, batch=2, rank=128
  stress   ~60 GB VRAM   Llama 3.1 70B, seq=4096, batch=1, rank=64
  monster  ~150 GB VRAM  Llama 3.1 70B, seq=8192, batch=4, rank=128

Examples:
  # Test on RTX 5090 (32 GB) — use 'large' profile
  python benchmark.py --profile large

  # Test on RTX PRO 6000 (48 GB) — use 'xlarge'
  python benchmark.py --profile xlarge

  # Test on DGX Station GB300 (288 GB) — use 'monster'
  python benchmark.py --profile monster

  # Run ALL profiles that fit your GPU
  python benchmark.py --profile all
        """,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="medium",
        choices=list(PROFILES.keys()) + ["all"],
        help="Benchmark profile to run (default: medium)",
    )
    args = parser.parse_args()

    if args.profile == "all":
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Running all profiles that fit in {vram_gb:.0f} GB VRAM...\n")

        vram_thresholds = {
            "small": 8, "medium": 16, "large": 24, "xlarge": 40, "stress": 60, "monster": 150,
        }
        for name, threshold in vram_thresholds.items():
            if vram_gb >= threshold:
                print(f"\n{'#' * 70}")
                print(f"# Running profile: {name}")
                print(f"{'#' * 70}")
                try:
                    run_benchmark(name)
                except Exception as e:
                    print(f"  FAILED: {e}")
                torch.cuda.empty_cache()
            else:
                print(f"\n  Skipping '{name}' — needs ~{threshold} GB, you have {vram_gb:.0f} GB")
    else:
        run_benchmark(args.profile)


if __name__ == "__main__":
    main()
