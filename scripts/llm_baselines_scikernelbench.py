#!/usr/bin/env python3
"""Run official LLM baseline prompts and score with SciKernelBench.

This harness keeps generation separate from evaluation: each baseline uses the
closest public prompt/inference recipe available from its paper, model card, or
official repo, while every generated kernel is evaluated by SciKernelBench.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable

from drkernel_scikernelbench_agent import (
    _json_default,
    add_scikernelbench_to_path,
    extract_kernel_code,
    load_problem,
    parse_eval_payload,
    read_text,
    run_eval_subprocess,
    score_eval,
    write_json,
)


KERNELCODER_PROMPT_TEMPLATE = Template(
    """
You are a Machine Learning Engineer trying to write custom cuda kernels to replace the pytorch operators in the given architecture to get speedups. You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom cuda kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

For [Imports], you will likely need but not limited to the following libraries:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

Here’s an example to show you the syntax of inline embedding custom operators from the cuda kernel in torch:

The pytorch module needed to be optimize is:
```python
$ref_arch_torch
```

The example new arch with custom cuda kernels looks like this:
```python
$ref_arch_kernel
```

And the PyTorch code you need to optimize is:
```python
$code
```

Optimize the architecture named Model with custom cuda kernels! Optimize the architecture named Model with custom cuda kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!

"""
)


AUTOTRITON_PROBLEM_STATEMENT = """You are given a pytorch function, and your task is to write the same triton implementation for it.
The triton implementation should change the name from Model to ModelNew, and have same input and output as the pytorch function."""

AUTOTRITON_PROBLEM_INSTRUCTION = """Optimize the architecture with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no input and init function, no other text, and NO testing code! **Remember to Name your optimized output architecture ModelNew, do not use Model again!**"""


KERNELLLM_PROMPT_TEMPLATE = """
<|begin_of_text|>You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


Here's an example to show you the syntax of inline embedding custom operators from the Triton DSL in torch: The example given architecture is:
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

```
The example new arch with custom Triton kernels looks like this:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)

```

You are given the following architecture:
```
{}
```

Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    model: str
    backend: str
    generator: str
    default_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    prompt_style: str
    prompt_source: str
    sample_count_source: str
    max_tokens_source: str
    sampling_source: str
    chat_template: bool = False
    enable_thinking: bool = False
    trust_remote_code: bool = True
    vllm_max_model_len: int = 32768
    vllm_gpu_memory_utilization: float = 0.90


BASELINES: dict[str, BaselineSpec] = {
    "kernelcoder": BaselineSpec(
        name="kernelcoder",
        model="lkongam/KernelCoder",
        backend="cuda",
        generator="vllm",
        default_samples=10,
        max_new_tokens=16384,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        prompt_style="concur_kernelcoder_hf_model_card_cuda_one_shot",
        prompt_source="https://huggingface.co/lkongam/KernelCoder",
        sample_count_source="ConCuR paper reports pass@10; using 10 samples.",
        max_tokens_source="Assumption: ConCuR trains with max sequence length 32768; use 16384 max generated tokens.",
        sampling_source="Assumption: HF card exposes temperature=1.0 in helper signature but omits SamplingParams.",
        chat_template=True,
        enable_thinking=True,
        vllm_max_model_len=32768,
        vllm_gpu_memory_utilization=0.90,
    ),
    "autotriton": BaselineSpec(
        name="autotriton",
        model="ai9stars/AutoTriton",
        backend="triton",
        generator="vllm",
        default_samples=10,
        max_new_tokens=16384,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        prompt_style="autotriton_paper_figure5_kernelbench_triton",
        prompt_source="AutoTriton paper Figure 5 and https://huggingface.co/ai9stars/AutoTriton",
        sample_count_source="AutoTriton paper Table 3 reports KernelBench pass@10; using 10 samples.",
        max_tokens_source="AutoTriton paper RL setup caps maximum response length at 16384.",
        sampling_source="Assumption: paper/model card do not state evaluation sampling parameters; use temperature=1.0/top_p=1.0 for pass@10 sampling.",
        chat_template=False,
        vllm_max_model_len=32768,
        vllm_gpu_memory_utilization=0.85,
    ),
    "kernelllm": BaselineSpec(
        name="kernelllm",
        model="facebook/KernelLLM",
        backend="triton",
        generator="vllm",
        default_samples=20,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.97,
        top_k=0,
        prompt_style="facebook_kernelllm_official_kernelllm_py_triton_template",
        prompt_source="https://huggingface.co/facebook/KernelLLM/blob/main/kernelllm.py",
        sample_count_source="KernelLLM model card reports Pass@k for k=1,10,20; using strongest reported k=20.",
        max_tokens_source="Official kernelllm.py generate_raw default is 2048 max_new_tokens.",
        sampling_source="KernelLLM model card states inference used temperature=1.0 and top_p=0.97.",
        chat_template=False,
        vllm_max_model_len=8192,
        vllm_gpu_memory_utilization=0.80,
    ),
    "dice": BaselineSpec(
        name="dice",
        model="deadlykitten4/DICE-8B",
        backend="cuda",
        generator="dice_sdar",
        default_samples=4,
        max_new_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        prompt_style="dice_official_kernelbench_cuda_one_shot",
        prompt_source="DICE paper Appendix A.1.2 and DICE/evaluation/src/prompt_constructor.py",
        sample_count_source="DICE paper does not state per-task sample count; using user-specified default of 4. Official DICE README command defaults to 1.",
        max_tokens_source="DICE paper Appendix A.4.4 and README use 4096 generated tokens.",
        sampling_source="DICE paper Appendix A.4.4: SDAR static decoding top_p=1.0, top_k=1, temperature=1.0, block size=4.",
        chat_template=True,
        vllm_max_model_len=8192,
    ),
}


def read_prompt_file(scikernelbench_root: Path, relative_path: str) -> str:
    path = scikernelbench_root / "src" / "kernelbench" / "prompts" / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required prompt resource not found: {path}")
    return path.read_text(encoding="utf-8")


def build_kernelcoder_prompt(reference_code: str, scikernelbench_root: Path) -> str:
    return KERNELCODER_PROMPT_TEMPLATE.substitute(
        ref_arch_torch=read_prompt_file(scikernelbench_root, "model_ex_add.py"),
        ref_arch_kernel=read_prompt_file(scikernelbench_root, "model_new_ex_add.py"),
        code=reference_code,
    )


def build_autotriton_prompt(reference_code: str, scikernelbench_root: Path) -> str:
    del scikernelbench_root
    return f"""{AUTOTRITON_PROBLEM_STATEMENT}
{AUTOTRITON_PROBLEM_INSTRUCTION}
    Now, you need to write the triton implementation for the following pytorch code:
    ```
    {reference_code}
    ```
"""


def build_kernelllm_prompt(reference_code: str, scikernelbench_root: Path) -> str:
    del scikernelbench_root
    return KERNELLLM_PROMPT_TEMPLATE.format(reference_code)


def load_dice_prompt_builder(dice_root: Path) -> Callable[[str], str]:
    evaluation_root = dice_root / "evaluation"
    if not evaluation_root.is_dir():
        raise FileNotFoundError(f"DICE evaluation directory not found: {evaluation_root}")
    example_arch_path = evaluation_root / "src" / "prompts" / "model_ex_add.py"
    example_new_arch_path = evaluation_root / "src" / "prompts" / "model_new_ex_add.py"
    if not example_arch_path.is_file():
        raise FileNotFoundError(f"DICE prompt example architecture not found: {example_arch_path}")
    if not example_new_arch_path.is_file():
        raise FileNotFoundError(f"DICE prompt example kernel not found: {example_new_arch_path}")
    example_arch = example_arch_path.read_text(encoding="utf-8")
    example_new_arch = example_new_arch_path.read_text(encoding="utf-8")

    problem_statement = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
    problem_instruction = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

    def _build(ref_arch_src: str) -> str:
        return (
            problem_statement
            + f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch}
        ``` \n
        """
            + f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """
            + problem_instruction
        )

    return _build


PROMPT_BUILDERS: dict[str, Callable[[str, Path], str]] = {
    "kernelcoder": build_kernelcoder_prompt,
    "autotriton": build_autotriton_prompt,
    "kernelllm": build_kernelllm_prompt,
}


def build_prompt(spec: BaselineSpec, reference_code: str, scikernelbench_root: Path, dice_root: Path | None) -> str:
    if spec.name == "dice":
        if dice_root is None:
            raise ValueError("--dice-root is required for the DICE baseline")
        return load_dice_prompt_builder(dice_root)(reference_code)
    return PROMPT_BUILDERS[spec.name](reference_code, scikernelbench_root)


def render_chat_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    apply_template_code = getattr(tokenizer.apply_chat_template, "__code__", None)
    if apply_template_code is not None and "enable_thinking" in apply_template_code.co_varnames:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_vllm_generator(spec: BaselineSpec, args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    model_name = args.model or spec.model
    llm = LLM(
        model=model_name,
        dtype=args.model_dtype,
        trust_remote_code=spec.trust_remote_code,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization or spec.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len or spec.vllm_max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": args.max_new_tokens or spec.max_new_tokens,
        "temperature": args.temperature if args.temperature is not None else spec.temperature,
        "top_p": args.top_p if args.top_p is not None else spec.top_p,
    }
    top_k = args.top_k if args.top_k is not None else spec.top_k
    if top_k is not None and top_k > 0:
        sampling_kwargs["top_k"] = top_k
    sampling_params = SamplingParams(**sampling_kwargs)
    return tokenizer, llm, sampling_params


def generate_vllm_response(tokenizer: Any, llm: Any, sampling_params: Any, prompt: str, spec: BaselineSpec) -> str:
    rendered = render_chat_prompt(tokenizer, prompt, spec.enable_thinking) if spec.chat_template else prompt
    outputs = llm.generate([rendered], sampling_params=sampling_params, use_tqdm=False)
    if len(outputs) != 1 or not outputs[0].outputs:
        raise RuntimeError("vLLM did not return exactly one generated output")
    return outputs[0].outputs[0].text


def load_dice_generator(spec: BaselineSpec, args: argparse.Namespace, dice_root: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    evaluation_root = dice_root / "evaluation"
    if not evaluation_root.is_dir():
        raise FileNotFoundError(f"DICE evaluation directory not found: {evaluation_root}")
    sys.path.insert(0, str(evaluation_root))
    from scripts.sdar_utils import block_diffusion_generate

    model_name = args.model or spec.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(args.model_device)
    model.eval()
    return tokenizer, model, block_diffusion_generate


def generate_dice_response(tokenizer: Any, model: Any, block_diffusion_generate: Any, prompt: str, args: argparse.Namespace, spec: BaselineSpec) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.batch_encode_plus(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=4096,
    )
    tokens = {key: value.to(args.model_device) for key, value in tokens.items()}
    input_length = tokens["input_ids"].shape[1]
    mask_id = tokenizer(tokenizer.mask_token)["input_ids"][0]
    with torch.no_grad():
        outputs = block_diffusion_generate(
            model,
            prompt=tokens,
            mask_id=mask_id,
            gen_length=args.max_new_tokens or spec.max_new_tokens,
            block_length=4,
            denoising_steps=4,
            temperature=args.temperature if args.temperature is not None else spec.temperature,
            top_k=args.top_k if args.top_k is not None else spec.top_k,
            top_p=args.top_p if args.top_p is not None else spec.top_p,
            remasking_strategy="low_confidence_static",
        )
    generated = outputs[0][input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).replace("<|MASK|>", "")


def rank_record(record: dict[str, Any]) -> tuple[float, float, float]:
    speedup = record.get("speedup")
    speedup_value = float(speedup) if speedup is not None else 0.0
    return (float(bool(record.get("correctness"))), speedup_value, float(bool(record.get("compiled"))))


def false_eval_payload(error: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "compiled": False,
            "correctness": False,
            "runtime": -1.0,
            "ref_runtime": -1.0,
            "metadata": {"generation_error": error, **(metadata or {})},
        },
    }


def run_problem(args: argparse.Namespace) -> int:
    if args.baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {args.baseline!r}. Expected one of {sorted(BASELINES)}")
    spec = BASELINES[args.baseline]
    if args.samples is None:
        args.samples = spec.default_samples

    scikernelbench_root = Path(args.scikernelbench_root).resolve()
    output_root = Path(args.output_dir).resolve()
    dice_root = Path(args.dice_root).resolve() if args.dice_root else None
    add_scikernelbench_to_path(scikernelbench_root)
    problem = load_problem(scikernelbench_root, args.level, args.problem_id)

    problem_dir = output_root / spec.name / f"level_{args.level}" / f"problem_{args.problem_id:03d}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    reference_path = problem_dir / "reference.py"
    reference_path.write_text(problem.code, encoding="utf-8")

    prompt = build_prompt(spec, problem.code, scikernelbench_root, dice_root)
    (problem_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    if spec.generator == "vllm":
        tokenizer, generator, sampling_params = load_vllm_generator(spec, args)
        generate = lambda: generate_vllm_response(tokenizer, generator, sampling_params, prompt, spec)
    elif spec.generator == "dice_sdar":
        if dice_root is None:
            raise ValueError("--dice-root is required for the DICE baseline")
        tokenizer, generator, dice_generate = load_dice_generator(spec, args, dice_root)
        generate = lambda: generate_dice_response(tokenizer, generator, dice_generate, prompt, args, spec)
    else:
        raise ValueError(f"Unknown generator type: {spec.generator}")

    model_name = args.model or spec.model
    run_metadata = {
        "event": "model_loaded",
        "baseline": spec.name,
        "problem_id": args.problem_id,
        "model": model_name,
        "generator": spec.generator,
        "backend": spec.backend,
        "samples": args.samples,
        "prompt_style": spec.prompt_style,
        "prompt_source": spec.prompt_source,
        "sample_count_source": spec.sample_count_source,
        "max_tokens_source": spec.max_tokens_source,
        "sampling_source": spec.sampling_source,
        "assumptions": [
            value
            for value in [spec.max_tokens_source, spec.sampling_source, spec.sample_count_source]
            if value.startswith("Assumption:") or "does not state" in value
        ],
        "config": vars(args),
    }
    print(json.dumps(run_metadata, default=_json_default), flush=True)
    write_json(problem_dir / "run_metadata.json", run_metadata)

    script_path = Path(__file__).resolve()
    sample_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    for sample_id in range(args.samples):
        sample_dir = problem_dir / f"sample_{sample_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        response_path = sample_dir / "response.txt"
        kernel_path = sample_dir / "kernel.py"
        eval_path = sample_dir / "scikernelbench_eval.json"

        try:
            print(json.dumps({"event": "generation_start", "baseline": spec.name, "problem_id": args.problem_id, "sample_id": sample_id}), flush=True)
            response = generate()
            response_path.write_text(response, encoding="utf-8")
            kernel_code = extract_kernel_code(response)
            if not kernel_code:
                kernel_path.write_text("# No kernel code extracted\n", encoding="utf-8")
                eval_payload = false_eval_payload("No Python kernel code block found")
                write_json(eval_path, eval_payload)
                turn_eval = parse_eval_payload(eval_payload)
            else:
                kernel_path.write_text(kernel_code + "\n", encoding="utf-8")
                turn_eval = run_eval_subprocess(
                    script_path=script_path,
                    scikernelbench_root=scikernelbench_root,
                    reference_path=reference_path,
                    kernel_path=kernel_path,
                    output_path=eval_path,
                    build_dir=sample_dir / "build",
                    args=args,
                )
        except Exception as exc:
            response_path.write_text("", encoding="utf-8")
            kernel_path.write_text("# Generation failed before kernel extraction\n", encoding="utf-8")
            eval_payload = false_eval_payload(repr(exc), {"exception_type": type(exc).__name__})
            write_json(eval_path, eval_payload)
            turn_eval = parse_eval_payload(eval_payload)

        record = {
            "sample_id": sample_id,
            "response_path": str(response_path),
            "kernel_path": str(kernel_path),
            "eval_path": str(eval_path),
            "compiled": turn_eval.compiled,
            "correctness": turn_eval.correctness,
            "runtime": turn_eval.runtime,
            "ref_runtime": turn_eval.ref_runtime,
            "speedup": turn_eval.speedup,
            "score": score_eval(turn_eval.compiled, turn_eval.correctness, turn_eval.speedup),
            "metadata": turn_eval.metadata,
            "elapsed_sec": time.time() - started,
        }
        sample_records.append(record)
        if best_record is None or rank_record(record) > rank_record(best_record):
            best_record = record
        print(json.dumps({"event": "sample_done", "baseline": spec.name, "problem_id": args.problem_id, **record}, default=_json_default), flush=True)

    if best_record is None:
        raise RuntimeError("No sample records were produced")

    best_kernel_path = problem_dir / "best_kernel.py"
    best_kernel_path.write_text(read_text(Path(best_record["kernel_path"])), encoding="utf-8")
    summary = {
        "baseline": spec.name,
        "level": args.level,
        "problem_id": args.problem_id,
        "problem_name": problem.name,
        "problem_path": problem.path,
        "model": model_name,
        "backend": spec.backend,
        "samples": sample_records,
        "best": {**best_record, "best_kernel_path": str(best_kernel_path)},
        "num_samples": args.samples,
        "num_compiled": sum(1 for record in sample_records if record["compiled"]),
        "num_correct": sum(1 for record in sample_records if record["correctness"]),
        "num_speedup_gt_1": sum(1 for record in sample_records if record["correctness"] and (record["speedup"] or 0) > 1.0),
        "prompt_style": spec.prompt_style,
        "prompt_source": spec.prompt_source,
        "sample_count_source": spec.sample_count_source,
        "max_tokens_source": spec.max_tokens_source,
        "sampling_source": spec.sampling_source,
        "config": vars(args),
    }
    write_json(problem_dir / "summary.json", summary)
    print(json.dumps({"event": "problem_done", "summary": summary}, default=_json_default), flush=True)
    return 0


def collect(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir).resolve()
    baselines = args.baselines or sorted(BASELINES)
    payload: dict[str, Any] = {"level": args.level, "baselines": {}}
    for baseline in baselines:
        summary_paths = sorted((output_root / baseline / f"level_{args.level}").glob("problem_*/summary.json"))
        records = []
        for summary_path in summary_paths:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary["best"]
            records.append(
                {
                    "problem_id": summary["problem_id"],
                    "problem_name": summary["problem_name"],
                    "num_samples": summary["num_samples"],
                    "num_compiled": summary["num_compiled"],
                    "num_correct": summary["num_correct"],
                    "num_speedup_gt_1": summary["num_speedup_gt_1"],
                    "best_correctness": bool(best["correctness"]),
                    "best_speedup": best["speedup"],
                    "best_runtime": best["runtime"],
                    "best_ref_runtime": best["ref_runtime"],
                    "summary_path": str(summary_path),
                }
            )
        payload["baselines"][baseline] = {
            "num_problem_summaries": len(records),
            "best_correct": sum(1 for record in records if record["best_correctness"]),
            "best_speedup_gt_1": sum(1 for record in records if record["best_correctness"] and (record["best_speedup"] or 0) > 1.0),
            "records": records,
        }
    write_json(output_root / f"level_{args.level}_front_collect_summary.json", payload)
    print(json.dumps(payload, indent=2, default=_json_default))
    return 0


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scikernelbench-root", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-problem")
    add_common_eval_args(run_parser)
    run_parser.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    run_parser.add_argument("--level", type=int, default=3)
    run_parser.add_argument("--problem-id", type=int, required=True)
    run_parser.add_argument("--samples", type=int, default=None)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--dice-root", default=None)
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--model-device", default="cuda:0")
    run_parser.add_argument("--model-dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    run_parser.add_argument("--max-new-tokens", type=int, default=None)
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument("--top-p", type=float, default=None)
    run_parser.add_argument("--top-k", type=int, default=None)
    run_parser.add_argument("--eval-device", default="cuda:0")
    run_parser.add_argument("--eval-visible-index", type=int, default=None)
    run_parser.add_argument("--eval-timeout", type=int, default=1200)
    run_parser.add_argument("--verbose-eval", action="store_true")
    run_parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    run_parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    run_parser.add_argument("--vllm-max-model-len", type=int, default=None)
    run_parser.set_defaults(func=run_problem)

    eval_parser = subparsers.add_parser("eval-one")
    add_common_eval_args(eval_parser)
    eval_parser.add_argument("--reference-path", required=True)
    eval_parser.add_argument("--kernel-path", required=True)
    eval_parser.add_argument("--output-path", required=True)
    eval_parser.add_argument("--build-dir", required=True)
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--verbose", action="store_true")
    from drkernel_scikernelbench_agent import eval_one

    eval_parser.set_defaults(func=eval_one)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument("--level", type=int, default=3)
    collect_parser.add_argument("--baselines", nargs="*", default=None)
    collect_parser.set_defaults(func=collect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run-problem":
        spec = BASELINES[args.baseline]
        args.backend = args.backend or spec.backend
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
