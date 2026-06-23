#!/usr/bin/env python3
"""Run a Dr.Kernel-style multi-turn agent and score with SciKernelBench.

This script intentionally uses SciKernelBench's evaluation code for every
kernel evaluation. KernelGYM/Dr.Kernel code is used only for the released model
and multi-turn agent protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANSWER_BLOCK_RE = re.compile(
    r"(?P<block>```answer[ \t]*(?:\r?\n)?(?P<code>.*?)(?:\r?\n)?```)",
    re.IGNORECASE | re.DOTALL,
)
KERNEL_CODE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in [
        r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=\#\s*End\b|$)",
        r"```python\s*#\s*Kernel\s*\n(.*?)```",
        r"#\s*Your\s+implementation:\s*\n(.*?)(?=\#\s*End\b|$)",
        r"#\s*Generated\s+kernel:\s*\n(.*?)(?=\#\s*End\b|$)",
    ]
]
GENERIC_CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\s*\n?(.*?)```", re.DOTALL)


INITIAL_PROMPT_TEMPLATE = """You are an expert GPU kernel engineer.

Given a PyTorch reference implementation, write an optimized Triton implementation.

Requirements:
- Define a drop-in replacement class named `ModelNew`.
- Preserve the same constructor and forward behavior as the reference `Model`.
- Use Triton kernels where useful.
- Do not change input shapes, output semantics, or random input helper functions.
- Return the implementation as one single ```python``` code block.

# Reference Implementation
```python
{reference_code}
```

Return an optimized Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step."""


FEEDBACK_PROMPT_TEMPLATE = """Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step."""


DEFAULT_DRKERNEL_STOP_TOKEN_IDS = "872,77091,151645,151644"


@dataclass
class TurnEval:
    compiled: bool
    correctness: bool
    runtime: float
    ref_runtime: float
    speedup: float | None
    score: float
    metadata: dict[str, Any]
    raw: dict[str, Any]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseException):
        return repr(obj)
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def add_scikernelbench_to_path(scikernelbench_root: Path) -> None:
    src = scikernelbench_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"SciKernelBench src directory not found: {src}")
    sys.path.insert(0, str(src))


def load_problem(scikernelbench_root: Path, level: int, problem_id: int):
    add_scikernelbench_to_path(scikernelbench_root)
    from kernelbench.dataset import construct_kernelbench_dataset

    dataset = construct_kernelbench_dataset(level=level, source="local")
    return dataset.get_problem_by_id(problem_id)


def extract_kernel_code(response: str) -> str | None:
    answer_match = ANSWER_BLOCK_RE.search(response)
    if answer_match:
        answer_text = answer_match.group("code").strip()
        code_blocks = GENERIC_CODE_BLOCK_RE.findall(answer_text)
        return code_blocks[-1].strip() if code_blocks else answer_text

    for pattern in KERNEL_CODE_PATTERNS:
        match = pattern.search(response)
        if match:
            return match.group(1).strip()

    code_blocks = GENERIC_CODE_BLOCK_RE.findall(response)
    if code_blocks:
        return code_blocks[-1].strip()

    return None


def summarize_metadata(metadata: dict[str, Any], max_chars: int = 4000) -> str:
    if not metadata:
        return ""
    text = json.dumps(metadata, default=_json_default, sort_keys=True)
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def score_eval(compiled: bool, correctness: bool, speedup: float | None) -> float:
    compilation_score = 0.3 if compiled else 0.0
    correctness_score = 0.4 if correctness else 0.0
    performance_score = 0.0
    if correctness and speedup is not None and speedup > 0:
        performance_score = 0.3 * min(speedup, 3.0) / 3.0
    return compilation_score + correctness_score + performance_score


def format_feedback(turn_eval: TurnEval) -> str:
    lines = [
        f"compiled={turn_eval.compiled}",
        f"correctness={turn_eval.correctness}",
        f"kernel_runtime={turn_eval.runtime}",
        f"reference_runtime={turn_eval.ref_runtime}",
        f"speedup={turn_eval.speedup}",
        f"score={turn_eval.score:.6f}",
    ]
    metadata = summarize_metadata(turn_eval.metadata)
    if metadata:
        lines.append(f"metadata={metadata}")
    return "\n".join(lines)


def parse_eval_payload(payload: dict[str, Any]) -> TurnEval:
    raw_result = payload.get("result") or {}
    compiled = bool(raw_result.get("compiled", False))
    correctness = bool(raw_result.get("correctness", False))
    runtime = float(raw_result.get("runtime", -1.0) or -1.0)
    ref_runtime = float(raw_result.get("ref_runtime", -1.0) or -1.0)
    speedup = None
    if correctness and runtime > 0 and ref_runtime > 0:
        speedup = ref_runtime / runtime
    return TurnEval(
        compiled=compiled,
        correctness=correctness,
        runtime=runtime,
        ref_runtime=ref_runtime,
        speedup=speedup,
        score=score_eval(compiled, correctness, speedup),
        metadata=raw_result.get("metadata") or {},
        raw=payload,
    )


def eval_one(args: argparse.Namespace) -> int:
    scikernelbench_root = Path(args.scikernelbench_root).resolve()
    add_scikernelbench_to_path(scikernelbench_root)

    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

    reference_code = read_text(Path(args.reference_path))
    kernel_code = read_text(Path(args.kernel_path))
    device = torch.device(args.device)
    precision = get_torch_dtype_from_string(args.precision)

    started = time.time()
    try:
        result = eval_kernel_against_ref(
            original_model_src=reference_code,
            custom_model_src=kernel_code,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=True,
            timing_method=args.timing_method,
            verbose=args.verbose,
            build_dir=args.build_dir,
            device=device,
            backend=args.backend,
            precision=precision,
        )
        if result is None:
            payload = {
                "ok": False,
                "error": "SciKernelBench returned None, likely a transient compile lock error",
                "elapsed_sec": time.time() - started,
            }
        else:
            if hasattr(result, "model_dump"):
                result_payload = result.model_dump()
            else:
                result_payload = result.dict()
            payload = {"ok": True, "result": result_payload, "elapsed_sec": time.time() - started}
    except Exception as exc:
        payload = {
            "ok": False,
            "error": repr(exc),
            "elapsed_sec": time.time() - started,
        }

    write_json(Path(args.output_path), payload)
    return 0 if payload.get("ok") else 2


def child_eval_env(visible_index: int | None) -> dict[str, str]:
    env = os.environ.copy()
    if visible_index is None:
        return env
    current = env.get("CUDA_VISIBLE_DEVICES")
    if not current:
        raise RuntimeError("--eval-visible-index requires CUDA_VISIBLE_DEVICES to be set")
    devices = [item.strip() for item in current.split(",") if item.strip()]
    if visible_index < 0 or visible_index >= len(devices):
        raise ValueError(f"eval visible index {visible_index} outside CUDA_VISIBLE_DEVICES={current}")
    env["CUDA_VISIBLE_DEVICES"] = devices[visible_index]
    return env


def run_eval_subprocess(
    *,
    script_path: Path,
    scikernelbench_root: Path,
    reference_path: Path,
    kernel_path: Path,
    output_path: Path,
    build_dir: Path,
    args: argparse.Namespace,
) -> TurnEval:
    cmd = [
        sys.executable,
        str(script_path),
        "eval-one",
        "--scikernelbench-root",
        str(scikernelbench_root),
        "--reference-path",
        str(reference_path),
        "--kernel-path",
        str(kernel_path),
        "--output-path",
        str(output_path),
        "--build-dir",
        str(build_dir),
        "--device",
        args.eval_device,
        "--backend",
        args.backend,
        "--precision",
        args.precision,
        "--timing-method",
        args.timing_method,
        "--num-correct-trials",
        str(args.num_correct_trials),
        "--num-perf-trials",
        str(args.num_perf_trials),
    ]
    if args.verbose_eval:
        cmd.append("--verbose")
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            env=child_eval_env(args.eval_visible_index),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.eval_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        if stdout:
            (output_path.parent / (output_path.stem + ".stdout.txt")).write_text(stdout, encoding="utf-8")
        if stderr:
            (output_path.parent / (output_path.stem + ".stderr.txt")).write_text(stderr, encoding="utf-8")
        payload = {
            "ok": False,
            "error": f"eval subprocess timed out after {args.eval_timeout} seconds",
            "elapsed_sec": time.time() - started,
            "result": {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "ref_runtime": -1.0,
                "metadata": {
                    "eval_timeout_sec": args.eval_timeout,
                    "timeout_command": cmd,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                },
            },
        }
        write_json(output_path, payload)
        return parse_eval_payload(payload)
    if completed.stdout:
        (output_path.parent / (output_path.stem + ".stdout.txt")).write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        (output_path.parent / (output_path.stem + ".stderr.txt")).write_text(completed.stderr, encoding="utf-8")
    if not output_path.exists():
        write_json(
            output_path,
            {
                "ok": False,
                "error": f"eval subprocess did not write output, returncode={completed.returncode}",
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            },
        )
    return parse_eval_payload(json.loads(output_path.read_text(encoding="utf-8")))


def make_initial_messages(reference_code: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": INITIAL_PROMPT_TEMPLATE.format(reference_code=reference_code)}]


def append_feedback(messages: list[dict[str, str]], response: str, feedback: str) -> None:
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": FEEDBACK_PROMPT_TEMPLATE.format(feedback=feedback)})


def render_messages(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    apply_template_code = getattr(tokenizer.apply_chat_template, "__code__", None)
    if apply_template_code is not None and "enable_thinking" in apply_template_code.co_varnames:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def parse_stop_token_ids(raw_ids: str, tokenizer: Any) -> list[int]:
    stop_ids: list[int] = []
    if raw_ids:
        for raw_id in raw_ids.split(","):
            item = raw_id.strip()
            if item:
                stop_ids.append(int(item))
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    if not stop_ids:
        raise ValueError("No stop token ids configured and tokenizer has no eos_token_id")
    return sorted(set(stop_ids))


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(args.model_device)
    model.eval()
    return tokenizer, model


def load_generator(args: argparse.Namespace):
    if args.generation_backend == "transformers":
        return (*load_model(args), None)
    if args.generation_backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype=args.model_dtype,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )
        tokenizer = llm.get_tokenizer()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=args.top_p,
            stop_token_ids=parse_stop_token_ids(args.stop_token_ids, tokenizer),
        )
        return tokenizer, llm, sampling_params
    raise ValueError(f"Unknown generation backend: {args.generation_backend}")


def generate_response(tokenizer: Any, model: Any, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    import torch

    prompt = render_messages(tokenizer, messages, args.enable_thinking)
    inputs = tokenizer([prompt], return_tensors="pt").to(args.model_device)
    stop_token_ids = parse_stop_token_ids(args.stop_token_ids, tokenizer)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_response_vllm(
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    prompt = render_messages(tokenizer, messages, args.enable_thinking)
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    if len(outputs) != 1 or not outputs[0].outputs:
        raise RuntimeError("vLLM did not return exactly one generated output")
    return outputs[0].outputs[0].text


def generate_agent_response(
    tokenizer: Any,
    generator: Any,
    sampling_params: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    if args.generation_backend == "transformers":
        return generate_response(tokenizer, generator, messages, args)
    if args.generation_backend == "vllm":
        return generate_response_vllm(tokenizer, generator, sampling_params, messages, args)
    raise ValueError(f"Unknown generation backend: {args.generation_backend}")


def run_agent(args: argparse.Namespace) -> int:
    scikernelbench_root = Path(args.scikernelbench_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    problem = load_problem(scikernelbench_root, args.level, args.problem_id)

    script_path = Path(__file__).resolve()
    problem_dir = output_dir / f"level_{args.level}" / f"problem_{args.problem_id:03d}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    reference_path = problem_dir / "reference.py"
    reference_path.write_text(problem.code, encoding="utf-8")

    tokenizer, generator, sampling_params = load_generator(args)
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "problem_id": args.problem_id,
                "model": args.model,
                "generation_backend": args.generation_backend,
                "model_device": args.model_device,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": parse_stop_token_ids(args.stop_token_ids, tokenizer),
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "vllm_max_model_len": args.vllm_max_model_len,
            },
            default=_json_default,
        ),
        flush=True,
    )

    all_samples = []
    best_record: dict[str, Any] | None = None
    for sample_id in range(args.samples):
        sample_dir = problem_dir / f"sample_{sample_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        messages = make_initial_messages(problem.code)
        turn_records = []

        for turn_id in range(1, args.max_turns + 1):
            print(
                json.dumps(
                    {
                        "event": "generation_start",
                        "problem_id": args.problem_id,
                        "sample_id": sample_id,
                        "turn_id": turn_id,
                    }
                ),
                flush=True,
            )
            response = generate_agent_response(tokenizer, generator, sampling_params, messages, args)
            print(
                json.dumps(
                    {
                        "event": "generation_done",
                        "problem_id": args.problem_id,
                        "sample_id": sample_id,
                        "turn_id": turn_id,
                        "response_chars": len(response),
                    }
                ),
                flush=True,
            )
            response_path = sample_dir / f"turn_{turn_id:02d}_response.txt"
            response_path.write_text(response, encoding="utf-8")

            kernel_code = extract_kernel_code(response)
            if not kernel_code:
                eval_payload = {
                    "ok": True,
                    "result": {
                        "compiled": False,
                        "correctness": False,
                        "runtime": -1.0,
                        "ref_runtime": -1.0,
                        "metadata": {"extraction_error": "No Python kernel code block found"},
                    },
                }
                eval_path = sample_dir / f"turn_{turn_id:02d}_scikernelbench_eval.json"
                write_json(eval_path, eval_payload)
                turn_eval = parse_eval_payload(eval_payload)
                kernel_path = sample_dir / f"turn_{turn_id:02d}_kernel.py"
                kernel_path.write_text("# No kernel code extracted\n", encoding="utf-8")
            else:
                kernel_path = sample_dir / f"turn_{turn_id:02d}_kernel.py"
                kernel_path.write_text(kernel_code + "\n", encoding="utf-8")
                eval_path = sample_dir / f"turn_{turn_id:02d}_scikernelbench_eval.json"
                print(
                    json.dumps(
                        {
                            "event": "eval_start",
                            "problem_id": args.problem_id,
                            "sample_id": sample_id,
                            "turn_id": turn_id,
                            "kernel_chars": len(kernel_code),
                        }
                    ),
                    flush=True,
                )
                turn_eval = run_eval_subprocess(
                    script_path=script_path,
                    scikernelbench_root=scikernelbench_root,
                    reference_path=reference_path,
                    kernel_path=kernel_path,
                    output_path=eval_path,
                    build_dir=sample_dir / f"turn_{turn_id:02d}_build",
                    args=args,
                )
            print(
                json.dumps(
                    {
                        "event": "eval_done",
                        "problem_id": args.problem_id,
                        "sample_id": sample_id,
                        "turn_id": turn_id,
                        "compiled": turn_eval.compiled,
                        "correctness": turn_eval.correctness,
                        "runtime": turn_eval.runtime,
                        "ref_runtime": turn_eval.ref_runtime,
                        "speedup": turn_eval.speedup,
                        "score": turn_eval.score,
                    },
                    default=_json_default,
                ),
                flush=True,
            )

            record = {
                "sample_id": sample_id,
                "turn_id": turn_id,
                "response_path": str(response_path),
                "kernel_path": str(kernel_path),
                "eval_path": str(eval_path),
                "compiled": turn_eval.compiled,
                "correctness": turn_eval.correctness,
                "runtime": turn_eval.runtime,
                "ref_runtime": turn_eval.ref_runtime,
                "speedup": turn_eval.speedup,
                "score": turn_eval.score,
                "metadata": turn_eval.metadata,
            }
            turn_records.append(record)
            if best_record is None or record["score"] > best_record["score"]:
                best_record = record

            append_feedback(messages, response, format_feedback(turn_eval))
            write_json(sample_dir / "conversation.json", messages)

        all_samples.append({"sample_id": sample_id, "turns": turn_records})

    if best_record is None:
        raise RuntimeError("No turn records were produced")

    best_kernel_path = problem_dir / "best_kernel.py"
    best_kernel_path.write_text(read_text(Path(best_record["kernel_path"])), encoding="utf-8")
    summary = {
        "level": args.level,
        "problem_id": args.problem_id,
        "problem_name": problem.name,
        "problem_path": problem.path,
        "model": args.model,
        "samples": all_samples,
        "best": {**best_record, "best_kernel_path": str(best_kernel_path)},
        "config": vars(args),
    }
    write_json(problem_dir / "summary.json", summary)
    print(json.dumps({"problem_id": args.problem_id, "best": summary["best"]}, default=_json_default))
    return 0


def collect(args: argparse.Namespace) -> int:
    scikernelbench_root = Path(args.scikernelbench_root).resolve()
    agent_output_dir = Path(args.agent_output_dir).resolve()
    add_scikernelbench_to_path(scikernelbench_root)
    from kernelbench.dataset import construct_kernelbench_dataset

    dataset = construct_kernelbench_dataset(level=args.level, source="local")
    summaries = sorted((agent_output_dir / f"level_{args.level}").glob("problem_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summaries found under {agent_output_dir}/level_{args.level}")

    eval_results: dict[str, list[dict[str, Any]]] = {}
    baseline = {f"level{args.level}": {}}
    records = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pid = int(summary["problem_id"])
        best = summary["best"]
        eval_entry = {
            "sample_id": 0,
            "compiled": bool(best["compiled"]),
            "correctness": bool(best["correctness"]),
            "metadata": best.get("metadata") or {},
            "runtime": float(best.get("runtime", -1.0) or -1.0),
            "runtime_stats": {},
        }
        eval_results[str(pid)] = [eval_entry]
        problem = dataset.get_problem_by_id(pid)
        ref_runtime = float(best.get("ref_runtime", -1.0) or -1.0)
        if ref_runtime > 0:
            baseline[f"level{args.level}"][problem.name] = {"mean": ref_runtime}
        records.append(
            {
                "problem_id": pid,
                "problem_name": problem.name,
                "compiled": eval_entry["compiled"],
                "correctness": eval_entry["correctness"],
                "runtime": eval_entry["runtime"],
                "ref_runtime": ref_runtime,
                "speedup": best.get("speedup"),
                "score": best.get("score"),
                "best_kernel_path": best.get("best_kernel_path"),
            }
        )

    run_dir = scikernelbench_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "eval_results.json", eval_results)
    write_json(run_dir / "drkernel_scikernelbench_records.json", records)

    timing_dir = scikernelbench_root / "results" / "timing" / args.hardware_name
    timing_dir.mkdir(parents=True, exist_ok=True)
    write_json(timing_dir / f"{args.baseline_name}.json", baseline)
    write_json(agent_output_dir / f"level_{args.level}" / "collection_summary.json", {
        "run_name": args.run_name,
        "hardware_name": args.hardware_name,
        "baseline_name": args.baseline_name,
        "num_results": len(eval_results),
        "num_baselines": len(baseline[f"level{args.level}"]),
    })
    print(f"Wrote {run_dir / 'eval_results.json'}")
    print(f"Wrote {timing_dir / (args.baseline_name + '.json')}")
    return 0


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scikernelbench-root", required=True)
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--timing-method", default="cuda_event")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-agent")
    add_common_eval_args(run_parser)
    run_parser.add_argument("--model", default="hkust-nlp/drkernel-14b")
    run_parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    run_parser.add_argument("--model-device", default="cuda:0")
    run_parser.add_argument("--model-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    run_parser.add_argument("--trust-remote-code", action="store_true")
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument("--level", type=int, default=3)
    run_parser.add_argument("--problem-id", type=int, required=True)
    run_parser.add_argument("--samples", type=int, default=8)
    run_parser.add_argument("--max-turns", type=int, default=3)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--max-new-tokens", type=int, default=8192)
    run_parser.add_argument("--stop-token-ids", default=DEFAULT_DRKERNEL_STOP_TOKEN_IDS)
    run_parser.add_argument("--temperature", type=float, default=1.0)
    run_parser.add_argument("--top-p", type=float, default=0.95)
    run_parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--eval-device", default="cuda:0")
    run_parser.add_argument("--eval-visible-index", type=int, default=None)
    run_parser.add_argument("--eval-timeout", type=int, default=900)
    run_parser.add_argument("--verbose-eval", action="store_true")
    run_parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    run_parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    run_parser.add_argument("--vllm-max-model-len", type=int, default=28672)
    run_parser.set_defaults(func=run_agent)

    eval_parser = subparsers.add_parser("eval-one")
    add_common_eval_args(eval_parser)
    eval_parser.add_argument("--reference-path", required=True)
    eval_parser.add_argument("--kernel-path", required=True)
    eval_parser.add_argument("--output-path", required=True)
    eval_parser.add_argument("--build-dir", required=True)
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--verbose", action="store_true")
    eval_parser.set_defaults(func=eval_one)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--scikernelbench-root", required=True)
    collect_parser.add_argument("--agent-output-dir", required=True)
    collect_parser.add_argument("--level", type=int, default=3)
    collect_parser.add_argument("--run-name", required=True)
    collect_parser.add_argument("--hardware-name", default="Pudong_DrKernel_SciKernelBench")
    collect_parser.add_argument("--baseline-name", default="baseline_time_torch")
    collect_parser.set_defaults(func=collect)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
