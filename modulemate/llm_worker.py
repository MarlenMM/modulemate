"""Private line-delimited worker protocol; never logs source text or loads code."""

import json
import sys


def task_json(task):
    # Escape literal model role delimiters while preserving the decoded data.
    # Ordinary comparisons and code such as n < 1 remain readable.
    return json.dumps(task, ensure_ascii=False).replace("<|", "\\u003c|")


def main():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from modulemate.structured import GuidedDecoding

    # Small prefill batches and a bounded allocator cache avoid memory spikes
    # on a laptop shared with a browser/IDE. No persistent prompt cache contains
    # a learner's materials or conversation after a request.
    mx.set_cache_limit(128 * 1024 * 1024)
    model, tokenizer = load(sys.argv[1], trust_remote_code=False)
    grammar = GuidedDecoding(tokenizer)

    def complete(request):
        task = dict(request["payload"])
        schema = task.pop("_response_schema", None)
        plain_text = task.pop("_plain_text", False)
        if plain_text:
            schema = None
        processors = []
        if schema:
            processors = [grammar.processor(schema)]

        def render_prompt():
            messages = [
                {"role": "system", "content": request["system"]},
                {
                    "role": "user",
                    "content": "Use these task data to follow the system instructions:\n"
                    + task_json(task)
                    + (
                        "\nUse the requested response format and answer the final request. No reasoning traces."
                        if plain_text
                        else "\nReturn the required JSON object."
                    )
                    + " If repair data are present, correct the identified issues rather than repeating the rejected draft.",
                },
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )

        prompt = render_prompt()
        omitted = 0
        while len(prompt) > 16000 and task.get("recent_conversation"):
            task["recent_conversation"].pop(0)
            omitted += 1
            prompt = render_prompt()
        if len(prompt) > 16000:
            raise ValueError("Context limit")
        parts = []
        final = None
        limit = max(128, min(2400, int(task.get("_max_tokens", 1200))))
        # Reproducible non-greedy sampling follows Qwen's instruct-mode
        # temperature/top-p/top-k. No presence penalty: exact source quotes
        # legitimately repeat words from the prompt.
        mx.random.seed(42)
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=limit,
            sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20),
            logits_processors=processors,
            prefill_step_size=256,
        ):
            parts.append(response.text)
            final = response
        if final is None or final.finish_reason != "stop":
            return {
                "error": "output_limit",
                "stats": {
                    "prompt_tokens": final.prompt_tokens if final else 0,
                    "generated_tokens": final.generation_tokens if final else 0,
                },
            }
        return {
            "text": "".join(parts),
            "stats": {
                "prompt_tokens": final.prompt_tokens,
                "generation_tokens": final.generation_tokens,
                "prompt_tps": final.prompt_tps,
                "generation_tps": final.generation_tps,
                "constrained": bool(processors),
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 42},
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
                "history_turns_omitted": omitted,
            },
        }

    # Per-request frames release prompt, generated text and KV state before
    # waiting for another request. Nothing from a prior turn is implicitly reused.
    while line := sys.stdin.readline():
        try:
            result = complete(json.loads(line))
            print(json.dumps(result, ensure_ascii=False), flush=True)
            del result
        except Exception:
            print('{"error":"generation_failed"}', flush=True)
        finally:
            del line
            mx.clear_cache()


if __name__ == "__main__":
    main()
