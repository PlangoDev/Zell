#!/usr/bin/env python3
"""Zell: synthetic chat + tool-call corpus generator (OpenRouter).

Generates a large, diverse instruction/chat/tool-calling corpus with a cheap,
fast teacher model and writes it as JSONL ({"messages": [...]}) that
tools/build_blend.py folds into the mixed pretrain blend as one weighted source.

The point: bake chatting, reasoning, coding help, and valid tool-call formatting
into the core from the first training step, rather than bolting them on later.

Tool-call turns are emitted in the Qwen2.5-native Hermes style: an assistant
turn whose content contains a single-line-JSON <tool_call>{...}</tool_call>
block, followed by a "tool"-role turn with the result, then the assistant's
natural-language answer. This matches the core's chat template.

Auth: set OPENROUTER_API_KEY in the environment.
    export OPENROUTER_API_KEY=sk-or-...
    python generate.py --target-tokens 50000000 --out-dir synth/out \
        --model "inclusionai/ling-2.6-flash" --concurrency 8

Teacher options (OpenRouter ids, verified pricing per 1M tok prompt/completion):
    inclusionai/ling-2.6-flash            $0.010/$0.030  cheapest smart MoE; ~$1.65 for a full 50M run (default)
    deepseek/deepseek-v4-flash            $0.090/$0.180  strong reasoning/JSON; ~$10 for 50M
    qwen/qwen3.5-flash-02-23              $0.065/$0.260  fast, long ctx
Genuinely free (rate-limited hard; use --concurrency 2-4, expect retries):
    qwen/qwen3-next-80b-a3b-instruct:free
    meta-llama/llama-3.3-70b-instruct:free
    qwen/qwen3-coder:free                 (best free option for the coding/tool slices)
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---- generation taxonomy: category -> (weight, builder) -----------------------
TOPICS = [
    "everyday life and routines", "science and how things work", "history and culture",
    "technology and the internet", "health, food, and fitness", "travel and geography",
    "books, film, and music", "money, work, and careers", "relationships and emotions",
    "philosophy and ethics", "nature and animals", "sports and games",
    "current-affairs-style explainers (timeless framing)", "DIY and practical skills",
    "language, writing, and grammar", "education and learning",
]
CODING = [
    "debugging a Python traceback", "writing a small algorithm", "explaining a data structure",
    "refactoring messy code", "a SQL query", "a regex", "a bash one-liner",
    "explaining an error message", "writing unit tests", "a small web-scraping snippet",
    "pandas data wrangling", "a recursion problem", "Big-O analysis", "async/concurrency",
]
REASONING = [
    "a multi-step word problem", "estimation / Fermi problem", "a logic puzzle",
    "comparing two options with trade-offs", "planning a project step by step",
    "a basic probability question", "unit conversion with reasoning",
]

# Fake-but-realistic tool schemas the teacher writes conversations around.
TOOLS = [
    {"name": "get_weather", "description": "Get current weather for a location",
     "parameters": {"type": "object", "properties": {
         "location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
         "required": ["location"]}},
    {"name": "web_search", "description": "Search the web for a query",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "num_results": {"type": "integer"}}, "required": ["query"]}},
    {"name": "calculator", "description": "Evaluate a math expression",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}},
    {"name": "send_email", "description": "Send an email",
     "parameters": {"type": "object", "properties": {
         "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
         "required": ["to", "subject", "body"]}},
    {"name": "create_calendar_event", "description": "Create a calendar event",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"},
         "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start"]}},
    {"name": "query_database", "description": "Run a read-only SQL query",
     "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "get_stock_price", "description": "Get the latest price for a ticker",
     "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "translate_text", "description": "Translate text to a target language",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string"}, "target_lang": {"type": "string"}}, "required": ["text", "target_lang"]}},
]

CATEGORY_WEIGHTS = [
    ("chat", 0.34),
    ("coding", 0.24),
    ("reasoning", 0.14),
    ("tool_single", 0.18),
    ("tool_multi", 0.10),
]

SYS_BASE = (
    "You are generating high-quality training conversations for a small assistant model. "
    "Output ONLY a JSON object, no prose, no markdown fences. Schema:\n"
    '{"messages": [{"role": "system|user|assistant|tool", "content": "..."}, ...]}\n'
    "Rules: natural, varied, genuinely helpful turns; assistant answers are correct and "
    "concise; no boilerplate self-introductions; vary phrasing and length; English. "
    "Do NOT include any real personal data."
)

TOOL_SYS = (
    SYS_BASE + "\nThis conversation MUST use a tool. The assistant calls a tool by emitting, "
    "as its message content, EXACTLY one line:\n"
    "<tool_call>\n{\"name\": <tool-name>, \"arguments\": {<json args>}}\n</tool_call>\n"
    "The arguments JSON must be valid, single-line, and match the tool schema. Then a "
    '"tool"-role message gives a plausible JSON result, then the assistant gives a natural '
    "final answer that uses the result. Available tool(s) for THIS conversation:\n__TOOLS__"
)


def build_prompt(category, rng):
    if category == "chat":
        topic = rng.choice(TOPICS)
        turns = rng.choice([1, 1, 2, 2, 3])
        return SYS_BASE, (f"Write a {turns}-exchange conversation where a user asks about "
                          f"'{topic}'. Make the user's wording realistic and specific.")
    if category == "coding":
        task = rng.choice(CODING)
        return SYS_BASE, (f"Write a conversation where a user needs help with {task}. Include "
                          f"real code in the assistant's answer using markdown code blocks.")
    if category == "reasoning":
        task = rng.choice(REASONING)
        return SYS_BASE, (f"Write a conversation where a user poses {task}. The assistant reasons "
                          f"step by step and gives a clear final answer.")
    if category == "tool_single":
        tool = rng.choice(TOOLS)
        sysmsg = TOOL_SYS.replace("__TOOLS__", json.dumps([tool]))
        return sysmsg, ("Write a single-tool-call conversation: user request, one assistant "
                        "tool_call, one tool result, one final assistant answer.")
    if category == "tool_multi":
        tools = rng.sample(TOOLS, k=min(2, len(TOOLS)))
        sysmsg = TOOL_SYS.replace("__TOOLS__", json.dumps(tools))
        return sysmsg, ("Write a multi-turn conversation needing 2 tool calls (can be different "
                        "tools). Each tool_call is followed by its tool result before the next.")
    raise ValueError(category)


def call_openrouter(model, system, user, api_key, max_tokens, temperature, timeout):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PlangoDev/zell",
        "X-Title": "Zell synthetic data",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def parse_messages(raw):
    """Extract and validate {"messages": [...]} from a model response."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s
        s = s.lstrip("json").strip().strip("`").strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    out = []
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def validate_tool_calls(msgs):
    """For tool conversations, require at least one well-formed <tool_call> JSON."""
    ok = False
    for m in msgs:
        if m["role"] != "assistant" or "<tool_call>" not in m["content"]:
            continue
        try:
            inner = m["content"].split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
            call = json.loads(inner)
            if "name" in call and isinstance(call.get("arguments"), dict):
                ok = True
        except (json.JSONDecodeError, IndexError):
            return False
    return ok


def approx_tokens(msgs, tok):
    text = "\n".join(m["content"] for m in msgs)
    if tok is not None:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return max(1, len(text) // 4)   # ~4 chars/token fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inclusionai/ling-2.6-flash")
    ap.add_argument("--target-tokens", type=int, default=50_000_000)
    ap.add_argument("--out-dir", default="synth/out")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--count-with-qwen", action="store_true",
                    help="count tokens with the Qwen tokenizer (exact) instead of chars/4")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set OPENROUTER_API_KEY in the environment")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"synth_{args.model.replace('/', '_').replace(':', '_')}.jsonl")

    tok = None
    if args.count_with_qwen:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    cats, weights = zip(*CATEGORY_WEIGHTS)

    # shared state
    state = {"tokens": 0, "convs": 0, "fail": 0}
    lock = threading.Lock()
    write_lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    fout = open(out_path, "a", encoding="utf-8")

    def worker(wid):
        rng = random.Random(args.seed * 1000 + wid)
        while not stop.is_set():
            category = rng.choices(cats, weights=weights, k=1)[0]
            system, user = build_prompt(category, rng)
            msgs = None
            for attempt in range(args.max_retries):
                if stop.is_set():
                    return
                try:
                    raw = call_openrouter(args.model, system, user, api_key,
                                          args.max_tokens, args.temperature, args.timeout)
                    cand = parse_messages(raw)
                    if cand and (not category.startswith("tool") or validate_tool_calls(cand)):
                        msgs = cand
                        break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 502, 503, 524):
                        time.sleep(min(2 ** attempt + rng.random(), 30))
                        continue
                    time.sleep(1 + rng.random())
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
                    time.sleep(1 + rng.random())
            if msgs is None:
                with lock:
                    state["fail"] += 1
                continue
            n = approx_tokens(msgs, tok)
            line = json.dumps({"messages": msgs, "category": category}, ensure_ascii=False)
            with write_lock:
                fout.write(line + "\n")
                fout.flush()
            with lock:
                state["tokens"] += n
                state["convs"] += 1
                if state["tokens"] >= args.target_tokens:
                    stop.set()
                if state["convs"] % 50 == 0:
                    dt = time.time() - t0
                    print(f"  {state['tokens']:,}/{args.target_tokens:,} tok  "
                          f"{state['convs']} convs  {state['fail']} fail  "
                          f"({state['tokens']/max(dt,1):,.0f} tok/s)", flush=True)

    print(f"  synth: model={args.model} target={args.target_tokens:,} tok -> {out_path}", flush=True)
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    fout.close()
    dt = time.time() - t0
    print(f"  synth: wrote {state['convs']:,} convs, ~{state['tokens']:,} tokens, "
          f"{state['fail']} failures in {dt:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
