#!/usr/bin/env python3
"""Drive a Kaggle script-kernel run of test_013 from the command line.

Auth: reads the KGAT bearer token from ~/.kaggle/access_token and calls the Kaggle
REST API directly (the bundled python client only speaks the legacy kaggle.json
username+key auth, but the server accepts the new token as a Bearer header).

It embeds showdown.py into a single self-contained script kernel (no private-repo
clone needed inside Kaggle), pushes it with GPU + Internet on, polls until the run
finishes, then prints the run log. Re-pushing the same slug makes a new version.

NOTE: the Kaggle API cannot request the 2x T4 accelerator, so API runs get a
single GPU. Use this for correctness + single-GPU numbers; do the final dual-GPU
run from a manual notebook.

Usage:
  python3 tools/kaggle_run.py --flags --smoke
  python3 tools/kaggle_run.py --flags "--train-tokens 5000000" --slug test013-brain-s5m
"""
import argparse, json, os, sys, time, urllib.request, urllib.parse, urllib.error

API = "https://www.kaggle.com/api/v1"
HERE = os.path.dirname(os.path.abspath(__file__))
SHOWDOWN = os.path.join(HERE, "..", "showdown.py")
TOKEN = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
HDR = {"Authorization": "Bearer " + TOKEN, "User-Agent": "plango-driver/1.0"}


def req(method, path, params=None, body=None, timeout=180):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = None
    headers = dict(HDR)
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def whoami():
    _, b = req("GET", "/hello")
    return json.loads(b)["userName"]


def build_source(flags):
    src = open(SHOWDOWN).read()
    header = (
        "# --- injected by kaggle_run.py ---\n"
        "import sys, subprocess\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', "
        "'transformers', 'datasets'], check=False)\n"
        f"sys.argv = ['showdown.py'] + {list(flags)!r}\n"
        "# --- end injection ---\n\n"
    )
    return header + src


def push(user, slug, title, source, gpu=True, net=True):
    body = {
        "slug": f"{user}/{slug}", "newTitle": title,
        "text": source, "language": "python", "kernelType": "script",
        "isPrivate": True, "enableGpu": gpu, "enableTpu": False, "enableInternet": net,
        "datasetDataSources": [], "competitionDataSources": [],
        "kernelDataSources": [], "modelDataSources": [], "categoryIds": [],
    }
    code, b = req("POST", "/kernels/push", body=body)
    return code, json.loads(b)


def status(user, slug):
    code, b = req("GET", "/kernels/status", params={"userName": user, "kernelSlug": slug})
    return code, json.loads(b)


def get_output(user, slug):
    code, b = req("GET", "/kernels/output", params={"userName": user, "kernelSlug": slug})
    return code, json.loads(b)


def print_log(out):
    log = out.get("log")
    if not log:
        print("(no log field; output keys: %s)" % list(out.keys()))
        return
    try:
        entries = json.loads(log)
        for e in entries:
            print(e.get("data", e) if isinstance(e, dict) else e, end="")
        print()
    except Exception:
        print(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default="--smoke", help="args passed to showdown.py")
    ap.add_argument("--slug", default=None, help="kernel slug (default derived from flags)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--poll", type=int, default=20, help="status poll seconds")
    ap.add_argument("--timeout", type=int, default=3600, help="max wait seconds")
    args = ap.parse_args()

    flags = args.flags.split()
    slug = args.slug or ("test013-" + ("smoke" if "--smoke" in flags else "run"))
    title = args.title or ("Test013 " + slug)

    user = whoami()
    print(f"user={user}  slug={slug}  flags={flags}  gpu={not args.no_gpu}", flush=True)

    source = build_source(flags)
    code, resp = push(user, slug, title, source, gpu=not args.no_gpu)
    print(f"push HTTP {code}: {json.dumps(resp)[:300]}", flush=True)
    if code >= 300 or resp.get("error"):
        print("PUSH FAILED", flush=True)
        sys.exit(1)
    ver = resp.get("versionNumber")
    # Kaggle derives the real slug from the title; take it from the response ref/url.
    real = resp.get("ref") or resp.get("url") or f"{user}/{slug}"
    real_slug = real.rstrip("/").split("/")[-1]
    print(f"pushed version {ver}; slug={real_slug}; url: https://www.kaggle.com/code/{user}/{real_slug}", flush=True)

    t0 = time.time()
    last = None
    while time.time() - t0 < args.timeout:
        time.sleep(args.poll)
        code, st = status(user, real_slug)
        s = (st.get("status") or "?")
        if s != last:
            print(f"  [{int(time.time()-t0)}s] status={s}  {st.get('failureMessage') or ''}", flush=True)
            last = s
        if s.lower() not in ("queued", "running"):
            break
    print(f"final status={last}  ({int(time.time()-t0)}s)", flush=True)

    code, out = get_output(user, real_slug)
    print("=" * 70 + "\nKERNEL LOG\n" + "=" * 70, flush=True)
    print_log(out)


if __name__ == "__main__":
    main()
