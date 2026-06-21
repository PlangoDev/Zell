# Running test_013 on Kaggle (2x T4)

## Step 0 — notebook settings (sidebar, do this first)

- Accelerator: GPU T4 x2
- Internet: On (requires a phone-verified Kaggle account)

The script launches as a real subprocess (`!python ...`), not inside the notebook
kernel, so `torch.multiprocessing` spawn and NCCL work.

## Step 1 — store the GitHub token (once, the secure way)

The repo is private, so cloning needs a token. Use a fine-grained PAT with
read access to `PlangoDev/plango-labs` (Contents: Read).

Recommended: Add-ons -> Secrets -> add a secret named `GITHUB_TOKEN` with the PAT
as its value. Then the token never appears in the notebook source.

## Cell 1 — dependencies

```python
!pip -q install -U transformers datasets
```

## Cell 2 — clone the repo with the token

Secure (token from Kaggle Secrets):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
os.chdir("/kaggle/working")
!rm -rf plango-labs
!git clone https://$GH_TOKEN@github.com/PlangoDev/plango-labs.git
os.chdir("/kaggle/working/plango-labs/test_013")
print("cwd:", os.getcwd())
```

Inline fallback if you are not using Secrets (do not share or save a public
notebook with the token filled in):

```python
import os
os.environ["GH_TOKEN"] = "PASTE_YOUR_PAT_HERE"
os.chdir("/kaggle/working")
!rm -rf plango-labs
!git clone https://$GH_TOKEN@github.com/PlangoDev/plango-labs.git
os.chdir("/kaggle/working/plango-labs/test_013")
```

## Cell 3 — smoke test (fast, validates the whole pipeline)

```python
!python showdown.py --smoke
```

Tiny sizes, tiny model. Perplexity will be junk; the point is that it runs end to
end with no crash (spawn/sampler/readout logic all exercised).

## Cell 4 — build the token cache (one-time, needs Internet)

```python
!python showdown.py --build-data
```

Streams Wikipedia and tokenizes it once into a uint16 memmap under
`/kaggle/working` (reused on later runs in the same notebook). Expect a few
minutes for 500M tokens. The plain run will also build it automatically if you
skip this, but doing it separately makes the build progress visible.

## Cell 5 — the real run (auto dual-GPU)

```python
!python showdown.py
```

Trains the brain on both T4s, then prints the scoreboard: LLM perplexity, Brain
perplexity, and MACs/token for each. To control the data budget:

```python
!python showdown.py --train-tokens 1000000000
```

## Optional

```python
!python showdown.py --single                 # force one GPU (sanity vs dual)
!python showdown.py --n-gran 32768 --batch 4096   # lower these first if VRAM OOMs
```

To pull later changes without recloning:

```python
import os; os.chdir("/kaggle/working/plango-labs"); 
!git pull
os.chdir("/kaggle/working/plango-labs/test_013")
```

## What good output looks like

- Smoke: a scoreboard prints, no traceback. Numbers are meaningless.
- Build: a line reporting the token count and the `.bin` path.
- Real run: per-block "trained ~N tokens" lines, then an LLM perplexity line, a
  Brain perplexity line, and the scoreboard with the brain's compute ratio.

## Troubleshooting

- VRAM OOM: lower `--n-gran` (try 32768) then `--batch`. The default n_gran=49152
  is sized to fill the T4.
- Clone fails (auth): the PAT needs Contents: Read on the private repo, and the
  owner is `PlangoDev`.
- Internet/streaming errors on `--build-data`: confirm Internet is On in the
  sidebar.
- NCCL init hangs at startup: usually a stale port from a previous run; restart
  the kernel (Run -> Restart) and rerun. `NCCL_P2P_DISABLE=1` is already set.
- Cold eval delay: the first eval downloads pythia-410m + WikiText-103; this is a
  one-time delay after training, unrelated to brain speed.
