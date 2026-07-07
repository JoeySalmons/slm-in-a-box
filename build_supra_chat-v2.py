#!/usr/bin/env python3
"""
build_supra_chat.py  (v3 — editable messages + configurable prompt format)
──────────────────────────────────────────────────────────────────────────
Packs a Supra-50M ONNX model + tokenizer into a single self-contained HTML
file that runs inference entirely in the browser via transformers.js + ORT.

v3 adds:
  • Edit past messages: hover any user/assistant bubble → "edit" opens an inline
    editor. Saving rewrites that turn in-place; it does NOT resend or regenerate —
    the edit only takes effect as context on your NEXT message.
  • Configurable prompt format: a "Prompt format" panel with editable System /
    preamble, User-turn and Assistant-turn boxes ({system} / {content} placeholders),
    a preset dropdown (Alpaca, Alpaca-with-input, ChatML, Vicuna, Raw) and a
    hardcoded "Reset to Alpaca". Alpaca (no-input) is the default and is byte-for-byte
    identical to the format v2 hardcoded. Output-cleanup stop markers are derived
    from the active template.
  • "Show greeting message" toggle (Settings): off = the chat starts empty so your
    first message is the first turn the model sees. (The greeting was always
    display-only and never part of the history sent to the model.)

v2 features (still present):
  • --offline  : embeds the transformers.js runtime AND the ORT WASM binary so
                 the file needs NO network connection, ever (true USB-stick mode).
  • A settings panel exposing the usual LLM knobs (temperature, top-p, top-k,
    max tokens, repetition penalty) + a system prompt, persisted to localStorage.
  • Stop / Clear / Copy controls, markdown rendering, tok/s stats, and an optional
    WebGPU backend with CPU fallback on init failure (NOT on runtime freeze — a
    synchronous main-thread freeze during inference cannot be intercepted; see
    finding #14 in the session notes).

USAGE
    python3 build_supra_chat.py [model_dir] [output.html] [options]

    model_dir   folder with the model + tokenizer files (default: current dir)
    output      output HTML path (default: supra50m_chat.html)

OPTIONS
    --offline / --cdn   embed the runtime (default) vs. load it from jsDelivr.
    --model-file NAME   ONNX filename to embed (default: model_int8.onnx).
    --runtime-dir DIR   where to find/cache the 3 runtime files (default: model_dir).
    --no-download       never hit the network; runtime files must already be local.
    --tjs-version VER   transformers.js version for offline assets (default: 3.8.1).

REQUIRED MODEL FILES (in model_dir)
    model_int8.onnx        ONNX weights (int8, fp32 activations — WASM-safe)
    tokenizer.json         HF fast-tokenizer JSON
    config.json            model architecture config
    generation_config.json eos/bos/pad token IDs
  (tokenizer_config.json is rewritten from scratch; not read.)

OFFLINE RUNTIME FILES (fetched or cached into --runtime-dir)
    transformers.min.js
    ort-wasm-simd-threaded.jsep.mjs     (patched at build time)
    ort-wasm-simd-threaded.jsep.wasm    (embedded as wasmBinary)

  All available from npm or jsDelivr:
    https://cdn.jsdelivr.net/npm/@huggingface/transformers@<VER>/dist/<file>

WHY THIS WORKS OFFLINE ON file://  (verified against the ORT 1.x source)
    • transformers.min.js is imported from a data: URL (ESM import of a data
      URL is allowed on file://; a blob: URL is not reliably importable there).
    • ORT loads its Emscripten glue (.mjs) from wasmPaths.mjs — we hand it a
      data: URL too. Single-threaded, so ORT imports it directly (no blob path).
    • The .wasm is supplied as env.backends.onnx.wasm.wasmBinary (an ArrayBuffer),
      so ORT never fetches it. We still patch the one `new URL(..,import.meta.url)`
      in the glue, because that expression is *computed* even when unused and a
      data: URL is not a valid base for relative URL resolution (it would throw).
"""

import argparse, base64, json, os, sys, io, tarfile, urllib.request

DIST = "dist/{f}"
JSDELIVR = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@{ver}/dist/{f}"
NPM_TARBALL = "https://registry.npmjs.org/@huggingface/transformers/-/transformers-{ver}.tgz"

TJS_JS   = "transformers.min.js"
ORT_MJS  = "ort-wasm-simd-threaded.jsep.mjs"
ORT_WASM = "ort-wasm-simd-threaded.jsep.wasm"

# The single line in the ORT glue that throws under a data: URL base.
# `import.meta.url` of a data: URL is the data: URL itself, and
# `new URL("x.wasm", "data:...")` raises "Invalid URL". The result is never
# used (we supply wasmBinary), so we replace it with a harmless literal.
MJS_PATCH_FROM = '(new URL("ort-wasm-simd-threaded.jsep.wasm",import.meta.url)).href'
MJS_PATCH_TO   = '"http://supra-local/ort-wasm-simd-threaded.jsep.wasm"'


# ── runtime asset acquisition ────────────────────────────────────────────────
def _read_local(runtime_dir, name):
    for cand in (os.path.join(runtime_dir, name),
                 os.path.join(runtime_dir, "runtime", name)):
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                return f.read()
    return None


def _download_one(ver, name):
    url = JSDELIVR.format(ver=ver, f=name)
    print(f"  ↓ {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "build_supra_chat"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _download_from_npm(ver):
    """Fallback: pull the npm tarball and extract the 3 dist files."""
    url = NPM_TARBALL.format(ver=ver)
    print(f"  ↓ npm tarball {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "build_supra_chat"})
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    out = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for name in (TJS_JS, ORT_MJS, ORT_WASM):
            m = tar.getmember(DIST.format(f=name).replace("dist/", "package/dist/"))
            out[name] = tar.extractfile(m).read()
    return out


def get_runtime(runtime_dir, ver, allow_download):
    """Return dict {name: bytes} for the 3 runtime files, cached to runtime_dir."""
    files, missing = {}, []
    for name in (TJS_JS, ORT_MJS, ORT_WASM):
        data = _read_local(runtime_dir, name)
        if data is None:
            missing.append(name)
        else:
            files[name] = data
            print(f"  ✓ local {name} ({len(data)/1e6:.1f} MB)", flush=True)

    if missing:
        if not allow_download:
            sys.exit(f"\nERROR: missing runtime files {missing} and --no-download set.\n"
                     f"Place them in {runtime_dir} or drop --no-download.\n")
        print(f"Fetching {len(missing)} runtime file(s) (transformers.js {ver})...", flush=True)
        try:
            for name in missing:
                files[name] = _download_one(ver, name)
        except Exception as e:
            print(f"  jsDelivr failed ({e}); trying npm tarball...", flush=True)
            files.update(_download_from_npm(ver))
        os.makedirs(runtime_dir, exist_ok=True)
        for name in missing:
            with open(os.path.join(runtime_dir, name), "wb") as f:
                f.write(files[name])
            print(f"  ✓ cached {name} → {runtime_dir}", flush=True)

    # Patch the glue so it survives a data: URL import.
    mjs = files[ORT_MJS].decode("utf-8")
    if MJS_PATCH_FROM in mjs:
        mjs = mjs.replace(MJS_PATCH_FROM, MJS_PATCH_TO, 1)
    elif MJS_PATCH_TO not in mjs:
        print("  ! WARNING: expected glue patch target not found; "
              "offline import.meta.url throw may occur.", flush=True)
    files[ORT_MJS] = mjs.encode("utf-8")
    return files


# ── HTML template ────────────────────────────────────────────────────────────
# Placeholders are unique, non-overlapping, and substituted longest-first.
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Supra 50M — Local Chat</title>
<style>
:root{
  --bg:#11131a; --panel:#171a22; --panel2:#1e222c; --line:#262b37;
  --ink:#e6e8ee; --muted:#8b93a7; --faint:#5b6478;
  --user:#3b5bdb; --filament:#ffb454; --good:#37c98a; --bad:#ff6b6b;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--sans);background:var(--bg);color:var(--ink);
  height:100dvh;display:flex;flex-direction:column;overflow:hidden}

header{display:flex;align-items:center;gap:10px;padding:10px 14px;
  background:var(--panel);border-bottom:1px solid var(--line)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--faint);flex-shrink:0;
  transition:background .3s,box-shadow .3s}
.dot.loading{background:var(--filament);box-shadow:0 0 8px var(--filament);animation:pulse 1.2s infinite}
.dot.ready{background:var(--good);box-shadow:0 0 7px var(--good)}
.dot.gen{background:var(--filament);box-shadow:0 0 9px var(--filament);animation:pulse .7s infinite}
.dot.error{background:var(--bad);box-shadow:0 0 7px var(--bad)}
.brand{font-weight:650;font-size:15px;letter-spacing:.2px}
.sub{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;
  font-family:var(--mono);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{font-size:10px;font-family:var(--mono);letter-spacing:.5px;color:var(--filament);
  border:1px solid #5a4422;background:#241c10;border-radius:999px;padding:2px 7px;text-transform:uppercase}
.spacer{flex:1}
.iconbtn{background:transparent;border:1px solid var(--line);color:var(--muted);
  border-radius:8px;height:32px;padding:0 10px;font-size:12px;cursor:pointer;
  display:flex;align-items:center;gap:6px;font-family:var(--sans)}
.iconbtn:hover{color:var(--ink);border-color:#39414f}
.iconbtn:disabled{opacity:.4;cursor:not-allowed}

#settings{display:none;background:var(--panel);border-bottom:1px solid var(--line);
  padding:14px;gap:14px;flex-wrap:wrap}
#settings.open{display:flex}
.field{display:flex;flex-direction:column;gap:5px;min-width:170px;flex:1 1 170px}
.field label{font-size:11px;color:var(--muted);font-family:var(--mono);
  display:flex;justify-content:space-between;letter-spacing:.3px}
.field label b{color:var(--filament);font-weight:600}
.field input[type=range]{width:100%;accent-color:var(--filament);height:4px}
.field select,.field textarea{background:var(--panel2);border:1px solid var(--line);
  color:var(--ink);border-radius:8px;padding:8px;font-size:13px;font-family:var(--sans);outline:none}
.field select:focus,.field textarea:focus{border-color:var(--user)}
.field textarea{resize:vertical;min-height:46px;font-size:13px}
.field.wide{flex-basis:100%}
.row{display:flex;gap:14px;align-items:center;flex-basis:100%;flex-wrap:wrap}
.check{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);font-family:var(--mono)}
.check input{accent-color:var(--filament)}
/* prompt-format sub-panel */
#tpl_toggle{cursor:pointer;user-select:none}
#tplpanel{display:none;flex-direction:column;gap:9px;margin-top:6px;
  border:1px solid var(--line);border-radius:9px;padding:11px;background:var(--panel2)}
#tplpanel.open{display:flex}
.tplrow{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.tpllabel{font-size:11px;color:var(--muted);font-family:var(--mono);letter-spacing:.3px}
#tplpanel textarea{background:var(--bg);border:1px solid var(--line);color:var(--ink);
  border-radius:8px;padding:8px;font-family:var(--mono);font-size:12.5px;line-height:1.45;
  resize:vertical;min-height:44px;outline:none;white-space:pre}
#tplpanel textarea:focus{border-color:var(--user)}
#tplpanel select{background:var(--bg);border:1px solid var(--line);color:var(--ink);
  border-radius:8px;padding:6px 8px;font-size:12.5px;font-family:var(--sans);outline:none}

#chat{flex:1;overflow-y:auto;padding:18px 14px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:90%;padding:11px 14px;border-radius:14px;font-size:14.5px;line-height:1.6;
  word-break:break-word;position:relative}
.msg.user{background:var(--user);color:#fff;align-self:flex-end;border-radius:14px 14px 4px 14px;white-space:pre-wrap}
.msg.bot{background:var(--panel2);border:1px solid var(--line);align-self:flex-start;border-radius:14px 14px 14px 4px}
.msg.err{background:#2a1416;border:1px solid #5a2024;color:#ff9b9b;align-self:flex-start;border-radius:14px}
.msg .tools{position:absolute;top:-9px;right:8px;display:none;gap:4px}
.msg.bot:hover .tools,.msg.user:hover .tools{display:flex}
.msg .tools button{font-size:10px;font-family:var(--mono);background:var(--panel);
  border:1px solid var(--line);color:var(--muted);border-radius:6px;padding:2px 7px;cursor:pointer}
.msg .tools button:hover{color:var(--ink)}
/* inline message editor */
.msg .editbox{width:100%;min-width:min(360px,72vw);background:var(--panel);
  border:1px solid var(--user);color:var(--ink);border-radius:9px;padding:8px 10px;
  font-family:var(--sans);font-size:14px;line-height:1.5;resize:vertical;outline:none}
.msg .editbar{display:flex;gap:7px;margin-top:7px}
.msg .editbar button{font-size:11px;font-family:var(--mono);border-radius:7px;
  padding:4px 11px;cursor:pointer;border:1px solid var(--line)}
.msg .editbar .editsave{background:var(--user);color:#fff;border-color:var(--user)}
.msg .editbar .editcancel{background:transparent;color:var(--muted)}
.msg .editbar .editcancel:hover{color:var(--ink)}
.editnote{font-size:10px;font-family:var(--mono);color:var(--faint);margin-left:auto;align-self:center}
.msg .meta{margin-top:7px;font-size:10.5px;color:var(--faint);font-family:var(--mono)}
/* markdown */
.msg.bot p{margin:0 0 8px}.msg.bot p:last-child{margin-bottom:0}
.msg.bot h1,.msg.bot h2,.msg.bot h3{font-size:15px;margin:10px 0 6px;color:var(--ink)}
.msg.bot ul,.msg.bot ol{margin:6px 0 8px 20px}.msg.bot li{margin:2px 0}
.msg.bot code{background:#0d0f15;border:1px solid var(--line);border-radius:5px;
  padding:1px 5px;font-family:var(--mono);font-size:12.5px}
.msg.bot pre{background:#0d0f15;border:1px solid var(--line);border-radius:9px;
  padding:11px;overflow-x:auto;margin:8px 0}
.msg.bot pre code{background:none;border:none;padding:0}
.msg.bot a{color:#8db4ff}
.caret{display:inline-block;width:7px;height:15px;background:var(--filament);
  vertical-align:-2px;animation:blink 1s step-end infinite;border-radius:1px}

footer{padding:11px 14px;background:var(--panel);border-top:1px solid var(--line)}
#inputrow{display:flex;gap:9px;align-items:flex-end}
#input{flex:1;background:var(--panel2);border:1px solid var(--line);border-radius:11px;
  padding:11px 13px;color:var(--ink);font-size:14.5px;font-family:var(--sans);outline:none;
  resize:none;height:46px;max-height:150px;line-height:1.4}
#input:focus{border-color:var(--user)}
#send{background:var(--user);color:#fff;border:none;border-radius:11px;height:46px;
  padding:0 18px;font-size:14px;font-weight:650;cursor:pointer;flex-shrink:0}
#send:disabled{background:#26336b;color:#8a93b8;cursor:not-allowed}
#stop{background:#3a2226;color:#ffb4b4;border:1px solid #5a2c32;border-radius:11px;height:46px;
  padding:0 16px;font-weight:650;cursor:pointer;flex-shrink:0;display:none}
.hint{font-size:10.5px;color:var(--faint);font-family:var(--mono);margin-top:7px;text-align:center}
.field .hint{text-align:left;margin-top:4px;line-height:1.4;color:var(--muted)}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@keyframes blink{50%{opacity:0}}
@media (prefers-reduced-motion:reduce){.dot,.caret{animation:none}}
::-webkit-scrollbar{width:8px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2b313d;border-radius:4px}
</style>
</head>
<body>
<header>
  <div class="dot loading" id="dot"></div>
  <span class="brand">Supra&nbsp;50M</span>
  <span class="pill" id="modepill">offline</span>
  <span class="sub" id="status">Decoding model…</span>
  <span class="spacer"></span>
  <button class="iconbtn" id="gear" title="Settings">⚙ Settings</button>
  <button class="iconbtn" id="clear" title="Clear conversation">Clear</button>
</header>

<div id="settings">
  <div class="field">
    <label>Temperature <b id="v_temp">0.85</b></label>
    <input type="range" id="temperature" min="0" max="1.5" step="0.05" value="0.85">
  </div>
  <div class="field">
    <label>Top-p <b id="v_top_p">0.90</b></label>
    <input type="range" id="top_p" min="0" max="1" step="0.05" value="0.9">
  </div>
  <div class="field">
    <label>Top-k <b id="v_top_k">50</b></label>
    <input type="range" id="top_k" min="0" max="100" step="1" value="50">
  </div>
  <div class="field">
    <label>Max new tokens <b id="v_max">200</b></label>
    <input type="range" id="max_new_tokens" min="16" max="512" step="16" value="200">
  </div>
  <div class="field">
    <label>Repetition penalty <b id="v_rep">1.15</b></label>
    <input type="range" id="repetition_penalty" min="1" max="1.5" step="0.01" value="1.15">
  </div>
  <div class="field">
    <label>Backend</label>
    <select id="backend">
      <option value="wasm">CPU (WASM) — recommended</option>
      <option value="webgpu">WebGPU — experimental, may freeze</option>
    </select>
    <div class="hint">This model is int8; the WebGPU path falls back to CPU per-op and can lock the tab. WASM is the supported path.</div>
  </div>
  <div class="field wide">
    <label>System prompt (inserted where {system} appears in the format)</label>
    <textarea id="system" placeholder="e.g. You are a concise, helpful assistant."></textarea>
  </div>
  <div class="field wide">
    <label>Prompt format <b id="tpl_toggle">show ▸</b></label>
    <div id="tplpanel">
      <div class="tplrow">
        <span class="tpllabel">Preset</span>
        <select id="tpl_preset">
          <option value="alpaca">Alpaca (no input) — default</option>
          <option value="alpaca_input">Alpaca (original, with input header)</option>
          <option value="chatml">ChatML</option>
          <option value="vicuna">Vicuna</option>
          <option value="raw">Raw (no formatting)</option>
          <option value="custom">Custom…</option>
        </select>
        <button class="iconbtn" id="tpl_reset" style="margin-left:auto">Reset to Alpaca</button>
      </div>
      <span class="tpllabel">System / preamble header — <b style="color:var(--filament)">{system}</b> = your system prompt above</span>
      <textarea id="tpl_preamble" rows="2" spellcheck="false"></textarea>
      <span class="tpllabel">User turn — <b style="color:var(--filament)">{content}</b> = the message</span>
      <textarea id="tpl_userTurn" rows="2" spellcheck="false"></textarea>
      <span class="tpllabel">Assistant turn — <b style="color:var(--filament)">{content}</b> = the reply; text before it cues generation</span>
      <textarea id="tpl_botTurn" rows="2" spellcheck="false"></textarea>
      <div class="hint">Whitespace and newlines are literal. Edits apply to your <b>next</b> message; they don't resend anything.</div>
    </div>
  </div>
  <div class="row">
    <label class="check"><input type="checkbox" id="persist" checked> Save chat &amp; settings on this device</label>
    <label class="check"><input type="checkbox" id="greeting" checked> Show greeting message</label>
    <span class="spacer"></span>
    <button class="iconbtn" id="reset">Reset defaults</button>
    <button class="iconbtn" id="reload">Apply backend &amp; reload model</button>
  </div>
</div>

<div id="chat"></div>

<footer>
  <div id="inputrow">
    <textarea id="input" placeholder="Type a message…" rows="1" disabled></textarea>
    <button id="stop">Stop</button>
    <button id="send" disabled>Send</button>
  </div>
  <div class="hint" id="hint">Enter to send · Shift+Enter for newline · everything runs locally</div>
</footer>

<script type="module">
// ════════════ EMBEDDED DATA (substituted at build time) ════════════
const MODEL_B64     = "PLACEHOLDER_MODEL_B64";
const CONFIG        = PLACEHOLDER_CONFIG;
const TOKENIZER     = PLACEHOLDER_TOKENIZER;
const TOKENIZER_CFG = PLACEHOLDER_TOKENIZER_CFG;
const GEN_CONFIG    = PLACEHOLDER_GEN_CONFIG;

// ════════════ DOM ════════════
const $ = id => document.getElementById(id);
const dot=$("dot"), statusEl=$("status"), chat=$("chat"), input=$("input"),
      send=$("send"), stop=$("stop"), modepill=$("modepill");

function setStatus(state, text){
  dot.className = "dot " + state;
  statusEl.textContent = text;
}
function b64ToBytes(b64){
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) buf[i] = bin.charCodeAt(i);
  return buf;
}
const modelBytes = b64ToBytes(MODEL_B64);

// ════════════ PROMPT TEMPLATES ════════════
// Each turn format uses {content} for the message; the preamble uses {system}
// for the user's system-prompt text. The pending assistant reply is cued by the
// text in botTurn that precedes {content}.
const TEMPLATES = {
  alpaca: {
    preamble:"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{system}",
    userTurn:"### Instruction:\n{content}\n\n",
    botTurn: "### Response:\n{content}\n\n",
  },
  alpaca_input: {
    preamble:"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{system}",
    userTurn:"### Instruction:\n{content}\n\n### Input:\n\n\n",
    botTurn: "### Response:\n{content}\n\n",
  },
  chatml: {
    preamble:"<|im_start|>system\n{system}<|im_end|>\n",
    userTurn:"<|im_start|>user\n{content}<|im_end|>\n",
    botTurn: "<|im_start|>assistant\n{content}<|im_end|>\n",
  },
  vicuna: {
    preamble:"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n{system}",
    userTurn:"USER: {content}\n",
    botTurn: "ASSISTANT: {content}\n",
  },
  raw: {
    preamble:"{system}",
    userTurn:"{content}\n",
    botTurn: "{content}\n",
  },
};
const ALPACA = TEMPLATES.alpaca;  // hardcoded reset target

// ════════════ SETTINGS (persisted) ════════════
const DEFAULTS = {temperature:0.85, top_p:0.9, top_k:50, max_new_tokens:200,
                  repetition_penalty:1.15, backend:"wasm", system:"", persist:true,
                  greeting:true,
                  template:{...ALPACA}};
let settings = {...DEFAULTS, template:{...ALPACA}};
let storageOK = false;
try { localStorage.setItem("__t","1"); localStorage.removeItem("__t"); storageOK = true; } catch(e){}

function loadSettings(){
  if(!storageOK) return;
  try{
    const s = JSON.parse(localStorage.getItem("supra_settings")||"{}");
    settings = {...DEFAULTS, ...s};
    settings.template = {...ALPACA, ...(s.template||{})};  // tolerate old/partial saves
  }catch(e){}
}
function saveSettings(){
  if(!storageOK || !settings.persist) return;
  try{ localStorage.setItem("supra_settings", JSON.stringify(settings)); }catch(e){}
}
function saveHistory(){
  if(!storageOK || !settings.persist){ try{localStorage.removeItem("supra_history");}catch(e){} return; }
  try{ localStorage.setItem("supra_history", JSON.stringify(history)); }catch(e){}
}
function loadHistory(){
  if(!storageOK) return [];
  try{ return JSON.parse(localStorage.getItem("supra_history")||"[]"); }catch(e){ return []; }
}

// reflect settings → controls
function syncControls(){
  for(const k of ["temperature","top_p","top_k","max_new_tokens","repetition_penalty"]) $(k).value = settings[k];
  $("backend").value = settings.backend;
  $("system").value  = settings.system;
  $("persist").checked = settings.persist;
  $("greeting").checked = settings.greeting;
  $("v_temp").textContent = (+settings.temperature).toFixed(2);
  $("v_top_p").textContent = (+settings.top_p).toFixed(2);
  $("v_top_k").textContent = settings.top_k;
  $("v_max").textContent  = settings.max_new_tokens;
  $("v_rep").textContent  = (+settings.repetition_penalty).toFixed(2);
  $("tpl_preamble").value = settings.template.preamble;
  $("tpl_userTurn").value = settings.template.userTurn;
  $("tpl_botTurn").value  = settings.template.botTurn;
  $("tpl_preset").value   = detectPreset(settings.template);
}
function detectPreset(t){
  for(const k in TEMPLATES){
    const p = TEMPLATES[k];
    if(p.preamble===t.preamble && p.userTurn===t.userTurn && p.botTurn===t.botTurn) return k;
  }
  return "custom";
}
// controls → settings
[["temperature","v_temp",2],["top_p","v_top_p",2],["top_k","v_top_k",0],
 ["max_new_tokens","v_max",0],["repetition_penalty","v_rep",2]].forEach(([id,label,dp])=>{
  $(id).addEventListener("input",()=>{
    settings[id] = id==="top_k"||id==="max_new_tokens" ? parseInt($(id).value) : parseFloat($(id).value);
    $(label).textContent = dp ? (+settings[id]).toFixed(dp) : settings[id];
    saveSettings();
  });
});
$("system").addEventListener("input",()=>{settings.system=$("system").value;saveSettings();});
$("backend").addEventListener("change",()=>{settings.backend=$("backend").value;saveSettings();});
$("greeting").addEventListener("change",()=>{settings.greeting=$("greeting").checked;saveSettings();});
$("persist").addEventListener("change",()=>{
  settings.persist=$("persist").checked;
  if(!settings.persist){ try{localStorage.removeItem("supra_settings");localStorage.removeItem("supra_history");}catch(e){} }
  else { saveSettings(); saveHistory(); }
});
$("gear").addEventListener("click",()=>$("settings").classList.toggle("open"));
$("reset").addEventListener("click",()=>{
  const persist = settings.persist;
  settings = {...DEFAULTS, persist, template:{...ALPACA}};
  syncControls(); saveSettings();
});

// ── prompt-format controls ──
["preamble","userTurn","botTurn"].forEach(f=>{
  $("tpl_"+f).addEventListener("input",()=>{
    settings.template[f] = $("tpl_"+f).value;
    $("tpl_preset").value = detectPreset(settings.template);
    saveSettings();
  });
});
$("tpl_preset").addEventListener("change",()=>{
  const k = $("tpl_preset").value;
  if(k==="custom") return;               // "custom" is a display-only state
  settings.template = {...TEMPLATES[k]};
  $("tpl_preamble").value = settings.template.preamble;
  $("tpl_userTurn").value = settings.template.userTurn;
  $("tpl_botTurn").value  = settings.template.botTurn;
  saveSettings();
});
$("tpl_reset").addEventListener("click",()=>{
  settings.template = {...ALPACA};
  $("tpl_preamble").value = ALPACA.preamble;
  $("tpl_userTurn").value = ALPACA.userTurn;
  $("tpl_botTurn").value  = ALPACA.botTurn;
  $("tpl_preset").value   = "alpaca";
  saveSettings();
});
$("tpl_toggle").addEventListener("click",()=>{
  const open = $("tplpanel").classList.toggle("open");
  $("tpl_toggle").textContent = open ? "hide ▾" : "show ▸";
});

loadSettings();
syncControls();
if(!storageOK){ $("persist").disabled = true; $("persist").parentElement.style.opacity=".5"; }

// ════════════ MARKDOWN (tiny, escape-first) ════════════
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function mdToHtml(src){
  const blocks=[];
  src = src.replace(/```([\s\S]*?)```/g,(_,c)=>{blocks.push("<pre><code>"+esc(c.replace(/^\n/,""))+"</code></pre>");return "\u0000"+(blocks.length-1)+"\u0000";});
  src = esc(src);
  src = src.replace(/`([^`]+)`/g,(_,c)=>"<code>"+c+"</code>");
  src = src.replace(/\*\*([^*]+)\*\*/g,"<b>$1</b>").replace(/(^|[^*])\*([^*]+)\*/g,"$1<i>$2</i>");
  src = src.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,'<a href="$2" rel="noopener">$1</a>');
  const lines = src.split("\n"); let html="", list=null;
  const closeList=()=>{ if(list){html+="</"+list+">";list=null;} };
  for(let ln of lines){
    let m;
    if(/^\u0000\d+\u0000$/.test(ln.trim())){ closeList(); html+=blocks[+ln.trim().replace(/\u0000/g,"")]; continue; }
    if((m=ln.match(/^(#{1,3})\s+(.*)/))){ closeList(); html+="<h"+m[1].length+">"+m[2]+"</h"+m[1].length+">"; continue; }
    if((m=ln.match(/^\s*[-*]\s+(.*)/))){ if(list!=="ul"){closeList();html+="<ul>";list="ul";} html+="<li>"+m[1]+"</li>"; continue; }
    if((m=ln.match(/^\s*\d+\.\s+(.*)/))){ if(list!=="ol"){closeList();html+="<ol>";list="ol";} html+="<li>"+m[1]+"</li>"; continue; }
    closeList();
    if(ln.trim()==="") html+=""; else html+="<p>"+ln+"</p>";
  }
  closeList();
  return html;
}

// ════════════ CHAT RENDERING ════════════
// `idx` (when given) is the message's index in `history`; only history-backed
// messages get an "edit" tool. Transient notices (greeting, "cleared") omit idx.
let history = [];
function addMsg(role, text, idx){
  const div = document.createElement("div");
  div.className = "msg " + (role==="user"?"user":role==="error"?"err":"bot");
  fillMsg(div, role, text, idx);
  chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
  return div;
}
function fillMsg(div, role, text, idx){
  div.dataset.role = role;
  div.dataset.raw  = text;
  if(idx==null) delete div.dataset.idx; else div.dataset.idx = idx;
  if(role==="bot") div.innerHTML = mdToHtml(text);
  else             div.textContent = text;      // user + error: literal
  if(role==="error") return;
  if(role==="bot" && div.dataset.meta){
    const meta=document.createElement("div"); meta.className="meta";
    meta.textContent=div.dataset.meta; div.appendChild(meta);
  }
  attachTools(div, role, text, idx);
}
function attachTools(div, role, text, idx){
  const tools=document.createElement("div"); tools.className="tools";
  if(idx!=null){
    const ed=document.createElement("button"); ed.textContent="edit";
    ed.onclick=()=>startEdit(div); tools.appendChild(ed);
  }
  if(role==="bot"){
    const cp=document.createElement("button"); cp.textContent="copy";
    cp.onclick=()=>copyText(div.dataset.raw??text); tools.appendChild(cp);
  }
  if(tools.children.length) div.appendChild(tools);
}
function deleteMsg(div, idx){
  // Remove from history (if history-backed) and shift the data-idx of every
  // later bubble down by one so their edit tools keep pointing at the right turn.
  if(idx!=null && idx>=0 && history[idx]!==undefined){
    history.splice(idx,1);
    saveHistory();
    chat.querySelectorAll(".msg").forEach(el=>{
      if(el!==div && el.dataset.idx!=null){
        const i=parseInt(el.dataset.idx);
        if(i>idx) el.dataset.idx = i-1;
      }
    });
  }
  div.remove();
}
function startEdit(div){
  if(generating || div.dataset.editing==="1") return;
  div.dataset.editing="1";
  const role = div.dataset.role;
  const cur  = div.dataset.raw ?? "";
  div.innerHTML="";
  const ta=document.createElement("textarea"); ta.className="editbox"; ta.value=cur;
  const bar=document.createElement("div"); bar.className="editbar";
  const save=document.createElement("button"); save.className="editsave"; save.textContent="Save";
  const cancel=document.createElement("button"); cancel.className="editcancel"; cancel.textContent="Cancel";
  const note=document.createElement("span"); note.className="editnote"; note.textContent="⌘/Ctrl+Enter · Esc";
  bar.appendChild(save); bar.appendChild(cancel); bar.appendChild(note);
  div.appendChild(ta); div.appendChild(bar);
  const autosize=()=>{ ta.style.height="auto"; ta.style.height=Math.min(ta.scrollHeight,420)+"px"; };
  ta.addEventListener("input",autosize); autosize(); ta.focus();
  const finish=(nv)=>{
    div.dataset.editing="0";
    // Re-read the index live: a delete elsewhere may have shifted it since edit began.
    const liveIdx = div.dataset.idx!=null ? parseInt(div.dataset.idx) : null;
    if(nv!=null && nv.trim()===""){ deleteMsg(div, liveIdx); return; }  // emptied → remove
    if(nv!=null){
      div.dataset.raw = nv;
      delete div.dataset.meta;                       // stats no longer match edited text
      if(liveIdx!=null && history[liveIdx]){ history[liveIdx].content = nv; saveHistory(); }
    }
    fillMsg(div, role, nv!=null?nv:cur, liveIdx);    // edits do NOT trigger generation
  };
  save.onclick   = ()=>finish(ta.value);
  cancel.onclick = ()=>finish(null);
  ta.addEventListener("keydown",e=>{
    if(e.key==="Enter" && (e.metaKey||e.ctrlKey)){ e.preventDefault(); finish(ta.value); }
    else if(e.key==="Escape"){ e.preventDefault(); finish(null); }
  });
}
function copyText(t){
  if(navigator.clipboard && window.isSecureContext){ navigator.clipboard.writeText(t); return; }
  const ta=document.createElement("textarea"); ta.value=t; ta.style.position="fixed"; ta.style.opacity="0";
  document.body.appendChild(ta); ta.select();
  try{ document.execCommand("copy"); }catch(e){} ta.remove();
}
function renderHistory(){
  chat.innerHTML="";
  history.forEach((m,i)=> addMsg(m.role==="user"?"user":"bot", m.content, i));
}

// ════════════ RUNTIME LOAD ════════════
// __RUNTIME_SETUP__

// ════════════ PROMPT (template-driven; Alpaca by default) ════════════
function splitTurn(fmt){
  const i = fmt.indexOf("{content}");
  return i<0 ? {pre:fmt, post:""} : {pre:fmt.slice(0,i), post:fmt.slice(i+9)};
}
function buildPrompt(hist){
  const t = settings.template;
  const sys = (settings.system||"").trim();
  let head = t.preamble || "";
  if(head.includes("{system}")) head = head.replace(/\{system\}/g, sys ? sys+"\n\n" : "");
  else if(sys)                  head = head + (head && !head.endsWith("\n\n") ? "\n\n" : "") + sys + "\n\n";
  let p = head;
  for(const m of hist){
    const fmt = m.role==="user" ? t.userTurn : t.botTurn;
    p += fmt.replace("{content}", m.content);
  }
  p += splitTurn(t.botTurn).pre;   // cue the pending assistant reply
  return p;
}
function stopMarkers(){
  const out=[];
  for(const fmt of [settings.template.userTurn, settings.template.botTurn]){
    const pre = splitTurn(fmt).pre.trim();
    if(pre) out.push(pre);
  }
  return out;
}
function cleanOutput(s){
  // cut where the model runs into the next turn marker (whichever comes first)
  let cut = s.length;
  for(const mk of stopMarkers()){
    const i = s.indexOf(mk);
    if(i>=0 && i<cut) cut = i;
  }
  return s.slice(0,cut).trim();
}

// ════════════ MODEL INIT ════════════
let generator=null, activeDevice="wasm";
async function loadModel(device){
  setStatus("loading","Initialising model · "+device.toUpperCase()+"…");
  generator = await pipeline("text-generation","supra-50m",{
    dtype:"fp32", device,
    progress_callback:(p)=>{ if(p.status==="progress"&&p.total){
      setStatus("loading","Loading model… "+Math.round(p.loaded/p.total*100)+"%"); }},
  });
}
async function init(){
  // Migrate the retired "auto" backend value to the supported WASM path.
  if(settings.backend==="auto"){ settings.backend="wasm"; $("backend").value="wasm"; saveSettings(); }
  let want = settings.backend;

  if(want==="webgpu" && !navigator.gpu){
    setStatus("loading","No WebGPU in this browser — using CPU…");
    want="wasm"; settings.backend="wasm"; $("backend").value="wasm"; saveSettings();
  }
  if(want==="webgpu"){
    // The freeze happens at generation, not load, but this is the safe checkpoint
    // to warn: the embedded model is int8 and the WebGPU EP falls back to CPU per-op.
    const ok = confirm(
      "WebGPU is experimental with this model.\n\n"+
      "The embedded weights are int8, which the WebGPU backend does not fully "+
      "support — it falls back to the CPU for many operations and can FREEZE the "+
      "tab so badly you have to force-close it.\n\n"+
      "CPU (WASM) is the supported, reliable path.\n\n"+
      "OK = try WebGPU anyway.   Cancel = use CPU (WASM).");
    if(!ok){ want="wasm"; settings.backend="wasm"; $("backend").value="wasm"; saveSettings(); }
  }

  try{
    await loadModel(want); activeDevice=want;
  }catch(err){
    if(want==="webgpu"){
      console.warn("WebGPU init failed, falling back to WASM:", err);
      setStatus("loading","WebGPU failed — falling back to CPU…");
      await loadModel("wasm"); activeDevice="wasm";
    } else throw err;
  }
  setStatus("ready","ready · "+activeDevice.toUpperCase()+(storageOK?"":" · no-storage"));
  input.disabled=false; send.disabled=false; input.focus();
  history = loadHistory();
  if(history.length) renderHistory();
  else if(settings.greeting) addMsg("bot","Hi — I am Supra 50M, a 50-million-parameter model running entirely in your browser. I make things up cheerfully and often; treat me as a curiosity, not a reference. What shall we talk about?");
  // When greeting is off the chat starts empty, so your first message is the first
  // turn the model sees. (The greeting was always display-only — never in history.)
}

(async()=>{
  try{ await setupRuntime(); await init(); }
  catch(err){ console.error(err); setStatus("error","Load failed — see console"); addMsg("error","Error: "+(err&&err.message||err)); }
})();

// ════════════ GENERATION ════════════
let generating=false, stopFlag=false;
// Watchdog: if no new token arrives within STALL_MS while streaming, abort.
// Note: this can only fire between tokens (when the event loop is free). It
// cannot interrupt a fully-synchronous main-thread freeze (e.g. WebGPU stuck
// on an unsupported int8 op) — nothing on the main thread can, short of a Worker.
const STALL_MS = 30000;
let watchdog=null, lastTokenTime=0, stallAbort=false;
function startWatchdog(){
  lastTokenTime = performance.now();
  stopWatchdog();
  watchdog = setInterval(()=>{
    if(generating && performance.now()-lastTokenTime > STALL_MS){
      stallAbort=true; stopFlag=true;   // next token boundary throws and unwinds
      stopWatchdog();
    }
  }, 1000);
}
function stopWatchdog(){ if(watchdog){ clearInterval(watchdog); watchdog=null; } }
async function run(){
  if(generating || !generator) return;
  const text = input.value.trim();
  if(!text) return;
  input.value=""; input.style.height="46px";
  generating=true; stopFlag=false; stallAbort=false;
  send.disabled=input.disabled=true; send.style.display="none"; stop.style.display="block";
  setStatus("gen","generating…");
  startWatchdog();

  history.push({role:"user", content:text});
  addMsg("user", text, history.length-1);

  const botDiv = addMsg("bot", "");
  botDiv.innerHTML = '<span class="caret"></span>';
  let out="", nTok=0; const t0=performance.now();

  try{
    const prompt = buildPrompt(history);
    const streamer = new TextStreamer(generator.tokenizer,{
      skip_prompt:true, skip_special_tokens:true,
      callback_function: tok=>{
        if(stopFlag) throw new Error(stallAbort ? "__STALL__" : "__STOP__");
        lastTokenTime = performance.now();
        out += tok; nTok++;
        botDiv.innerHTML = mdToHtml(out) + '<span class="caret"></span>';
        chat.scrollTop = chat.scrollHeight;
      },
    });
    await generator(prompt,{
      max_new_tokens: settings.max_new_tokens,
      do_sample: settings.temperature>0,
      temperature: Math.max(settings.temperature, 1e-3),
      top_p: settings.top_p,
      top_k: settings.top_k,
      repetition_penalty: settings.repetition_penalty,
      streamer,
    });
  }catch(err){
    stopWatchdog();
    if(err.message!=="__STOP__" && err.message!=="__STALL__"){
      botDiv.className="msg err"; botDiv.textContent="Error: "+err.message; finishGen(); return;
    }
  }
  stopWatchdog();

  const secs=(performance.now()-t0)/1000;
  const final = cleanOutput(out) || "[no output]";
  const tag = stallAbort ? " · timed out (no token in "+(STALL_MS/1000)+"s)" : (stopFlag?" · stopped":"");
  botDiv.dataset.meta = nTok+" tok · "+secs.toFixed(1)+"s · "+(nTok/secs||0).toFixed(1)+" tok/s"+tag;

  history.push({role:"assistant", content:final});
  fillMsg(botDiv, "bot", final, history.length-1);  // adds meta + edit/copy tools
  saveHistory();
  finishGen();
}
function finishGen(){
  stopWatchdog();
  generating=false; send.disabled=input.disabled=false;
  send.style.display="block"; stop.style.display="none";
  setStatus("ready","ready · "+activeDevice.toUpperCase());
  input.focus();
}

// ════════════ CONTROLS ════════════
send.addEventListener("click", run);
stop.addEventListener("click", ()=>{ stopFlag=true; });
input.addEventListener("keydown", e=>{
  if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); run(); }
});
input.addEventListener("input", ()=>{ input.style.height="46px"; input.style.height=Math.min(input.scrollHeight,150)+"px"; });

$("clear").addEventListener("click", ()=>{
  if(generating) return;
  history=[]; saveHistory(); chat.innerHTML="";
  if(settings.greeting) addMsg("bot","Cleared. Fresh start — what next?");
});
$("reload").addEventListener("click", async ()=>{
  if(generating) return;
  $("settings").classList.remove("open");
  send.disabled=input.disabled=true;
  try{ await init(); }catch(e){ setStatus("error","reload failed"); }
});
</script>
</body>
</html>"""

# Runtime-setup fragments injected at // __RUNTIME_SETUP__
SETUP_CDN = r"""
setStatus("loading","Loading runtime from CDN…");
modepill.textContent = "cdn";
let pipeline, env, TextStreamer;
async function setupRuntime(){
  const m = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@__VER__/dist/transformers.min.js");
  pipeline = m.pipeline; env = m.env; TextStreamer = m.TextStreamer;
  env.allowRemoteModels = false;
  env.allowLocalModels  = true;
  env.localModelPath    = "http://supra-local/";
  env.backends.onnx.wasm.proxy = false;
  env.backends.onnx.wasm.numThreads = (typeof crossOriginIsolated!=="undefined" && crossOriginIsolated)
      ? (navigator.hardwareConcurrency||4) : 1;
  // CDN mode lets ORT fetch its own .wasm from jsDelivr (cached after first load).
}
"""

SETUP_OFFLINE = r"""
setStatus("loading","Starting embedded runtime…");
modepill.textContent = "offline";
const ORT_MJS_B64  = "PLACEHOLDER_ORT_MJS_B64";
const ORT_WASM_B64 = "PLACEHOLDER_ORT_WASM_B64";
const TJS_B64      = "PLACEHOLDER_TJS_B64";
let pipeline, env, TextStreamer;
async function setupRuntime(){
  // ESM import from a data: URL works on file://; blob: is not reliably importable there.
  const tjsUrl = "data:text/javascript;base64," + TJS_B64;
  const m = await import(tjsUrl);
  pipeline = m.pipeline; env = m.env; TextStreamer = m.TextStreamer;

  env.allowRemoteModels = false;
  env.allowLocalModels  = true;
  env.localModelPath    = "http://supra-local/";

  // file:// is a unique origin → ORT's proxy Worker can't be created from it.
  env.backends.onnx.wasm.proxy = false;
  // SharedArrayBuffer (multi-thread) needs cross-origin isolation, impossible on file://.
  env.backends.onnx.wasm.numThreads = (typeof crossOriginIsolated!=="undefined" && crossOriginIsolated)
      ? (navigator.hardwareConcurrency||4) : 1;

  // Hand ORT its glue via a data: URL and the wasm bytes via wasmBinary, so it
  // never performs a network fetch for either. (Verified against ORT source.)
  env.backends.onnx.wasm.wasmPaths = {
    mjs:  "data:text/javascript;base64," + ORT_MJS_B64,
    wasm: "http://supra-local/ort-wasm-simd-threaded.jsep.wasm",  // dummy; wasmBinary wins
  };
  env.backends.onnx.wasm.wasmBinary = b64ToBytes(ORT_WASM_B64).buffer;
}
"""

# Fetch interceptor (shared) — serves the embedded model/config/tokenizer.
FETCH_INTERCEPTOR = r"""
const FAKE_HOST = "http://supra-local/";
const _fetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (inp, opts) => {
  const url = typeof inp==="string" ? inp : inp.toString();
  const c = url.split("?")[0];
  const J = o => { const s=JSON.stringify(o);
    return new Response(s,{status:200,headers:{"Content-Type":"application/json","Content-Length":String(s.length)}}); };
  if(c.includes("config.json") && !c.includes("tokenizer")) return J(CONFIG);
  if(c.includes("tokenizer_config"))                        return J(TOKENIZER_CFG);
  if(c.includes("tokenizer.json"))                          return J(TOKENIZER);
  if(c.includes("generation_config"))                       return J(GEN_CONFIG);
  if(c.endsWith(".onnx")) return new Response(modelBytes.slice(0),{status:200,
      headers:{"Content-Type":"application/octet-stream","Content-Length":String(modelBytes.byteLength)}});
  if(c.startsWith(FAKE_HOST)) return new Response("Not found",{status:404});
  return _fetch(inp,opts);
};
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", nargs="?", default=".")
    ap.add_argument("output", nargs="?", default="supra50m_chat.html")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--offline", dest="offline", action="store_true", default=True)
    g.add_argument("--cdn", dest="offline", action="store_false")
    ap.add_argument("--model-file", default="model_int8.onnx")
    ap.add_argument("--runtime-dir", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--tjs-version", default="3.8.1")
    args = ap.parse_args()

    md = args.model_dir
    p = lambda n: os.path.join(md, n)
    runtime_dir = args.runtime_dir or md

    # ── model + tokenizer ──
    print("Reading model...", flush=True)
    with open(p(args.model_file), "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("ascii")
    print(f"  model b64: {len(model_b64)/1e6:.1f} MB", flush=True)

    with open(p("tokenizer.json"), encoding="utf-8") as f:
        tokenizer_raw = f.read()
    with open(p("config.json"), encoding="utf-8") as f:
        config = json.load(f)
    config["use_cache"] = True   # PATCH: ONNX exported with KV-cache despite config

    tokenizer_config = {
        "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False, "model_max_length": 5120,
        "tokenizer_class": "PreTrainedTokenizerFast",  # PATCH: generic fast class
    }
    with open(p("generation_config.json"), encoding="utf-8") as f:
        gen_config = json.load(f)
    gen_config["eos_token_id"] = 2

    config_json  = json.dumps(config,           separators=(",", ":"))
    tok_cfg_json = json.dumps(tokenizer_config,  separators=(",", ":"))
    gen_cfg_json = json.dumps(gen_config,        separators=(",", ":"))

    # ── runtime setup fragment ──
    if args.offline:
        rt = get_runtime(runtime_dir, args.tjs_version, not args.no_download)
        tjs_b64  = base64.b64encode(rt[TJS_JS]).decode("ascii")
        mjs_b64  = base64.b64encode(rt[ORT_MJS]).decode("ascii")
        wasm_b64 = base64.b64encode(rt[ORT_WASM]).decode("ascii")
        print(f"  runtime b64: js {len(tjs_b64)/1e6:.1f}MB · "
              f"glue {len(mjs_b64)/1e6:.2f}MB · wasm {len(wasm_b64)/1e6:.1f}MB", flush=True)
        setup = SETUP_OFFLINE
        setup = setup.replace('"PLACEHOLDER_ORT_MJS_B64"',  '"' + mjs_b64 + '"')
        setup = setup.replace('"PLACEHOLDER_ORT_WASM_B64"', '"' + wasm_b64 + '"')
        setup = setup.replace('"PLACEHOLDER_TJS_B64"',      '"' + tjs_b64 + '"')
        runtime_setup = setup + "\n" + FETCH_INTERCEPTOR
    else:
        runtime_setup = (SETUP_CDN.replace("__VER__", args.tjs_version)
                         + "\n" + FETCH_INTERCEPTOR)

    # ── assemble ──
    print("Substituting placeholders...", flush=True)
    html = TEMPLATE.replace("// __RUNTIME_SETUP__", runtime_setup)
    html = html.replace('"PLACEHOLDER_MODEL_B64"', '"' + model_b64 + '"')
    html = html.replace("PLACEHOLDER_CONFIG",        config_json)
    html = html.replace("PLACEHOLDER_TOKENIZER_CFG", tok_cfg_json)   # CFG before TOKENIZER
    html = html.replace("PLACEHOLDER_TOKENIZER",     tokenizer_raw)
    html = html.replace("PLACEHOLDER_GEN_CONFIG",    gen_cfg_json)

    print(f"Writing {len(html)/1e6:.1f} MB → {args.output}", flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    sz = os.path.getsize(args.output)/1e6
    print(f"Done!  {sz:.1f} MB written to {args.output}", flush=True)
    print()
    if args.offline:
        print("FULLY OFFLINE build — no network needed, ever. Open in Chrome.")
        print("First init takes a few seconds (decoding ~%d MB of embedded data)." % sz)
    else:
        print("CDN build — needs internet the first time to fetch ~22 MB of runtime,")
        print("then works offline via browser cache. Open in Chrome.")


if __name__ == "__main__":
    main()
