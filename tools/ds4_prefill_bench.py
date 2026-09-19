"""Prefill benchmark for the served DeepSeek endpoint: random-word prompts (no prefix-cache hits),
max_tokens=4. Usage: ds4_prefill_bench.py [WORDS ...]  (~2.7 tokens/word; default ladder below).
The 1M run used 365000 words (986k tokens)."""
import json,time,random,urllib.request,sys
words=open('/usr/share/dict/words').read().split()
def go(n):
    random.seed(n*7919+int(time.time()))
    p=" ".join(random.choice(words) for _ in range(n))+"\n\nReply with the single word: done."
    b=json.dumps({"model":"deepseek-v4-flash","messages":[{"role":"user","content":p}],"max_tokens":4,"temperature":0,"chat_template_kwargs":{"thinking":False}}).encode()
    t=time.time()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request("http://sparkone.local:8000/v1/chat/completions",b,{"Content-Type":"application/json"}),timeout=3600))
        w=time.time()-t; pt=r["usage"]["prompt_tokens"]; print(f"{pt:>7} tokens {w:7.1f}s {pt/w:6.0f} tok/s reply={r['choices'][0]['message']['content']!r}",flush=True)
    except Exception as e:
        print(f"{n} words FAILED after {time.time()-t:.0f}s: {e}",flush=True); sys.exit(1)
for n in ([int(a) for a in sys.argv[1:]] or [4000,8000,16000,32000,64000,128000]): go(n)
