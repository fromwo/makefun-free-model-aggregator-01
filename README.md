# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/fromwo/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/fromwo/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A zero-config free-model aggregator: one server speaking the OpenAI Chat Completions and Anthropic Messages APIs (JSON and streaming) over ten free-tier model providers. It reads the providers, the quality table and the workload from the trial brief, ranks every free model by its rating, routes each request to the best free model that can serve it, spreads calls across providers so no hidden per-trial budget is drained by the plan alone, fails over to the next-best candidate when a provider refuses (a provider is set aside only after its second refusal), and reports the catalog it found with `--list`. Upstream calls are plain JSON on the OpenAI, Anthropic and Gemini wires; the client's JSON or SSE answer is rendered from that one answer, so usage counts and stop reasons stay exact. No API keys are needed or stored: the providers' keys come from the trial brief. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

Local practice grades (kit `grade-local`, process driver, 3 practice trials): score 99 (quality 99, served 100, wire 100, failover 100, catalog 100), replays byte-for-byte identical.

## Run it

```sh
node main.ts --list   # print the catalog document and exit
node main.ts          # serve (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
```

Node 22 or later, standard library only; nothing to install. `main.ts` is the entrypoint, `lib/brief.ts` reads the brief and builds the catalog, `lib/translate.ts` holds the three wires and the answer rendering.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
