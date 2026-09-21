# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cos'è

Rizzo Flow: versione locale del pattern di **Jev** (TypeSafe, "System One": state non strutturato →
decisioni tipizzate con probabilità, zero generazione di testo), ispirata a **SemIf**
(`~/Git-projects/SemIf`, commit `ca3ba65`). Nessun training: si leggono i logit delle sole lettere
di risposta in un forward pass. Modello: **XHToken/Spark-X2.5-4B** (architettura `spark2_5`, pesi
originali, rev. `0bcb356…`) su **MLX / Apple Silicon** (sviluppo su M4 Pro 24 GiB). Precisione
BF16 di default; Q8/Q4 quantizzati in memoria (affine, group size 64). Config d'uso normale: **Q8**.
`config.MODELS` elenca i checkpoint supportati: `4b` (default) e `1.7b` (XHToken/Spark-X2.5-1.7B, rev.
`14d6e83…`, stessa architettura/tokenizer/contesto 1M, ~3.4 GB). CLI: `--size 4b|1.7b` su
`download/decide/serve/evaluate`; `backend.load` riconosce il checkpoint da `hidden_size`
(`config.identify`) e l'ID servito diventa `rizzo-spark-x2.5-1.7b-q8`. **Il supporto 1.7B è stato
aggiunto senza eseguire alcun test, né unitario né sul modello** (richiesta esplicita dell'utente).

README, doc e messaggi all'utente sono in italiano; codice, commenti e docstring in inglese.
Repository pubblico: <https://github.com/Rizzo-AI-Academy/rizzo-flow> (remote `origin` via SSH, branch `main`;
il README pubblico è in inglese, quello italiano storico è in `docs/README.it.md`).

## Comandi

```bash
uv sync --extra mlx --extra test --locked        # setup
.venv/bin/rizzo download                          # pesi (~8 GB) in models/Spark-X2.5-4B
.venv/bin/pytest -q                               # 29 test, ~1 s, nessun peso richiesto
.venv/bin/pytest tests/test_compat.py::test_systemone_wire_shape   # test singolo
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts
.venv/bin/rizzo serve --bits 8                    # API + playground su 127.0.0.1:8017
.venv/bin/rizzo decide examples/ticket.json --bits 8
.venv/bin/rizzo evaluate benchmarks/smoke.jsonl --compare-modes --output results/local-x.json
.venv/bin/rizzo schema > request.schema.json      # rigenerare dopo modifiche a schema.py
.venv/bin/python scripts/validate_checkpoint.py --bits 8 --output results/local-validation
```

`.claude/launch.json` definisce il server di anteprima `rizzo-q8` (porta 8017). Caricare il
modello richiede ~15 s e ~5 GiB (Q8) o ~8.4 GiB (BF16): **un solo processo con pesi alla volta**,
e fermare il server prima di misurare tempi.

I test non caricano mai il checkpoint 4B: `test_service.py`/`test_compat.py` usano `FakeBackend`
+ `CharacterTokenizer` (il fake favorisce sempre il secondo candidato); `test_mlx.py` usa la vera
architettura Spark ridotta con pesi casuali. La verifica sul modello reale si fa a mano (server +
curl/playground) o con `validate_checkpoint.py`.

## Architettura (flusso di una richiesta)

`schema.py` (contratto Pydantic strict, `extra=forbid`) → `prompts.compile_request` →
`backend.SparkBackend.score` → `decisions.decode` → `responses.py` (la risposta è ri-validata
prima di uscire). `engine.Engine` orchestra tutto sotto un `Lock` (un solo modello residente:
richieste HTTP concorrenti sono serializzate; il parallelismo è *dentro* la richiesta).

- **Slot a token singolo.** Ogni candidato (opzioni + speciali `__insufficient__`,
  `__below_range__`, `__above_range__`) è una lettera maiuscola A–Z. `prompts.py` verifica che
  ogni lettera sia un token singolo e che `encode(prompt + lettera) == tokens + [id]`. Da qui il
  limite `MAX_SLOTS = 26` in `schema.py` (`require_slots`): 26 opzioni/livelli con
  `allow_abstain: false`, 25 con astensione, 24/23 ancore per `numeric`. SemIf usa lo stesso
  trucco con 16 lettere. Per andare oltre servirebbero etichette a 2 token (chain rule),
  sì/no per opzione, o etichette `AA…ZZ` (497/676 sono token singoli in Spark) — non implementato,
  decisione dell'utente: restare a 26.
- **Prefisso condiviso.** Messaggio user = `render_state(state)` + `render_question(...)`.
  Lo state è identico per tutte le domande → prefill una volta (blocchi da 512), confine del
  prefisso verificato token per token (ultimo token scartato per i merge BPE, mai dedotto dalla
  lunghezza). `branch_cache` clona le cache native (attenzione piena + sliding-window rotante),
  i suffissi vanno in microbatch (`--batch-size`, default 4, max 16) ordinati per lunghezza con
  padding a destra; si legge l'ultima posizione reale. Cache scartata a fine richiesta.
  `mode: "direct"` disattiva il riuso (riferimento di verifica).
- **Proiezione selettiva.** `selected_logits` moltiplica l'hidden state solo per le righe di
  vocabolario delle lettere ammesse (anche con pesi quantizzati): delta 0 rispetto al vocabolario
  pieno.
- **Decodifica pura.** `decisions.py` è deterministico e senza I/O: softmax (con temperatura),
  status `ok | insufficient_evidence | out_of_range | uncertain` da `policy`, medie pesate per
  `score`/`numeric` condizionate alle sole opzioni valide.
- **Calibrazione.** `calibration.py`: temperature scaling per tipo, legato a un `fingerprint`
  (hash di pesi, tokenizer, precisione, runtime, `PROMPT_VERSION`). **Cambiare il prompt impone
  di incrementare `PROMPT_VERSION`**, il che invalida le calibrazioni esistenti (voluto).
- **Template Spark.** Il chat template antepone `you are a helpful assistant.\n\n` al system
  prompt e, con `enable_thinking=False`, il prompt termina con `<|Bot|></think>`: la lettera è il
  primo token dopo `</think>`, senza spazio (`"A"` = 46, `" A"` = 401 sono token diversi).

### Due API nello stesso server (`api.py`)

- `POST /v1/decisions` — nativa: `boolean | choice | score | numeric`, astensione, `policy`,
  logit, statistiche, hash dei prompt.
- `POST /v1/systemone`, `GET /v1/models` — **stessa interfaccia di TypeSafe/Jev**
  (<https://docs.typesafe.ai/api>), implementata in `compat.py` come traduzione verso le primitive
  native: `noul`→`boolean`, `choice` (chiavi libere → ID posizionali `o{i}` e ritorno), `score`
  (max 10 livelli). Sempre `allow_abstain: false`. `instructions`/`criteria` strutturati →
  JSON canonico. `confidence = (n·p_max − 1)/(n − 1)` (formula della demo nella doc TypeSafe; quella
  reale non è pubblica, non è calibrata). `model` accetta `rizzo-latest`, l'ID locale e qualunque
  `jev-*`; la risposta riporta **sempre** l'ID locale (`rizzo-spark-x2.5-4b-q8`), mai Jev.
  `usage.output_tokens` è sempre 0. Tempi/fingerprint in `x_rizzo` (estensione; gli SDK TypeSafe
  ignorano campi extra). Bearer auth solo se è impostata `RIZZO_API_KEY`. Pensato per funzionare
  con gli SDK ufficiali via `TYPESAFE_BASE_URL=http://127.0.0.1:8017` (non ancora provato con
  l'SDK reale).
- `GET /playground` (`playground.html`, pagina singola senza dipendenze esterne, servita dal
  package): builder noul/choice/score, esempi, editor JSON per entrambi gli endpoint, barre di
  probabilità, metriche (round-trip, inferenza, prefill, microbatch, token in cache), cURL.
  Bilingue IT/EN: dizionario `I18N` + `t(key, vars)` nello script, attributi `data-i18n`,
  `data-i18n-html`, `data-i18n-attr`; ogni nuova stringa UI va aggiunta in entrambe le lingue.
  Logo servito da `/playground/logo.png` (`src/rizzo_flow/logo.png`, copia ridotta di `assets/`).

## Regole del progetto

- **Onestà sulle misure.** Probabilità dichiarate non calibrate; mai presentarsi come Jev; mai
  dichiarare superiorità su Jev/SemIf senza dati. Ogni limite osservato va scritto nei README.
- **Risultati create-only.** `cli.write_json` e gli script aprono i file in modalità `"x"`: non
  sovrascrivere né riscrivere report in `results/`; per nuovi esperimenti usare un percorso nuovo
  (`results/local-*` è ignorato). `results/SHA256SUMS` copre i report storici.
- **Niente troncamento silenzioso.** Input oltre i limiti → `ValueError` → HTTP 422.
- **Non ottimizzare sul test.** Le fixture proprie (`benchmarks/smoke.jsonl`) sono già state usate
  per rivedere il prompt (vedi `benchmarks/README.md`). Per il lavoro sul prompt esiste uno split
  dev/held-out (sotto): l'held-out si guarda una volta sola.

## Stato del lavoro (21 settembre 2026)

### Fatto
1. API compatibile Jev + playground + test (`compat.py`, `api.py`, `playground.html`,
   `tests/test_compat.py`). Verificato con pesi reali Q8: 3 domande ≈ 540 ms; 8 noul = 1 prefill
   (126 token, 184 ms) + 2 microbatch, 932 ms totali, 0 token generati.
2. Limite portato da 23 a 26 slot. Verificato sul modello reale: 26 opzioni, slot Z scelto
   correttamente, ≈ 610 ms.
3. `scripts/semif_compare.py`: esegue le fixture di SemIf (`authored144`, `perturbations108`,
   `shape777`; hash verificati identici a quelli pubblicati) con il **loro** `evaluate.py` e lo
   stesso perimetro di tempo (modello caldo; prompt+tokenizzazione+forward+readout inclusi;
   caricamento e scrittura esclusi). `--system rizzo|semif`.
4. `prompts.py` rifattorizzato in `render_state` / `render_question` (comportamento v2 invariato,
   `PROMPT_VERSION = "spark-decisions-v2"`), così le varianti si provano per monkeypatch.

### Risultati Rizzo Q8 su fixture SemIf (prompt v2) — `results/semif-compare/rizzo-q8/`

| Misura | Rizzo Q8 (M4 Pro) | SemIf Qwen3.5-4B Q8 pubblicato (M5 Max) |
| --- | ---: | ---: |
| authored144, balanced accuracy media per famiglia | 0.758 | 0.819 |
| perturbations108 | 0.706 | 0.766 |
| Latenza per decisione, stato corto, p50 / p95 | 254 / 259 ms | non confrontabile |
| shape777 shared (37 stati ~2k token × 21 criteri) | 3.92 dec/s, 5.33 s/stato | non confrontabile |
| shape777 direct (3 stati) | 0.31 dec/s | non confrontabile |
| Cambi argmax shared vs direct (63 decisioni) | 0 (max Δp 0.057) | — |

Shared ≈ 12.6× direct. Famiglia debole: `rule_application` (0.689; 0.481 sulle perturbazioni,
NLL 1.83 = errori molto sicuri). I tempi SemIf vengono da un altro Mac: **valgono solo per la
qualità**. Le etichette SemIf non sono adjudicate da umani; 6 punti ≈ 9 righe.
Differenze volute: ogni sistema usa il proprio prompt; WANLI/TypeSafe/Every non inclusi.

Risultati storici sulle fixture proprie (smoke Q8: mediana 304 ms, 18/20, picco 4.88 GiB; stato
lungo shared 2.8× direct) e limiti osservati: `results/README.md`.

### In sospeso: miglioramento del prompt (interrotto dall'utente)

`scripts/prompt_lab.py dev|held [varianti]` — split per `group_id` alternato dentro ogni famiglia
(72 base + 54 perturbate per parte), Q8, più lo smoke proprio come controllo sui tipi nativi.
Log dei due giri su **dev**: `results/prompt-lab/`.

| Variante (dev) | base | perturbato | flip inversione ordine | smoke |
| --- | ---: | ---: | ---: | ---: |
| `v2-current` (system vecchio, state e domanda in JSON) | 0.754 | 0.711 | 6 | 0.90 |
| `a-text-all` (system nuovo, `<evidence>` testo, MCQ testo) | 0.827 | 0.852 | 2 | 0.95 |
| `i-systemA-json-mcq` (system nuovo, state JSON, MCQ testo) | 0.806 | 0.852 | 1 | 0.95 |
| `b` solo domanda in testo | 0.784 | 0.852 | 1 | 0.90 |
| `c` solo system nuovo | 0.811 | 0.796 | 2 | 0.95 |
| con più istruzioni (`f`, `g`, `k`: regole, astensione, nota ordine) | 0.79–0.81 | 0.80–0.82 | 2–3 | 0.90–0.95 |

Conclusioni: la domanda come scelta multipla in testo ("Question:", "A. …", "Answer with the
letter of the best option.") e un system prompt corto e orientato alla decisione aiutano entrambi
e riducono il bias di posizione; allungare il system prompt non aiuta. `a` e `i` sono pari entro
il rumore (1 riga ≈ 1.4 punti); candidata preferita `i` (NLL e stabilità migliori, state in JSON
più robusto a testo che imita i delimitatori; usa `json.dumps` senza `sort_keys` per conservare
l'ordine delle chiavi dell'utente).

**Non fatto:** l'held-out non è mai stato eseguito (`prompt_lab.py held v2-current,i-systemA-json-mcq`);
il prompt spedito è ancora v2. Per adottare la variante: portare `SYSTEM_A`, `json_state`, `mcq`
in `prompts.py`, incrementare `PROMPT_VERSION` a v3, rieseguire test e smoke, rilanciare
`semif_compare.py` in una nuova directory di output e aggiornare i README. I numeri dev non sono
un risultato: sono serviti a scegliere.

### Da fare
- Lato SemIf del confronto sullo stesso Mac: serve scaricare `Qwen/Qwen3.5-4B` (~9 GB, rev.
  `851bf6e…`) e un venv separato con `mlx==0.32.2` + `mlx-lm` al commit `a63e24c` (diverso da
  quello di questo progetto). **Chiedere conferma all'utente prima di scaricare.** Poi
  `semif_compare.py --system semif --bits 8` con `PYTHONPATH`/venv di SemIf.
- Probabilità molto concentrate (spesso 0.9999) → la `confidence` compatibile Jev è quasi sempre
  ≈ 1: serve temperature scaling (`rizzo calibrate`) su dati del dominio prima di usare soglie.
- Bias di posizione residuo: debiasing per permutazione (un microbatch in più) non implementato.
- Il modello sceglie poco astensione/fuori scala (casi documentati in `results/README.md`).
