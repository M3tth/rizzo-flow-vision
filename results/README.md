# Risultati locali — 21 settembre 2026

Hardware: **Apple M4 Pro, 24 GiB** di memoria unificata. Modello originale Spark-X2.5-4B,
revisione e hash dei file registrati in ogni risposta. Runtime e dipendenze sono fissati.
Tutti i tempi sotto escludono caricamento del modello e warmup; includono compilazione
della richiesta e inferenza GPU sincronizzata. Non descrivono un servizio remoto.

## Esecuzioni correnti

| Misura | BF16 | 8 bit |
| --- | ---: | ---: |
| Mediana su 17 richieste smoke | 389 ms | 304 ms |
| p95 sulle stesse richieste | 1.256 s | 1.506 s |
| Picco allocazione MLX nello smoke | 8.38 GiB | 4.88 GiB |
| Argmax corretto, 20 decisioni smoke | 17/20 | 18/20 |
| Copertura: decisioni con status `ok` | 17/20 | 17/20 |
| Argmax corretto sulle 9 perturbazioni | 8/9 | 9/9 |
| Cambi argmax shared/direct nello smoke | 1/20 | 0/20 |

Report correnti: [BF16](spark-bf16-final/summary.json), [8 bit](spark-q8-final/summary.json).
Ogni directory contiene la risposta reale all'esempio API, tutte le risposte del benchmark,
logit, probabilità, tempi e differenze tra esecuzione diretta e condivisa.

Sono fixture di sviluppo piccole e semplici, usate anche durante la revisione del prompt.
**I dati non dimostrano che Q8 sia più accurato in generale, né superiorità su Jev o SemIf.**
Il p95 risente della richiesta con quattro domande, più lunga delle altre. Le ripetizioni
sono limitate; confronti temporali più forti richiedono un protocollo dedicato e più misure.
Le misure di memoria sono del solo allocatore MLX, non di tutto il processo macOS.

## Limiti osservati

- Anche con Q8, il modello risponde "no" anziché "dati insufficienti" in un caso di pagamento
  non registrato, e seleziona un'ancora interna anziché "sopra scala" per un prezzo esplicito di 900 EUR
  con ancore 100/200/300. Le opzioni di astensione e fuori scala sono disponibili e il codice le
  tratta correttamente, ma il modello può non selezionarle quando dovrebbe.
- BF16 sbaglia inoltre un booleano in italiano con probabilità molto vicine, circa 0.52 contro 0.48.
  Il confronto direct/shared può cambiare queste decisioni vicine: non è equivalenza bit per bit.
- La quantizzazione modifica 1 dei 20 argmax dello smoke; massimo spostamento di probabilità 0.1263.
  [Confronto di precisione](precision-comparison.json).
- Una perturbazione Q8 cambia scelta e status rispetto all'originale, anche se risulta corretta
  nella variante. Massimo spostamento di probabilità 0.7137. Il contesto irrilevante non è sempre innocuo.
  [Confronto per ID semantico](q8-stability.json).
- Nessuna probabilità è stata calibrata sui dati del dominio dell'utente. Il temperature fitting
  è implementato e testato, ma richiede un insieme di calibrazione e una verifica separata.

## Riuso di uno stato lungo

Misura con prompt v2, quattro domande e contesto oltre la finestra locale di 512 token:

| Precisione | Shared, mediana di 2 | Direct, 1 misura | Rapporto direct/shared |
| --- | ---: | ---: | ---: |
| BF16 | 3.123 s | 8.601 s | 2.75× |
| 8 bit | 3.482 s | 9.841 s | 2.83× |

Nessun argmax è cambiato su queste quattro domande. Probabilità comunque non identiche.
Questa misura appartiene ai report `spark-bf16-v2-validation/long-state.json` e
`spark-q8-v2-validation/long-state.json`, precedenti all'aggiunta della validazione formale
della risposta JSON e alla correzione di una diversa domanda nello smoke. Il codice del
calcolo MLX e il prompt v2 non sono cambiati. Qui Q8 risparmia memoria ma non tempo.

## Controlli eseguiti

- **24 test superati**, inclusi veri calcoli con l'architettura Spark ridotta e pesi casuali.
- Proiezione selettiva confrontata con il vocabolario completo in BF16, Q4 e Q8 nei test piccoli.
- Sul checkpoint 4B, delta massimo dei logit pari a **0** nella domanda usata per il controllo
  della proiezione, sia BF16 sia Q8.
- Cache sliding-window oltre il confine, isolamento delle copie, padding, riordino dei batch,
  ripetizioni, numeri non finiti, input invalidi, policy, fitting della temperatura e API.
- API provata anche con il checkpoint 4B reale e validazione dello schema delle risposte.
- Ruff passa. Due deprecation warning provengono dalle dipendenze di test FastAPI/Starlette;
  non sono fallimenti. Il tokenizer emette anche un avviso sulla configurazione custom Spark:
  l'inferenza usa esplicitamente l'implementazione MLX ufficiale, non AutoModel.

Q4 è disponibile e verificato nei test dell'architettura ridotta; non è stato eseguito
un benchmark di qualità del checkpoint 4B quantizzato a 4 bit.

## Storia ed evidenza

`spark-bf16-validation` conserva l'esperimento con prompt v1 e una copia dei sorgenti
di quell'esperimento in `source/`. `spark-*-v2-validation` conserva il prompt v2 con le
fixture iniziali. La domanda ambigua relativa al pagamento è documentata in
`../benchmarks/README.md`; le fixture originali sono conservate come `*-v1.jsonl`.
I report originali non sono stati riscritti e non vanno usati come risultato corrente.

`SHA256SUMS` verifica l'integrità dei report e delle copie storiche dei sorgenti:

```bash
cd results
shasum -a 256 -c SHA256SUMS
```

## Confronto con SemIf (in corso)

`scripts/semif_compare.py` esegue le fixture di SemIf (`authored144`, `perturbations108`,
`shape777`, commit `ca3ba65`) con le metriche di SemIf (`benchmarks/evaluate.py`) e lo stesso
perimetro di tempo: modello caldo; prompt, tokenizzazione, forward e readout inclusi;
caricamento e scrittura file esclusi. Ogni sistema usa il proprio prompt e il proprio modello.

Rizzo Flow, Spark-X2.5-4B **Q8**, M4 Pro 24 GiB — [report](semif-compare/rizzo-q8/report.json):

| Misura | Rizzo Q8 (M4 Pro) | SemIf Qwen3.5-4B Q8 pubblicato (M5 Max) |
| --- | ---: | ---: |
| authored144, balanced accuracy media per famiglia | 0.758 | 0.819 |
| perturbations108, stessa metrica | 0.706 | 0.766 |
| Latenza per decisione, stato corto (p50 / p95) | 254 / 259 ms | non confrontabile |
| shape777 shared, 37 stati × 21 criteri (~2k token) | 3.92 decisioni/s, 5.33 s per stato | non confrontabile |
| shape777 direct, 3 stati | 0.31 decisioni/s | non confrontabile |
| Cambi argmax shared/direct su 63 decisioni | 0 (max Δp 0.057) | — |

I valori SemIf vengono dal suo `results/mlx/2026-09-17-q8-fixed/summary.json`, misurati su un
altro Mac: **valgono per la qualità, non per i tempi**. Il confronto dei tempi richiede di
eseguire SemIf su questa macchina (`--system semif`), non ancora fatto. La famiglia più debole
di Rizzo è `rule_application` (0.689; 0.481 sulle perturbazioni, NLL 1.83: errori molto sicuri).
Le fixture non hanno etichette adjudicate da umani (dichiarato da SemIf) e sono piccole.
