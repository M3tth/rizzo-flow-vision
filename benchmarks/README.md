# Fixture e limiti

`smoke.jsonl`: 17 richieste, 20 decisioni etichettate. Include booleani, routing, score,
lettura di valori numerici su ancore, dati assenti, valori fuori scala e un ticket in italiano.
`perturbations.jsonl`: 9 varianti con ordine delle scelte invertito o contesto irrilevante.
Gli ID delle opzioni permettono di confrontare lo stesso significato dopo un riordino.

Sono fixture di sviluppo. Sono state lette durante l'implementazione e hanno contribuito
alla revisione del prompt: **non costituiscono un test indipendente o una misura generale
delle capacità del modello**. Le quantità numeriche sono semplici esempi di lettura,
non previsioni del mercato o problemi di regressione su dati reali.

Le versioni `*-v1.jsonl` conservano una domanda iniziale ambigua:
"Does the supplied record explicitly confirm that the invoice has been paid?".
Con stato di pagamento non registrato, "no" è un'interpretazione ragionevole,
ma l'etichetta iniziale chiedeva `__insufficient__`. La versione corrente chiede invece
se la fattura è stata pagata, specificando che uno stato assente è indeterminato.
Non confrontare direttamente le accuracy ottenute sulle due formulazioni.

I report registrano l'hash del dataset. Non cambiare retroattivamente risposte, etichette
o misure nei report salvati. Per nuovi esperimenti usare un percorso di output nuovo.

Un test di produzione dovrebbe includere dati separati da quelli usati per scrivere
prompt o calibrare probabilità, casi difficili e mancanti, lingue del dominio,
perturbazioni, astensioni corrette/errate e costi operativi degli errori.
