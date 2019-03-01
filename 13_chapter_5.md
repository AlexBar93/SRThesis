# Risultati

## Tempi e performance

In questo capitolo illustrerò i vari risultati ottenuti confrontando i tempi di calcolo della rete per processare una immagine tra le varie reti e valutando i vari miglioramenti di qualità delle immagini.

### EDSR vs WDSR

Come prima analisi ho confrontato i tempi di calcolo delle due reti per super-risoluzione implementate. In figura \ref{edsrvswdsr} sono rappresentati i risultati ottenuti dopo aver utilizzato per 100 volte le reti su una singola immagine di dimensioni 510x339. Come si può notare la rete WDSR è molto più veloce, oltre un fattore 10 di velocità. Ciò è dovuto principalmente al fatto che questa versione della rete ha molti meno parametri della contendente e i layer di convoluzione hanno quindi molti meno filtri e meno operazioni da svolgere. Tuttavia è stato dimostrato [@wdsr] che la struttura della rete WDSR, grazie all'omissione dei layer finali di convoluzione dopo l'upsampling dell'immagine, a parità di parametri è notevolmente più efficiente della struttura della rete EDSR.

![Confronto delle velocità della rete WDSR e della rete EDSR, normalizzate sulla velocità media della rete EDSR, per 100 run su immagini 510x339. \label{edsrvswdsr}](immagini/edsr_wdsr.png){ width=100% }

### Numero di core

Come seconda analisi ho studiato l'andamento della velocità di calcolo in funzione del numero di core fisici utilizzati dalla macchina durante i test. In questo caso il confronto è tra 100 run della rete WDSR su una singola immagine di dimensioni 510x339. I test sono stati effettuati con 2, 4, 8, 16 e 32 core. Come si può notare dal grafico in figura \ref{core}, l'andamento della velocità in funzione del numero di core è parabolico, non lineare. Questo andamento è normale in quanto all'aumentare del numero di core aumenta il tempo necessario in cui il master thread (che gestisce tutti gli altri) deve distribuire le informazioni necessarie per i calcoli ad ogni core o recuperare i risultati ottenuti per procedere al successivo ciclo di istruzioni. In generale aumentare il numero di core non è solo dispendioso in termini economici ma anche energetici, soprattutto nel caso in cui sia necessario addestrare una rete, procedura che può richiedere svariati giorni. Da questo grafico si evince quindi un tradeoff tra utilizzo dei core della macchina e corrispettivo consumo energetico ed effettivo guadagno in velocità.

![Confronto della velocità della rete WDSR in funzione del numero di core utilizzati durante il forward della rete. Per ogni numero di core sono state effettuate 100 run su immagini 510x339. \label{core}](immagini/cores.png){ width=100% }

### Byron vs Darknet

Come ultima analisi temporale ho confrontato la velocità di calcolo tra Byron e la libreria su cui è basata, Darknet. Visto che quest'ultima non ha implementato il layer di pixel-shuffle, non è possibile testare le reti per super-risoluzione come mezzo per valutare a parità di rete la velocità delle due librerie. Tuttavia visto che il punto di forza di Darknet sulle altre librerie è la velocità nell'object detection con YOLO, ho confrontato le due librerie e calcolato lo speedup relativo di Byron per 100 run della rete YOLOv3 su una singola immagine di dimensioni 608x608. Come si può notare dal grafico in figura \ref{byvsdark}, c'è un aumento di velocità di circa un fattore 2. Inoltre dal grafico si evince anche che la distribuzione delle velocità di Byron è più piccata mentre quella di Darknet è più ampia: ciò significa che Byron è più consistente, fatto probabilmente dovuto alla diversa gestione dei core delle librerie. Ritengo che questo speedup abbia margini di miglioramento in quanto la libreria può ancora essere migliorata dal punto di vista dell'implementazione dei layer più costosi in termini di tempi di calcolo come per esempio il layer di convoluzione [@winograd].

![Confronto delle velocità della libreria Darknet e della libreria Byron, normalizzate sulla velocità media della libreria Darknet, per 100 run della rete YOLOv3 su immagini 608x608. \label{byvsdark}](immagini/darkvsby.png){ width=100% }

### PSNR e SSIM
In figura \ref{psnr} riporto il confronto tra i PSNR misurati su 60 immagini del validation set del dataset DIV2K per tre diversi metodi di upsample: bicubico, super-risoluzione con WDSR e super-risoluzione con EDSR. Come si può notare c'è un notevole miglioramento nelle immagini super-risolute rispetto al semplice upsample bicubico. Infatti un aumento di PSNR di 0.25 è già visibile a occhio nudo, come si può notare nelle figure \ref{examples}. Tra le due reti invece, sebbene la differenza sia meno evidente, prevale la EDSR come qualità. Tuttavia è importante ricordare che questa implementazione della rete WDSR ha meno di 1/10 dei parametri della contendente, e quindi i risultati sono ragionevolmente peggiori. Se avessimo avuto lo stesso numero di parametri per le due reti, la struttura della WDSR avrebbe riportato risultati notevolmente migliori [@wdsr]. Ciò avrebbe comportato tuttavia un notevole aumento dei tempi di calcolo: in questo caso c'è un tradeoff tra qualità del risultato e tempi richiesti.

![Confronto dei risultati ottenuti su 60 immagini del validation set del DIV2K in termini di PSNR in funzione del metodo di upsample utilizzato. In bianco viene riportato il valore medio del PSNR sul set in esame. \label{psnr}](immagini/psnr_violin.png){ width=100% }

In figura \ref{ssim} riporto il confronto tra i SSIM misurati sulle stesse 60 immagini del validation set del dataset DIV2K utilizzate anche per calcolare il PSNR, ed anche in questo caso distinguendo i tre metodi impiegati. I risultati ottenuti sono concordi con le misure di PSNR precedentemente illustrate, e confermano che le reti per super-risoluzione migliorano notevolmente la qualità di un'immagine ricampionata ripristinandola fedelmente.

![Confronto dei risultati ottenuti su 60 immagini del validation set del DIV2K in termini di SSIM in funzione del metodo di upsample utilizzato. In bianco viene riportato il valore medio del SSIM sul set in esame. \label{ssim}](immagini/ssim_violin.png){ width=100% }

### Confronto visuale

Nelle figure in questo paragrafo illustro in maniera qualitativa e riporto PSNR e SSIM per varie immagini che sono state super-risolute dal validation set del DIV2K, confrontando l'immagine originale con l'immagine LR dopo un ricampionamento bicubico e dopo aver applicato le due reti per super-risoluzione implementate.

![](immagini/0828_results.png){ width=100% }
![](immagini/0845_results.png){ width=100% }
![](immagini/0861_results.png){ width=100% }
![](immagini/0887_results.png){ width=100% }
![Confronto dei risultati ottenuti su alcune immagini del validation set del DIV2K in termini di PSNR e SSIM in funzione del metodo di upsample utilizzato. \label{examples}](immagini/0843_results.png){ width=100% }

## Super-risoluzione e object detection

Uno dei problemi principali di YOLO, oltre alla precisione nella localizzazione, è la detection di oggetti piccoli e vicini [@yolo]. Per questo motivo è plausibile aspettarsi che l'utilizzo di una rete per super-risoluzione per migliorare la qualità di un'immagine prima di applicarvi la rete YOLO per object detection ne migliori i risultati e la precisione. Per verificare questa ipotesi ho effettuato due test, entrambi su immagini contententi persone. In entrambi i casi ho analizzato con YOLO tre immagini: l'immagine di partenza LR (che viene ridimensionata dalla rete a 608x608 linearmente prima dei calcoli), l'immagine ridimensionata bicubicamente con scala 4 e l'immagine super-risoluta dalla rete EDSR. Nel primo caso YOLO opera su una patch di dimensioni 608x608 nelle immagini upsampled e di dimensioni 152x152 nell'immagine LR. Nel secondo caso la patch ha dimensioni rispettivamente 300x300 e 75x75. Si possono notare dalle figure \ref{yolotest1} e \ref{yolotest2} alcuni dettagli:

- Nel caso 1 le persone sono di dimensioni considerevoli e già nell'immagine a bassa risoluzione YOLO riesce a trovarne qualcuna. Il metodo di upsample bicubico non migliora il numero di detection ma migliora leggermente le probabilità di classificazione. Nell'immagine super-risoluta invece aumenta notevolmente il numero di detection e anche le probabilità di classificazione sono migliori.
- Nel caso 2 le immagini sono a una risoluzione inferiore e infatti YOLO non riesce a trovare nessuna persona nell'immagine LR. Con l'upsample bicubico vengono trovate solo 2 persone, mentre nell'immagine super-risoluta le detection aumentano a 7. Ciò valida l'ipotesi dell'efficacia della super-risoluzione applicata in congiunzione con l'object detection soprattutto nel caso di immagini piccole.

![Confronto dell'efficacia della rete YOLO per object detection su una patch LR di dimensioni 152x152 al variare del metodo di upsample utilizzato. \label{yolotest1}](immagini/yolo_big_results.png){ width=100% }

![Confronto dell'efficacia della rete YOLO per object detection su una patch LR di dimensioni 75x75 al variare del metodo di upsample utilizzato. \label{yolotest2}](immagini/yolo_small_results.png){ width=100% }

## Conclusioni

In questo lavoro di tesi ho implementato Byron, una libreria per reti neurali stato dell'arte in termini di performance nel campo dell'object detection e della super-risoluzione su CPU, ottimizzata per sistemi informatici con elevato numero di core, dimostrando anche una possibile applicazione congiunta delle due metodologie di elaborazione immagini al fine di migliorare i risultati della object detection.

Il miglioramento delle performance delle reti di object detection su immagini super-risolute è noto già da alcuni lavori [@underwatersr, @satellitesr], ma non è l'unico possibile campo di applicazione della super-risoluzione. Potendo addestrare i modelli delle reti su dataset particolari come per esempio immagini di microscopia o di risonanze magnetiche, probabilmente sarebbe possibile migliorarne la qualità per scopi pratici in campo medico. Anche in questo caso sono già state svolte delle ricerche (@srmicro) ma l'argomento è ancora una novità in moltissimi ambienti e quindi ha molto potenziale per essere sviluppato e molte possibili applicazioni.

Altri lavori futuri più centrati sulla libreria che sui modelli implementabili comprendono:

- Byron su tutti i sistemi operativi: per ora Byron è stato testato esaustivamente solo su ambiente Linux ma è stato progettato per essere multi piattaforma e compatibile anche con sistemi operativi Windows e Mac. Ulteriori test sono necessari per verificare la compatibilità e l'ottimizzazione su altri sistemi e sicuramente avere una libreria che funzioni in qualsiasi ambiente è un ottimo obiettivo da raggiungere.
- Byron su GPU: vero che l'idea alla base di Byron è l'ottimizzazione per CPU multi-core quali i server di bio-informatica, ma qualsiasi libreria per reti neurali che si rispetti deve avere anche un'implementazione su GPU di pari passo a quella CPU in modo da poter sfruttare qualsiasi hardware disponibile nel miglior modo possibile. A questo scopo sarebbe ideale implementare sia una versione in CUDA che una versione in OpenCL della libreria, in modo da garantirsi il funzionamento sulla maggioranza delle GPU moderne.
- Ottimizzazioni di codice: altri possibili miglioramenti su cui sto già lavorando riguardano l'implementazione e lo sviluppo di nuovi algoritmi per ottimizzare i layer più intensivi delle reti quali per esempio layer di convoluzione e batch-normalization. Uno di essi per esempio è l'algoritmo Winograd[@winograd] per la convoluzione che dovrebbe garantire un notevole speedup per i layer con filtri di dimensione 3x3 che ormai sono alla base della maggior parte delle reti per elaborazione immagini.
