# Risultati

## Tempi e performance

In questo capitolo illustrerò i vari risultati ottenuti confrontando i tempi di calcolo della rete per processare una immagine tra le varie reti e valutando i vari miglioramenti di qualità delle immagini.

### EDSR vs WDSR

Come prima analisi ho confrontato i tempi di calcolo delle due reti per super-risoluzione implementate. In figura \ref{edsrvswdsr} sono rappresentati i risultati ottenuti dopo aver utilizzato per 100 volte le reti su una singola immagine di dimensioni 510x339. Come si può notare la rete WDSR è molto più veloce, oltre un fattore 10 di velocità. Ciò è dovuto principalmente al fatto che questa versione della rete ha molti meno parametri della contendente, e i layer di convoluzione hanno quindi molti meno filtri e meno operazioni da svolgere. Tuttavia è stato dimostrato (ref) che la struttura della rete WDSR, grazie all'omissione dei layer finali di convoluzione dopo l'upsampling dell'immagine, a parità di parametri è notevolmente più efficiente della struttura della rete EDSR.

boxplot

### Numero di core

Come seconda analisi ho studiato l'andamento della velocità di calcolo in funzione del numero di core fisici utilizzati dalla macchina durante i test. In questo caso il confronto è tra 100 run della rete WDSR su una singola immagine di dimensioni 510x339. Come si può notare dal grafico in figura \ref{core}, non c'è una corrispondenza uno a uno tra numero di core e speedup. Ciò significa che, per esempio, raddoppiare il numero di core non comporta un raddoppiamento della velocità di calcolo. Questo andamento è normale in quanto all'aumentare del numero di core aumenta il tempo necessario in cui il master thread (che gestisce tutti gli altri) deve distribuire le informazioni necessarie per i calcoli ad ogni core o recuperare i risultati ottenuti per procedere al successivo ciclo di istruzioni. Sicuramente si può ridurre questo tempo di overhead gestendo meglio le funzioni della libreria OpenMP utilizzata per gestire il multi-threading.

catplot e spezzata collegata

### Byron vs Darknet

Come ultima analisi temporale ho confrontato la velocità di calcolo tra Byron e la libreria su cui è basata, Darknet. Visto che quest'ultima non ha implementato il layer di pixel-shuffle, non è possibile testare le reti per super-risoluzione come mezzo per valutare a parità di rete la velocità delle due librerie. Per questo motivo ho implementato anche in Byron una delle reti più utilizzate della libreria Darket, chiamata YOLO, di cui parlerò più approfonditamente nel prossimo capitolo. Di seguito riporto quindi lo speedup relativo di Byron per 100 run della rete YOLOv3 su una singola immagine di dimensioni 608x608. Come si può notare dal grafico in figura \ref{byvsdark}, c'è un aumento di velocità di circa un fattore 2. Ritengo che questo speedup abbia ampi margini di miglioramento in quanto Darknet non solo utilizza in modo non ottimale il multi-threading ma può anche essere migliorata dal punto di vista dell'implementazione dei layer più costosi in termini di tempi di calcolo come per esempio il layer di convoluzione, utilizzando algoritmi recenti quali il Winograd che sono risultati molto più efficienti dell'im2col (ref).

boxplot

## Qualità delle immagini

### PSNR

Il peak signal to noise ratio (PSNR) è una misura che solitamente viene adottata per misurare la bontà di compressione di un'immagine rispetto all'originale. Rappresenta il rapporto tra la massima potenza del segnale e il rumore di fondo. Viene solitamente espresso in decibel (dB) perchè le immagini hanno una gamma dinamica molto ampia e quindi avere una scala logaritmica rende i numeri più gestibili. Nel caso della super-risoluzione viene usato per confrontare l'immagine super-risoluta, in output dalla rete, con quella originale ad alta risoluzione prima che venisse ricampionata bicubicamente. Matematicamente viene definito come:

$$ PSNR = 20 \cdot  log_{10}\left ( \frac{max(I)}{\sqrt{MSE}} \right ) $$

dove $max(I)$ è il massimo valore assumibile dai pixel dell'immagine, solitamente 1 per immagini con valori decimali e 255 per immagini con valori interi. $MSE$ è il Mean Square Error e indica la discrepanza quadratica media fra i valori dell'immagine super-risoluta ed i valori dell'immagine originale. Matematicamente viene definito come:

$$ MSE = \frac{1}{WH}\sum_{i=0}^{W-1}\sum_{j=0}^{H-1}\left \| I(i,j) - K(i,j)\right \|^{2} $$

dove $W$ , $H$ sono rispettivamente larghezza e altezza dell'immagine e $I$, $K$ sono rispettivamente immagine originale e immagine super-risoluta.

Il PSNR è uno degli indici di qualità più popolari tra le immagini, anche se non sempre ha un collegamento diretto con una qualità visuale percettibile dall'occhio umano. Faccio notare che sia EDSR che WDSR sono state addestrate per massimizzare questo valore (e quindi la verosimiglianza all'immagine originale). Una funzione diversa per l'ottimizzazione, per esempio la cosiddetta visual loss che dovrebbe essere una misura della qualità visuale percepita dall'occhio umano, può dare risultati visivamente migliori anche se con PSNR peggiori. La scelta del criterio per misurare la qualità dell'immagine e di conseguenza la funzione da ottimizzare per la rete rimane al giorno d'oggi un dibattito aperto e con varie opzioni valide.

In figura \ref{psnr} riporto il confronto tra i PSNR misurati su 60 immagini del validation set del dataset DIV2K per tre diversi metodi di upsample: bicubico, super-risoluzione con WDSR e super-risoluzione con EDSR. Come si può notare c'è un notevole miglioramento nelle immagini super-risolute rispetto al semplice upsample bicubico. Tra le due reti invece, sebbene la differenza sia meno evidente, prevale la EDSR come qualità. Tuttavia è importante notare che la rete WDSR ha meno di 1/10 dei parametri della contendente, e quindi i risultati sono ragionevolmente peggiori. Se avessimo avuto lo stesso numero di parametri per le due reti, la struttura della WDSR avrebbe riportato risultati notevolmente migliori (ref). Ciò avrebbe comportato tuttavia un notevole aumento dei tempi di calcolo.

boxplot

### SSIM

Un altro indice di qualità delle immagini molto usato è il structural similarity index (SSIM), una funzione molto complessa che cerca di valutare la somiglianza strutturale tra due immagini e che tiene anche conto del miglioramento visivo valutabile dall'occhio umano. Nella figura \ref{ssimmat} è illustrato il diagramma dei calcoli necessari per ottenere il SSIM tra due immagini. Matematicamente può essere espresso come:

$$
SSIM(I, K) = \frac{1}{N}\sum_{i=1}^{N} SSIM(x_{i}, y_{i})
$$

dove abbiamo $N$ box dell'immagine di dimensioni arbitrarie, solitamente 11x11 o 8x8. Per ogni box il SSIM è calcolato come:

$$
SSIM(x, y) = \frac{(2\mu_{x}\mu_{y} + c_{1})(2\sigma_{xy}+c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2})}
$$

dove $\mu$ rappresenta la media, $\sigma^{2}$ la varianza, $\sigma_{xy}$ la covarianza, $c_{1}$ e  $c_{2}$ due parametri fissati per evitare divergenze al denominatore. Per calcolare il SSIM tra le immagini ho usato l'apposita funzione dalla libreria di python per calcoli statistici _skimage_.

![Diagramma per il calcolo dell'indice SSIM. \label{ssimmat}](immagini/ssim.png){ width=100% }

In figura \ref{ssim} riporto il confronto tra i SSIM misurati sulle stesse 60 immagini del validation set del dataset DIV2K utilizzate anche per calcolare il PSNR, ed anche in questo caso distinguendo i tre metodi impiegati. I risultati ottenuti sono concordi con le misure di PSNR precedentemente illustrate, e confermano che le reti per super-risoluzione migliorano notevolmente la qualità di un'immagine ricampionata ripristinandola fedelmente.

boxplot


### Confronto visuale

grafico tempi vs psnr di bicubica, edsr, wdsr

confronto patch bicubica / HR / EDSR / WDSR
