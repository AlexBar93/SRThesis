# Immagini digitali

Nel mondo del processing di immagini digitali, queste vengono solitamente rappresentate come tensori tridimensionali dove le tre dimensioni rappresentano altezza, larghezza e numero di canali dell'immagine. Ogni elemento del tensore ha un valore numerico che a seconda del formato dell'immagine può essere compreso nell'intervallo [0,255] o nell'intervallo [0,1]. Altri formati meno utilizzati hanno range diversi, ma la scelta di come rappresentare l'immagine digitale è solitamente lasciata all'utente. Byron si appoggia alla libreria OpenCV [@opencv_library] per caricare e salvare in memoria velocemente le immagini, ma ha un oggetto proprio per la loro elaborazione una volta che sono state importate, dotato di tutte le funzioni più comuni di elaborazione immagini. La necessità di utilizzo di un oggetto separato da OpenCV è dovuta principalmente alla parallelizzazione: in Byron una sessione parallela viene aperta all'inizio del programma principale e viene chiusa alla fine. Appoggiarsi a funzioni esterne per il calcolo parallelo come per esempio quelle implementate nella libreria OpenCV creerebbe ulteriori sessioni parallele che verrebbero aperte e chiuse ad ogni elaborazione delle immagini. Ciò è da evitare in quanto l'apertura e la chiusura di una sessione parallela richiede un tempo finito in cui bisogna creare i thread e distribuirli sui core o raggrupparli ed è quindi meglio rimanere all'interno di una singola sessione parallela per tutta la durata di esecuzione del programma.

## Super-risoluzione

La super-risoluzione è una tecnica in generale utilizzata per migliorare la risoluzione spaziale di un'immagine. Può essere applicata nella sua forma base solo a immagini che contengono aliasing, ovvero che sono state sotto-campionate o ridotte in dimensione. In queste immagini il contenuto ad alta frequenza dell'immagine ad alta risoluzione (HR) desiderata è nascosto nel contenuto a bassa frequenza dell'immagine a bassa risoluzione (LR) di input. Di conseguenza applicare algoritmi per ripristinare questo contenuto (eventualmente dopo aver ingrandito l'immagine alle giuste dimensioni) consente di riprodurre immagini abbastanza simili all'output desiderato.

I primi metodi di super-risoluzione di immagini digitali consistevano nella stima del contenuto ad alta frequenza tramite associazione di patch di immagine LR con la corrispettiva immagine HR. Queste patch (ovvero riquadri dell'immagine di dimensioni piccole, solitamente inferiori a 50x50), venivano prese dopo aver filtrato l'immagine con un filtro di edge detection (che trova i bordi degli oggetti) o dalla trasformata di Fourier dell'immagine per avere direttamente il contenuto in frequenza. In generale un filtro di elaborazione immagini è un operatore spaziale che esegue delle operazioni su un intorno di un pixel su cui viene applicato. Una volta imparato un "dizionario" di queste associazioni, da un insieme di coppie di immagini HR e LR note, era possibile applicarlo su un'immagine a LR e ottenerne una versione ad alta risoluzione. Si noti che in questo caso le immagini di input e output non avevano necessariamente dimensioni diverse: la risoluzione spaziale infatti dipende non solo dalle dimensioni dell'immagine ma anche dal passo di campionamento.

Nel 2014, grazie al lavoro del Department of Information Engineering dell'Università di Hong Kong, è nata l'idea di utilizzare i layer di convoluzione delle ormai sempre più popolari reti neurali per imparare in maniera automatica un analogo del dizionario delle patch, che in questo caso sarebbe rappresentato da un insieme di moltissime feature. Nacque così la prima ANN per super-risoluzione, chiamata SRCNN [@SRCNN] (Super-Resolution Convolutional Neural Network), che consisteva semplicemente in tre layer di convoluzione. Il primo estraeva le patch a LRdall'immagine, il secondo collegava queste patch organizzate in un vettore a molte dimensioni a un altro vettore che idealmente rappresentava le patch HR. Infine l'ultimo layer riorganizzava le patch ad alta risoluzione e le combinava per ottenere le immagini HR di output.

Da allora sono stati fatti molti progressi nella ricerca in questo campo, ma le idee fondamentali non sono cambiate. Semplicemente i modelli hanno molti più layer, grazie alla sempre crescente potenza di calcolo dei computer odierni, e utilizzano quindi contromisure per riuscire a gestire l'addestramento con un numero enorme di parametri.

## Preprocessing

Le reti per super-risoluzione seguono un metodo di addestramento supervisionato ed hanno quindi bisogno di un dataset di immagini di input e di output attesi. Nel caso della super-risoluzione l'immagine di input consiste di una versione ridimensionata dell'immagine di output, a bassa risoluzione. Questo introduce un fattore di aliasing che la rete dovrà quindi imparare a nullificare quando ingrandisce l'immagine, ripristinando i contenuti ad alta frequenza e quindi la risoluzione dell'output ottenuto. Queste reti sono molto grandi e hanno un numero di parametri dell'ordine $10^7$: di conseguenza se l'input fosse modificato con dei filtri per simulare i problemi realistici delle immagini reali come rumore e sfocature, la rete imparerebbe in parte anche a risolvere questi problemi.

Per le analisi svolte in questa tesi ho considerato solamente un fattore di scala di 4 e un metodo di riduzione dell'immagine bicubico. Inoltre seguendo la procedura standard viene rimossa la media (RGB) del dataset utilizzato durante l'addestramento prima di processare un'immagine con la rete, e dopo aver ottenuto l'immagine di output viene sommata nuovamente. Altre accortezze come riscalare tutti i valori dell'immagine per normalizzarli, che dipendono dal modello della rete e dai pesi usati, sono state tenute in considerazione per avere risultati ottimali.

### Ricampionamento bicubico

Il ricampionamento bicubico, che solitamente viene diviso in _upsampling_ e _downsampling_ o analogamente chiamati _upscaling_ e _downscaling_, consiste in un metodo di interpolazione per determinare i valori dei pixel di un'immagine dopo che questa è stata cambiata di dimensioni (rispettivamente aumentata o diminuita). Il nome deriva dalla complessità massima dell'algoritmo di interpolazione usato, in cui l'operazione più complicata eseguita in questo caso è appunto il cubo del valore di un pixel. L'interpolazione viene eseguita in un intorno di 4 pixel. In generale i filtri scelti per il ricampionamento bicubico appartengono a una famiglia con la seguente forma:

$$ k(x) = \frac{1}{6}\left\{\begin{matrix}
(12-9B-6C)|x|^{3} + (-18+12B+6C)|x|^{2} + & \\
+ (6-2B) & $se $ \ |x| < 1 \\
 (-B-6C)|x|^{3} + (6B+30C)|x|^{2} + & \\
(-12B-48C)|x| + (8B+24C) & $se $ \ 1 \leq |x| <2 \\
0 & $ altrimenti$
\end{matrix}\right.$$

Tra i più popolari cito le opzioni B=0, C=0.75 usata dalla libreria OpenCV [@opencv_library] o B=0, C=0.5 solitamente chiamato filtro di Catmull-Rom usato dalle librerie di Matlab [@matlabob].
Byron utilizza di default l'opzione (0, 0.75) ma può essere agevolmente cambiata secondo necessità. Inoltre in Byron ho anche implementato il filtro di Lanczos, che garantisce risultati migliori (soprattutto quando si parla di upsampling dove si vede effettivamente il miglioramento di qualità rispetto a un upsample lineare) e ha un intorno di 8 pixel.

Per applicare questi filtri bisogna ridimensionarli in modo che siano larghi quanto 4 pixel dell'immagine più piccola (quindi quella di partenza nel caso dell'upsampling e quella ridimensionata in caso di downsampling). Poi per ogni pixel dell'immagine obiettivo, bisogna calcolare tutti i contributi dei pixel che rientrano nel range del filtro per il pixel obiettivo e pesarli per il rispettivo valore del filtro a quella distanza dall'origine. Bisogna solitamente anche normalizzare i pesi del filtro in modo da non avere un aumento o una diminuzione della intensità complessiva dell'immagine.

## Qualità delle immagini

Per valutare la bontà delle immagini SR (super-resolution) in output dalla rete bisogna confrontarle con le originali HR prima che esse venissero ridimensionate. Ci sono varie misure della somiglianza tra immagini, tra cui PSNR e SSIM.

### PSNR

Il peak signal to noise ratio (PSNR) è una misura che solitamente viene adottata per misurare la bontà di compressione di un'immagine rispetto all'originale. Matematicamente viene definito come:

$$ PSNR = 20 \cdot  log_{10}\left ( \frac{max(I)}{\sqrt{MSE}} \right ) $$

dove $max(I)$ è il massimo valore assumibile dai pixel dell'immagine, solitamente 1 per immagini con valori decimali e 255 per immagini con valori interi. $MSE$ è il Mean Square Error e indica la discrepanza quadratica media fra i valori dell'immagine super-risoluta ed i valori dell'immagine originale, viene definito come:

$$ MSE = \frac{1}{WH}\sum_{i=0}^{W-1}\sum_{j=0}^{H-1}\left \| I(i,j) - K(i,j)\right \|^{2} $$

dove $W$ , $H$ sono rispettivamente larghezza e altezza dell'immagine e $I$, $K$ sono rispettivamente immagine originale e immagine super-risoluta.

Il PSNR è quindi il rapporto tra la massima potenza del segnale e il rumore di fondo. Viene solitamente espresso in decibel (dB) perchè le immagini hanno un intervallo dinamico ampio e quindi avere una scala logaritmica rende i numeri più gestibili.
Il PSNR è uno degli indici di qualità più popolari tra le immagini [@psnr_ssim], anche se non sempre ha un collegamento diretto con una qualità visuale percettibile dall'occhio umano. Faccio notare che sia EDSR che WDSR sono state addestrate per massimizzare questo valore (e quindi la verosimiglianza all'immagine originale). Una funzione diversa per l'ottimizzazione, per esempio la cosiddetta _visual loss_ che dovrebbe essere una misura della qualità visuale percepita dall'occhio umano, può dare risultati visivamente migliori anche se con PSNR peggiori. La scelta del criterio per misurare la qualità dell'immagine e di conseguenza la funzione da ottimizzare per la rete rimane al giorno d'oggi un dibattito aperto e con varie opzioni valide.

### SSIM

Un altro indice di qualità delle immagini molto usato è il _Structural SIMilarity index_ (SSIM), una funzione molto complessa che cerca di valutare la somiglianza strutturale tra due immagini e che tiene anche conto del miglioramento visivo valutabile dall'occhio umano. Nella figura \ref{ssimmat} è illustrato il diagramma dei calcoli necessari per ottenere il SSIM tra due immagini. Matematicamente può essere espresso come:

$$
SSIM(I, K) = \frac{1}{N}\sum_{i=1}^{N} SSIM(x_{i}, y_{i})
$$

dove abbiamo $N$ box dell'immagine di dimensioni arbitrarie, solitamente 11x11 o 8x8. Per ogni box il SSIM è calcolato come:

$$
SSIM(x, y) = \frac{(2\mu_{x}\mu_{y} + c_{1})(2\sigma_{xy}+c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2})}
$$

dove $\mu$ rappresenta la media, $\sigma^{2}$ la varianza, $\sigma_{xy}$ la covarianza, $c_{1}$ e  $c_{2}$ due parametri fissati per evitare divergenze al denominatore. Per calcolare il SSIM tra le immagini ho usato l'apposita funzione dalla libreria di Python per elaborazione ed analisi immagini *scikit_image*[@scikit_image].

![Diagramma di flusso delle istruzioni necessarie per il calcolo dell'indice SSIM tra due segnali x e y. \label{ssimmat}](immagini/ssim.png){ width=100% }

## Dataset DIV2K

Il dataset utilizzato durante l'addestramento e il test delle reti analizzate in questa tesi è il DIV2K [@div2k] (DIVerse 2K resolution high quality images), composto da 1000 immagini a risoluzione 2K. Questo dataset è stato realizzato appositamente per la NTIRE (New Trends in Image Restoration and Enhancement) challenge del 2017 ed è stato poi riutilizzato anche per le NTIRE degli anni successivi. La NTIRE challenge prende luogo ogni anno durante la conferenza CVPR (Computer Vision and Pattern Recognition) e ha come obiettivo l'avanzamento dello stato dell'arte nel campo dell'elaborazione di immagini digitali per vari compiti, tra cui la super-risoluzione. In questa gara vari team di ricerca propongono modelli di network con l'obiettivo di migliorare la risoluzione di un'immagine ridimensionata in vari modi. La squadra che riesce ad avere il PSNR medio sulle immagini di validazione del dataset DIV2K più alto si aggiudica il primo posto. I metodi di downsampling delle immagini cambiano ad ogni anno, ma comprendono generalmente il ricampionamento bicubico e un ricampionamento che simula l'acquisizione immagini di una fotocamera digitale.

Il dataset DIV2K è diviso in:

- Training: 800 immagini HR per l'addestramento delle reti con rispettive versioni riscalate (LR) con vari fattori di scala (2, 3 e 4) e con metodi di ricampionamento diversi;
- Validation: 100 immagini HR con rispettive versioni LR che vengono usate per la validazione e il test dai team per provare, controllare e migliorare il loro modello durante la gara;
- Test: 100 immagini LR su cui i team devono fare i test con il loro modello finale. I risultati SR su queste immagini verranno confrontati con le originali HR che non sono a disposizione del pubblico per valutare i concorrenti e stabilire il vincitore.
