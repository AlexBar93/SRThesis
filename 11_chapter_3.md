# Reti neurali

## Layer
I layer che compongono le reti neurali possono essere di varia natura, a seconda delle funzioni che devono svolgere. Nella libreria Byron ho implementato tutti quelli più comunemente utilizzati al momento, ma per lo scopo di questa tesi mi concentrerò solo su quelli necessari alla super-risoluzione e alla object detection.

### Convoluzione

La convoluzione è un'operazione matematica tra due funzioni che consiste nell'integrare la prima funzione con la seconda traslata di un certo valore. Viene indicato con convoluzione sia il risultato dell'operazione che l'operazione stessa. Nel campo dell'elaborazione immagini, la prima funzione è data dall'immagine mentre la seconda da un filtro che scorre su di essa, e viene utilizzata una versione discreta invece che continua.
Matematicamente l'operazione di convoluzione discreta in due dimensioni è formulabile come:

$$ C = f * I $$
$$ C[i,j] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} f[u,v] \cdot I[i-u,j-v] $$

dove $C$ è il risultato della convoluzione, $f$ è il filtro di dimensioni $k \times k$ e $I$ è l'immagine. $C[i,j]$ è il valore di un singolo pixel dell'immagine risultante dall'operazione. Nel caso di un'immagine a più canali, anche il filtro deve avere lo stesso numero di canali e il risultato della convoluzione sarà semplicemente la somma dei risultati di tutti i canali.
Nel processing di segnali questa operazione è in realtà chiamata correlazione incrociata. L'unica differenza tra le due operazioni è che nella convoluzione il filtro viene rovesciato prima di essere applicato all'immagine, ma visto che i pesi del filtro vengono aggiustati dalla rete durante l'addestramento, questo rovesciamento è superfluo. Di conseguenza nel campo delle ANN si usa il termine convoluzione intercambiandolo con correlazione incrociata.

Di base quindi abbiamo come parametri le dimensioni dell'immagine e quelle del filtro (entrambi rappresentati come tensori numerici tridimensionali).
Nel campo del deep learning poi vengono solitamente usati alcuni parametri addizionali, tra cui:

* _Pad_: Definisce se aggiungere dei pixel ai bordi dell'immagine prima di applicare la convoluzione. Solitamente vengono aggiunti pixel neri, dei pixel con valori riflessi rispetto al bordo o dei pixel con i valori circolari come se l'immagine fosse arrotolata su sè stessa. Ciò permette di gestire le dimensioni dell'immagine in output dalla convoluzione e le condizioni al contorno dell'operazione.
* _Stride_: Definisce se applicare il filtro su tutta l'immagine o se saltare alcuni pixel. Per esempio uno stride di 2 equivale a applicare il filtro prendendo solo 1 pixel dell'immagine ogni 2. Questo parametro permette di ridurre le dimensioni dell'immagine in output.
* Numero di filtri: Ogni filtro deve avere tanti canali quante le dimensioni delle immagini in partenza. Il numero di canali dell'output dipenderà dal numero di filtri che scegliamo di applicare.

I parametri della convoluzione vengono scelti prima dell'addestramento e di solito sono invariabili e caratteristici della struttura della rete. Quello che invece la rete impara e modifica durante l'addestramento sono i valori nei filtri, solitamente chiamati pesi: ciò permette di insegnare alla rete ad applicare trasformazioni anche molto complesse all'immagine in input, a seconda del numero di filtri e di pesi aggiustabili.

#### Implementazione dell'algoritmo

Applicare direttamente la formula della convoluzione passando il filtro su tutta l'immagine è un processo molto lento, e in genere i layer di convoluzione occupano la maggioranza del tempo di calcolo delle reti di elaborazione immagini. La complessità computazionale per la convoluzione diretta su un'immagine di dimensioni $H \times W \times C$ con un filtro di dimensioni $k \times k$ è un $O(HWCk^2)$. Per migliorare le performance ci sono vari algoritmi di convoluzione che cercano di ottimizzare diversi aspetti dell'operazione. Uno dei più utilizzati, che ho implementato in Byron, consiste in due fasi: Im2col (image to columns) e GEMM (GEneral Matrix Multiply).

L'Im2col è una trasformazione che "appiattisce" l'immagine originale trasformandola in una enorme matrice, dove ogni colonna contiene tutti gli elementi a cui deve essere applicato un singolo filtro in un singolo step. Il numero di colonne di questa matrice dipende quindi da quante volte il filtro deve scorrere sull'immagine per coprirla interamente. Questa fase consiste solo nella copia di dati dell'immagine in una matrice delle giuste dimensioni, dove alcuni elementi sono solitamente ripetuti perchè rientrano nelle finestre di applicazione del filtro più volte man mano che si sposta. Di conseguenza non richiede calcoli (escluse eventuali conversioni tra gli indici) ma solo copie di dati in memoria, ed ha una complessità $O(HWC)$. Tuttavia è molto onerosa dal punto di vista della memoria, in quanto l'immagine trasformata è sempre molto più grande dell'originale. Ciò non è un problema nell'ottica di implementazione di Byron, in quanto i server e i computer tipicamente usati in bioinformatica hanno grandi quantità di memoria disponibile.

Una volta eseguito l'Im2col sull'immagine, per ottenere il risultato della convoluzione è necessario moltiplicare tra di loro le matrici dell'immagine appiattita e del filtro. I filtri vengono anche essi appiatiti in modo da essere una matrice in cui ogni riga è la riorganizzazione unidimensionale di un filtro, e avremo quindi tante righe quanti i filtri da applicare. Questo passaggio costituisce il vero vantaggio di questo modo di eseguire la convoluzione, in quanto per molto tempo anche prima dell'avvento delle reti neurali l'operazione di moltiplicazione tra matrici è stata ottimizzata per essere il più efficiente possibile nell'utilizzo della memoria cache del processore. Per fare ciò è necessario seguire una serie di tecnicismi, tra cui per esempio preservare il Single Instruction Multiple Data stream (SIMD) del processore: ogni processore moderno ha infatti una memoria chiamata cache (solitamente divisa in vari livelli chiamati L1, L2 etc) molto piccola ma molto vicina al processore che permette di risparmiare tempo rispetto all'accesso alla RAM. Su questa memoria è possibile eseguire la stessa istruzione in parallelo su tutti gli elementi, il che permette un certo grado di vettorizzazione (che dipende dal processore) a livello di singolo core. Essendo la memoria cache molto piccola, saper gestire attentamente gli accessi e l'ordine dei calcoli che vengono eseguiti in essa permette di ottenere dei buoni speedup [@gemm] a parità di complessità computazionale dell'operazione di GEMM, che è solitamente un $O(N^3)$ per matrici $N \times N$. Inoltre il tempo di copia dei dati per l'Im2col rispetto al tempo di calcolo necessario per la GEMM diventa trascurabile per matrici grandi, implicando un'alta intensità aritmetica (cioè vengono svolte molte operazioni al secondo).

Ovviamente poi le operazioni necessarie per le moltiplicazioni tra matrici vengono divise tra i vari core, garantendo la parallelizzazione massima possibile e dividendo così il carico di lavoro in parti più piccole: ogni core si occuperà di calcolare il risultato per un singolo filtro. Per questo motivo solitamente il numero di filtri dei layer di convoluzione delle reti neurali è una potenza di 2, come anche solitamente il numero di core su un computer. A questo proposito è facile notare come le GPU siano così superiori alle CPU in termini di performance nel campo delle reti neurali: sebbene abbiano core molto meno potenti, sono immensamente più numerosi e generalmente parlando la struttura hardware delle GPU è molto più specifica e pensata appositamente per favorire il calcolo parallelo. Ne consegue che operazioni come la convoluzione risultano molto più veloci sulle GPU rispetto alle CPU, a parità di algoritmo [@gemm].

In figura \ref{im2col} è rappresentato uno schema dell'operazione di Im2col seguita dalla GEMM.

![Schema esplicativo dell'algoritmo Im2col con un filtro 2x2 su un'immagine a 3 canali, seguito dall'algoritmo GEMM tra la matrice dei filtri e la matrice dell'immagine. \label{im2col}](immagini/im2col.png){ width=100% }


### Blocchi residui e concatenazioni

Nelle strutture delle reti che presenterò sono presenti dei collegamenti a livelli precedenti della rete, solitamente chiamati blocchi residui. Ciò perchè gli output di alcuni layer vengono passati in avanti nella rete senza essere modificati e vengono poi sommati a layer successivi. Questi layer avranno quindi un residuo da un livello meno profondo della rete. Durante l'addestramento ci saranno quindi alcuni layer che, oltre a imparare a svolgere il compito necessario per ottenere l'output desiderato, si occupano di dare un contributo ai layer precedenti della rete a cui magari il gradiente della funzione di errore non arriva abbastanza grande da essere significativo nell'aggiustamento dei pesi. Infatti nel caso in cui il gradiente sia nullo, verrà semplicemente retro-propagata la funzione identità da questi layer [@residual].

Oltre ai blocchi residui un altro metodo popolare per propagare le feature da livelli meno profondi del network a livelli più profondi (cioè verso l'output) consiste nei layer di _route_, che semplicemente concatenano l'output di due o più layer creando immagini con un numero di canali pari alla somma dei canali delle immagini concatenate.

### Attivazioni ReLU

Le funzioni di attivazione ReLU (Rectified Linear Unit) hanno la seguente forma:

$$ f(x) = max(0,x) $$

e sono sempre più utilizzate nelle DNN in quanto permettono al network di ottenere rappresentazioni sparse delle feature. Sperimentalmente [@relu] ciò comporta vari benefici:

- _Information disentangling_: uno degli obiettivi delle DNN è riuscire a distinguere i vari fattori che causano le variazioni tra i dati in input e i dati in output. Una rappresentazione densa è altamente _entangled_ perchè quasi qualsiasi cambiamento nell'input modifica la maggior parte della rappresentazione dei dati nella rete. Viceversa una rappresentazione sparsa risente meno di cambiamenti nell'input garantendo la conservazione dell'insieme delle feature.  
- Rappresentazione di informazione in maniera efficiente: diversi input potrebbero contenere diverse quantità di informazione utile allo scopo della rete e sarebbero perciò più adeguatamente rappresentati con una struttura di dati di dimensione variabile. Variare il numero di neuroni attivi nella rete (i neuroni il cui output è 0 vengono considerati "spenti") consente alla rete di gestire le dimensioni della rappresentazione a seconda dell'informazione nell'input.
- Distribuite ma sparse: le rappresentazioni sparse sono esponenzialmente più efficienti delle rappresentazioni dense distribuite, che a loro volta sono esponenzialmente più efficienti delle rappresentazioni puramente locali [@relu], dove la potenza dell'esponente dipende dal numero di feature non nulle.
- Riduzione del problema di scomparsa del gradiente: nel caso in cui l'attivazione sia positiva, il valore della derivata non è limitato a un range finito come nel caso delle attivazioni logistiche o tangenti iperboliche. Nel caso in cui l'attivazione sia nulla, introdurre un grado di non-linearità nella rete non è un problema nella propagazione del gradiente finchè ci sono neuroni in cui il flusso di informazione riesce a retro-propagarsi.

Per questi motivi nei modelli analizzati molti layer hanno attivazioni ReLU mentre altri avranno semplici attivazioni lineari (ovvero il risultato dell'attivazione è identico all'output del layer).

Una versione leggermente diversa dell'attivazione ReLU, che in qualsiasi caso trasmette l'informazione anche se con attivazione molto ridotta, si chiama Leaky ReLU ed ha la seguente forma:

$$ f(x) = \left\{\begin{matrix}
x & $se $ \ x > 0 \\
0.01 \cdot x & $altrimenti$
\end{matrix}\right. $$


### Pixel-shuffle

Molte delle prime reti neurali per super-risoluzione pre-processavano l'immagine a bassa risoluzione di input con un upsample bicubico. In seguito l'immagine già delle dimensioni uguali a quella dell'output atteso veniva passata alla rete che cercava di migliorarne appunto la risoluzione. Questo rendeva l'addestramento più semplice per la rete ma molto più lento in quanto l'immagine di input aveva già dimensioni notevoli, aumentando di un fattore pari alla scala al quadrato i calcoli richiesti durante i forward dei vari layer. Per ovviare a questo problema è stato introdotto qualche anno fa un layer chiamato pixel-shuffle[@shuffle], anche noto come layer di convoluzione sub-pixel. Questo layer mescola appunto i canali di un'immagine a bassa risoluzione per generarne una con meno canali ma con dimensioni maggiori. Praticamente consiste nel riorganizzare le dimensioni del tensore dell'immagine, ma anche nel mescolare tra loro i vari pixel durante l'operazione. Matematicamente la funzione applicata è la seguente:

$$ PS(I [x,y,c]) = I [x//r , y//r , C\cdot r\cdot x\%r + C\cdot y\%r + c ] $$

e trasforma un'immagine $[H \times W \times C^{2}]$ in una immagine  $[rH \times rW \times C]$.
Nella formula il simbolo " // " rappresenta il quoziente della divisione intera mentre " % " rappresenta il resto.
Se l'immagine invece è in formato channel-first, ovvero ordinata come [C,H,W], la funzione di pixel-shuffle è leggermente diversa ma analoga. In Byron sono state implementate entrambe le versioni, in modo da garantire la massima versatilità possibile.
In figura \ref{pixshuff} riporto uno schema esplicativo dell'operazione di shuffle applicata in seguito ad una convoluzione.

Faccio notare che la funzione `PixelShuffle` della libreria PyTorch opera su immagini [C,H,W] mentre la funzione `depth_to_space` della libreria Tensorflow esegue il pixel-shuffle su immagini [H,W,C].

Il vantaggio principale nell'utilizzo di questo layer è l'incremento di velocità della rete [@shuffle]: infatti utilizzando questo layer alla fine (o comunque in uno dei layer finali) della rete, è possibile estrarre tutte le feature necessarie per la super-risoluzione (che in questo caso sono i vari filtri applicati) direttamente dall'immagine a bassa risoluzione di input, applicando vari layer di convoluzione, per poi riorganizzarle nell'immagine finale di dimensioni volute.

![Schema esplicativo dell'operazione di riorganizzazione tensoriale effettuata dal layer di pixel shuffle applicato dopo un layer di convoluzione. \label{pixshuff}](immagini/pixelshuffle.png){ width=100% }

### Batch normalization

Con l'aumento del numero di layer delle DNN permesso dalla sempre crescente potenza di calcolo dei computer odierni sono sorti vari problemi soprattutto durante l'addestramento, tra cui l'esplosione e la scomparsa del gradiente. Una delle possibili soluzioni a questi problemi consiste nel layer di batch normalization [@batchnorm]. Come già detto precedentemente, solitamente l'input di una rete viene normalizzato (cioè tutti gli input vengono riscalati per avere valori compresi in un range scelto, solitamente [0,1]) o standardizzato (cioè i valori degli input vengono divisi per la media del dataset e in seguito gli viene sottratta la deviazione standard, per avere dei dati distribuiti con media 0 e deviazione standard 1). Ciò aiuta l'addestramento in quanto riduce il range dinamico dei dati in input a un range fisso, permettendo alla rete di estrarre feature più robuste e più velocemente [@standardize]. Tuttavia se la rete ha un elevato numero di layer, a seconda dei valori dei pesi l'output dei layer potrebbe tornare ad avere range dinamici ampi. Per ovviare a questo problema, si interpone un layer di batch normalization dopo il layer della rete da normalizzare. In questo modo non solo l'input della rete ma anche l'output dei vari layer viene standardizzato. Ogni layer di batch normalization ha due pesi (per ogni batch), un fattore di scala e un bias, che modificano l'output standardizzato permettendo di cambiarne media e deviazione standard. Questi pesi possono venire aggiustati durante l'addestramento. Il termine batch nel nome deriva dal gruppo di dati su cui viene effettuata la normalizzazione nel layer, che in questo caso è appunto un batch utilizzato durante l'addestramento con Stochastic Gradient Descent (SGD) [@sgd].

### Layer YOLO

Il layer YOLO è il punto di forza della rete YOLOv3 e permette l'estrazione delle feature necessarie alla object detection in un unico passaggio dell'immagine attraverso la rete. L'immagine di input della rete viene divisa in una griglia $S \times S$. Se il centro di un oggetto ricade all'interno di una cella della griglia, questa sarà responsabile di rilevare quell'oggetto. Ogni cella prevede $B$ box e i rispettivi punteggi di confidenza, definiti come $Pr(object) \times IOU(prevision/ground truth)$. $Pr$ rappresenta la probabilità che il box contenga un oggetto, mentre $IOU$ (intersection over union) rappresenta la precisione del box rispetto all'oggetto effettivo da rilevare. Ogni box ha 5 previsioni $t_x$, $t_y$, $t_w$, $t_h$ e $t_o$, che poi permettono di calcolare:

$$ b_x = \sigma(t_x) + c_x $$
$$ b_y = \sigma(t_y) + c_y $$
$$ b_w = p_w e^{t_w} $$
$$ b_h = p_h e^{t_h} $$
$$ Confidenza = \sigma (t_o) $$

Le coordinate $b_x$ e $b_y$ rappresentano il centro del box rispettivamente ai bordi della cella (di posizione $c_x$, $c_y$), quindi assumono valori compresi tra 0 e 1. Anche la confidenza, essendo una misura di probabilità, è compresa in questo intervallo. Per garantire che durante l'addestramento questi valori rimangano effettivamente in questo range, viene applicata una funzione di attivazione logistica $\sigma$ a queste previsioni, con la seguente forma:

$$\sigma(t) = \frac{1}{1+e^{-t}}$$

Le dimensioni del box invece vengono determinate da $b_w$ e $b_h$ a partire da  $p_w$ e $p_h$, ovvero le dimensioni dei box a priori. Si sceglie di utilizzare dei box a priori di dimensione fissa invece di prevedere direttamente da zero le dimensioni dei box perchè è stato dimostrato in altre reti come la Faster R-CNN [@frcnn] che usare dei box a priori e determinare l'offset e la confidenza dei box per la detection a partire da essi migliora le performance e rende l'addestramento più stabile. Nella Faster R-CNN le dimensioni dei box fissi sono decise a priori e a mano. In YOLOv3 sono decise a priori ma sono state scelte dopo aver effettuato un k-mean clustering [@yolov2] sui dataset VOC e COCO per avere un buon rapporto efficienza / precisione. Si è optato per avere 9 box fissi diversi, da cui la rete parte per trovare i box finali.

Ogni cella prevede anche *C* probabilità condizionali di classe $Pr(Classe_{i} /oggetto)$. A test time queste probabilità vengono moltiplicate per i punteggi di confidenza per ottenere punteggi specifici di classe per ogni oggetto. L'insieme di tutte le previsioni, che costituisce l'output del layer, viene codificato in un tensore $S \times S \times (B \times 5 + C)$. Il 5 è dato dalle 4 coordinate e della confidenza.

## Modelli utilizzati

### EDSR

Il primo modello che ho analizzato e riprodotto in Byron si chiama Enhanced Deep Super Resolution (EDSR)[@edsr] ed è appunto una DNN per la super-risoluzione che si è classificata al primo posto nella NTIRE challenge del 2017 [@ntire2017].

La struttura di base dell'EDSR è la SRResNet, una modifica della ResNet [@resnet] (famosa rete neurale a blocchi residui nel campo dell'elaborazione immagini) pensata per la super-risoluzione, con ulteriori modifiche pensate per velocizzare l'addestramento e aumentare la qualità dell'immagine ottenuta. In particolare vengono rimossi i layer di batch normalization che risultano non solo poco efficaci per velocizzare l'addestramento ma anzi richiedono molto tempo in più per effettuare i calcoli di normalizzazione necessari. Infatti è stato dimostrato [@edsr] che per task di cosiddetta low-level vision come la super-risoluzione, dove non è necessario svolgere compiti difficili come l'object detection, mantenere un ampio range dinamico di output è benefico per i risultati e non ha ripercussioni sull'addestramento.

La struttura della rete EDSR è illustrata nell'immagine \ref{edsr}. Essa consiste in:

- Un layer di convoluzione che prende l'immagine LR come input, con 256 filtri.
- Un gruppo di 32 blocchi residui, ognuno a sua volta composto da:
  * Un layer di convoluzione con 256 filtri.
  * Un layer di attivazione ReLU.
  * Un altro layer di convoluzione con 256 filtri.
  * Una moltiplicazione del risultato ottenuto per il fattore di scala, in questo caso equivalente a 0.1 , prima di sommare l'output del blocco residuo al suo input e continuare l'elaborazione nella rete.
- Un layer di convoluzione con 256 filtri, a cui viene sommato l'output del primo layer di convoluzione della rete.
- Un blocco per l'upsample dell'immagine, che nel caso del fattore di scala (x4) utilizzato è composto da:
  * Un layer di convoluzione con 1024 filtri.
  * Un layer di pixel-shuffle con scala $r = 2$.
  * Un layer di convoluzione con 1024 filtri.
  * Un layer di pixel shuffle con scala $r = 2$.
- Un layer finale di convoluzione che ha come output l'immagine super-risoluta, con 3 filtri.

![Struttura della rete EDSR per super-risoluzione. I vari blocchi indicano la tipologia dei layer, tra cui convoluzione, blocchi residui, pixel-shuffle, e attivazioni ReLU. \label{edsr}](immagini/edsr.png){ width=80% }

I vari blocchi residui con i rispettivi layer di convoluzione hanno la funzione di trovare le feature ed il contenuto ad alta frequenza nell'immagine a bassa risoluzione di input, mentre il primo layer di convoluzione crea una versione con il contenuto a bassa frequenza dell'immagine, che poi viene sommata alla componente ad alta frequenza estratta dai blocchi residui. Infine l'immagine così elaborata attraversa ulteriori layer di convoluzione e pixel-shuffle per venire ridimensionata a dimensioni 4 volte superiori a quelle di partenza. Il blocco di upsample è l'unica parte della rete che cambia a seconda del fattore di scala scelto per l'upscale. Con un fattore 2 questo blocco è formato solamente da un layer di convoluzione con 1024 filtri e da un layer di pixel-shuffle con scala 2; con un fattore 3 è formato da un layer di convoluzione con 2304 filtri e da un layer di pixel-shuffle con scala 3; con un fattore di 4 come nel caso in analisi è formato da due parti che applicano entrambe un fattore di scala 2.  A causa del grande numero di filtri e delle dimensioni delle immagini in input nei layer di convoluzione del blocco di upsample, questi ultimi risultano di gran lunga i più lenti della rete e occupano buona parte del tempo di calcolo. Nella tabella \ref{tab_edsr} riporto il numero di parametri utilizzati nei layer del modello. In totale i pesi della rete sono oltre 43 milioni.


----------------------------------------------------------------------------------------------------------------
Layer                                  Canali input/ouput      Dim. filtri             Parametri
--------------------------------       -------------------     ---------------         -------------
Convoluzione input                     3 / 256                 3x3                     6912

Conv. (blocco residuo)                 256 / 256               3x3                     589824  

Convoluzione (pre-shuffle)             256 / 256               3x3                     589824

Conv. (blocco upsample)                256 / 1024              3x3                     2359296

Convoluzione output                    256 / 3                 3x3                     6912

----------------------------------------------------------------------------------------------------------------

Table: Parametri per ogni tipo di layer della rete EDSR, in funzione del numero di canali in input e output del layer e delle dimensioni del filtro di convoluzione utilizzato.  \label{tab_edsr}


### WDSR

Il secondo modello che ho analizzato e riprodotto in Byron si chiama Wide Deep Super Resolution (WDSR)[@wdsr] ed è un miglioramento della EDSR. Questa rete si è classificata al primo posto nella NTIRE challenge del 2018, nelle track con metodo di downsample sconosciuto, e ha performato molto bene anche nella track con metodo di downsample bicubico. Rispetto alla EDSR, la WDSR modifica principalmente due aspetti:

- **La struttura del network**: Come mostrato in figura \ref{wdsr}, la struttura della rete WDSR è leggermente più semplice della rete EDSR. Essa infatti non ha i layer di convoluzione dopo il pixel-shuffle e inoltre nel caso del fattore di scala x4 laddove il blocco di upsample della EDSR consiste in multipli layer di pixel-shuffle con $r = 2$ e layer di convoluzione, nella WDSR l'upsample è formato unicamente da un layer di pixel-shuffle con $r = 4$. Ciò permette un notevole risparmio di tempo in quanto i layer di convoluzione nel blocco di upsample della EDSR sono i più pesanti in termini di tempi di calcolo e parametri. Inoltre a differenza della EDSR dove c'è un blocco residuo tra il primo layer di convoluzione e il blocco di upsample per aggiungere il contenuto a bassa frequenza all'immagine di output, nella WDSR il contenuto a bassa frequenza viene processato in un ramo completamente separato della rete che viene sommato solamente alla fine al contenuto ad alta frequenza. Un esempio di ciò può essere visto nella figura \ref{wdsr_freq}.
- **I blocchi residui**: Aumentare la profondità e i parametri delle reti neurali generalmente migliora le performance a discapito dei tempi di calcolo. Per migliorare effettivamente le performance senza cambiare complessità computazionale (e quindi senza aggiungere parametri), la WDSR propone dei blocchi residui leggermente diversi basati sulla congettura [@mobilenet] che i layer di attivazione ReLU, sebbene garantiscano la non-linearità della rete e la stabilità durante l'addestramento, impediscono in parte il flusso di informazione dai layer meno profondi, che nel caso delle reti per super-risoluzione sono quelli con il contenuto a bassa frequenza da cui deve venire estrapolato quello ad alta frequenza. Per ovviare a questo problema senza aumentare il numero di parametri, nella WDSR c'è il cosiddetto "allargamento del passaggio". Esso consiste semplicemente nel ridurre il numero di canali in input ed aumentare il numero di canali in output (quest'ultimo dato dal numero di filtri) ai layer di convoluzione prima dei layer ReLU. Ciò consente di avere un maggior numero di canali su cui viene applicata l'attivazione, consentendo un migliore passaggio dell'informazione lungo la rete ma mantenendo la non-linearità necessaria. Per compensare a questo aumento di parametri pre-attivazione ovviamente i layer di convoluzione dopo i layer ReLU hanno un aumento dei canali in input e una riduzione dei canali in output, in modo da conservare il numero totale dei parametri (che in un layer di convoluzione è dato dal prodotto tra numero di canali in input e in output e le dimensioni del filtro).

![Output dei due rami della rete WDSR. A sinistra il contenuto ad alta frequenza, ottenuto dopo layer di convoluzione, blocchi residui e pixel-shuffle. A destra il contenuto a bassa frequenza, ottenuto semplicemente con un layer di convoluzione e un pixel-shuffle. \label{wdsr_freq}](immagini/wdsr_freq.png){ width=100% }

![Confronto delle strutture delle reti EDSR (a sinistra) e WDSR (a destra). Si noti che anche la struttura interna del residual body è differente tra le due strutture. \label{wdsr}](immagini/WDSR.png){ width=100% }

In questa tesi ho utilizzato, come precedente specificato, dei modelli già addestrati e quindi non mi è stato possibile modificare la struttura delle reti. Al momento sono reperibili soltanto versioni ridotte della rete WDSR, e quella utilizzata nel mio caso ha un numero di parametri notevolmente inferiore rispetto alla rete EDSR: in totale sono poco più di 3,5 milioni di pesi, meno di 1/10 della precedente rete utilizzata. Nella tabella \ref{tab_wdsr} ho riportato il numero di parametri per ogni layer del modello della rete WDSR utilizzato. Come si può inoltre notare dal numero di filtri, è stato utilizzato un fattore di allargamento del passaggio pari a 6. Le performance che ho riscontrato dalla rete ovviamente risentono molto di questo numero ridotto di parametri, ma è stato verificato [@wdsr] che in caso di parità di parametri la struttura della rete WDSR genera immagini di output di qualità nettamente superiore alla rete EDSR ed è anche più efficiente in termini di tempi di calcolo.

----------------------------------------------------------------------------------------------------------------
Layer                                  Canali input/ouput      Dim. filtri             Parametri
--------------                         -------------------     -------------------     -------------------
Convoluzione input 1                   3 / 32                  3x3                     864

Conv. 1 (blocco residuo)               32 / 192                3x3                     55296  

Conv. 2 (blocco residuo)               192 / 32                3x3                     55296  

Convoluzione (pre-shuffle)             32 / 48                 3x3                     13824

Conv. input 2 (pre-shuffle)            3 / 48                  5x5                     3600

----------------------------------------------------------------------------------------------------------------

Table: Parametri per ogni layer della rete WDSR, in funzione del numero di canali in input e output del layer e delle dimensioni del filtro di convoluzione utilizzato.  \label{tab_wdsr}

\newpage

### YOLO

Le prime reti neurali per object detection erano formate da un insieme di classificatori applicati a varie scale e posizioni dell'immagine, per riconoscere vari tipi di oggetti. Altri metodi popolari usavano prima un sistema di segmentazione, per trovare nell'immagine regioni con oggetti da classificare, e in seguito utilizzavano classificatori su queste regioni per determinare a quale classe appartenesse l'oggetto. Con la rete YOLO[@yolo] (You Only Look Once) invece l'object detection diventa un problema di regressione dai pixel dell'immagine direttamente alle coordinate dei box contenenti gli oggetti e alle probabilità delle rispettive classi. Ciò comporta due vantaggi principali: YOLO è molto veloce, tanto da poter essere eseguito in real-time anche su registrazioni live da videocamere [@yolov2], e inoltre commette molti meno errori di falsi positivi sul background perchè ha uno sguardo d'insieme sull'immagine invece di dividerla in zone. Il problema principale di YOLO è la localizzazione: sebbene sia un ottimo classificatore, le posizioni dei box attorno agli oggetti non sono sempre molto precise [@yolo]. La versione attuale di YOLO è la v3 [@yolov3]. La rete è strutturata nel seguente modo:

![Modello della rete neurale Darknet53, in cui vengono riportati tipo di layer, numero di filtri, dimensione del filtro e dimensione dell'output di tutti i layer componenti la rete. Il $/ 2$ nell'immagine vicino alla dimensione del filtro indica che è stato utilizzato uno _stride_ di 2. \label{darknet53}](immagini/darknet53.png){ width=50% }

* Una parte della rete si occupa dell'estrazione delle feature map, ed è in realtà una modifica della rete per detection (ma non classificazione) chiamata Darknet53, il cui modello è raffigurato nell'immagine \ref{darknet53}. Rispetto alla versione in figura, nel caso della rete YOLOv3 le dimensioni dell'output sono diverse (il primo layer ha dimensioni dell'immagine 608x608 e gli altri scalano di conseguenza) e inoltre non vengono usati i layer (Avgpool, Connected, Softmax) dopo l'ultimo gruppo di blocchi residui.
* A partire dall'output dell'ultimo blocco residuo viene aggiunto un blocco di detection, composto da:
  - Un layer di convoluzione con 512 filtri di dimensione 1x1.
  - Un layer di convoluzione con 1024 filtri di dimensione 3x3.
  - Altre due coppie di layer con stesse dimensioni e filtri dei precedenti.
  - Un layer di convoluzione con 255 filtri di dimensione 1x1. L'output di questo layer ha dimensioni 19x19 e rappresenta la feature map a scala fine della rete.
  - Un layer YOLO che calcola le previsioni a scala fine.
* A partire dal terzultimo layer di convoluzione precedente viene aggiunto un blocco di upsample e detection, composto da:
  - Un layer di convoluzione con 256 filtri di dimensione 1x1.
  - Un layer di upsample lineare che raddoppia le dimensioni dell'immagine senza interpolazione.
  - Un layer di _route_ che concatena l'output del layer precedente con quello dell'ultimo layer del penultimo gruppo di blocchi residui. Ciò permette di prendere la feature map di dimensioni 38x38 calcolata precedentemente nel network ed aggiungervi alcuni dei dettagli estrapolati dalla scala fine.
  - Un layer di convoluzione con 256 filtri di dimensione 1x1.
  - Un layer di convoluzione con 512 filtri di dimensione 3x3.
  - Altre due coppie di layer con stesse dimensioni e filtri dei precedenti.
  - Un layer di convoluzione con 255 filtri di dimensione 1x1. L'output di questo layer ha dimensioni 38x38 e rappresenta la feature map a scala media della rete.
  - Un layer YOLO che calcola le previsioni a scala media.
* Come nel caso precedente a partire dal terzultimo layer di convoluzione precedente viene aggiunto un ulteriore blocco di upsample e detection, che però ha i layer di convoluzione con la metà dei filtri (quindi 128 e 256 rispettivamente), tranne l'ultimo in quanto il numero di filtri del layer di convoluzione prima del layer YOLO dipende dalla forma del tensore delle previsioni. Questo blocco concatena la feature map di dimensioni 76x76 calcolata nel terzultimo gruppo di blocchi residui a quella precedente a scala media dopo che questa è stata raddoppiata in dimensioni, in maniera analoga al passaggio da scala fine a scala media. In questo modo avremo anche un layer YOLO che effettua object detection su una feature map di scala grande che tiene anche conto delle precedenti feature map elaborate.

Ogni layer YOLO prevede 3 box, quindi i 9 box fissi a priori vengono distribuiti sulle 3 scale.
Tutti i layer di convoluzione del modello hanno anche un layer di attivazione Leaky ReLU e un layer di batch normalization dopo di loro, tranne i layer subito prima dei layer YOLO, che hanno un'attivazione lineare e non hanno batch normalization.
In tabella \ref{tab_yolo} è riportato il numero di parametri per ogni layer della rete. In totale sono quasi 62 milioni.

Il network viene addestrato su immagini di dimensione variabile da 320x320 a 608x608 (andando per multipli di 32, che è il fattore di downscale della rete). Questo rende i filtri più sensibili alle informazioni dettagliate date dalla risoluzione maggiore (rispetto ai 224x224 di YOLOv1) e inoltre rende il network più robusto perchè i filtri si adattano a trovare oggetti a varie scale. Per la detection il network sarà facilmente riscalabile scegliendo le dimensioni di input offrendo un tradeoff tra precisione e velocità. Durante i test in questa tesi ho sempre utilizzato YOLOv3 con dimensioni dell'immagine di input 608x608 per avere i risultati di qualità migliore possibile.
La funzione di errore usata dal network durante l'addestramento è molto complessa [@yolo], ma è per minimizzare la somma dei quadrati residui, con alcune modifiche:

* l'errore viene pesato 1/10 per i box in cui non ci sono oggetti rilevati, per dare più importanza all'errore sulle coordinate;
* vengono previste le versioni logaritmiche delle dimensioni del box invece di prevedere direttamente altezza e larghezza (come visto nel layer YOLO) per dare più importanza agli errori sui box piccoli rispetto a quelli sui box grandi;
* durante l'addestramento solo il box con l'IOU migliore verrà assegnato ad ogni oggetto, in questo modo ci sarà più specializzazione nelle previsioni migliorando l'accuratezza.

----------------------------------------------------------------------------------------------------------------
Layer                                  Canali input/ouput      Dim. filtri            Parametri
--------------                         -------------------     -----------------      -------------------
Convoluzione input                     3 / 32                  3x3                    864

Convoluzione 1                         32 / 64                 3x3                    18432  

Conv. 1 (blocco residuo 1)             64 / 32                 1x1                    2048  

Conv. 2 (blocco residuo 1)             32 / 64                 3x3                    18432  

Convoluzione 2                         64 / 128                3x3                    73728

Conv. 1 (blocco residuo 2)             128 / 64                1x1                    8192  

Conv. 2 (blocco residuo 2)             64 / 128                3x3                    73728  

Convoluzione 3                         128 / 256               3x3                    294912

Conv. 1 (blocco residuo 3)             256 / 128               1x1                    32768  

Conv. 2 (blocco residuo 3)             128 / 256               3x3                    294912  

Convoluzione 4                         256 / 512               3x3                    1179648

Conv. 1 (blocco residuo 4)             512 / 256               1x1                    131072  

Conv. 2 (blocco residuo 4)             256 / 512               3x3                    1179648  

Convoluzione 5                         512 / 1024              3x3                    4718592

Conv. 1 (blocco residuo 5)             1024 / 512              1x1                    524288  

Conv. 2 (blocco residuo 5)             512 / 1024              3x3                    4718592  

Conv. 1 (scala fine)                   1024 / 512              1x1                    524288  

Conv. 2 (scala fine)                   512 / 1024              3x3                    4718592  

Conv. 3 (scala fine)                   1024 / 512              1x1                    524288  

Conv. 4 (scala fine)                   512 / 1024              3x3                    4718592  

Conv. 5 (scala fine)                   1024 / 512              1x1                    524288  

Conv. 6 (scala fine)                   512 / 1024              3x3                    4718592  

Conv. pre-yolo (scala fine)            1024 / 255              1x1                    261120    

Conv. pre-upsample                     512 / 256               1x1                    131072           

(scala media)          

Conv. 1 (scala media)                  768 / 256               1x1                    196608  

Conv. 2 (scala media)                  256 / 512               3x3                    1179648  

Conv. 3 (scala media)                  512 / 256               1x1                    131072  

Conv. 4 (scala media)                  256 / 512               3x3                    1179648  

Conv. 5 (scala media)                  512 / 256               1x1                    131072  

Conv. 6 (scala media)                  256 / 512               3x3                    1179648  

Conv. pre-yolo (scala media)           512 / 255               1x1                    130560    

Conv. pre-upsample                     256 / 128               1x1                    32768    

(scala grande)

Conv. 1 (scala grande)                 384 / 128               1x1                    49152  

Conv. 2 (scala grande)                 128 / 256               3x3                    294912  

Conv. 3 (scala grande)                 256 / 128               1x1                    32768  

Conv. 4 (scala grande)                 128 / 256               3x3                    294912  

Conv. 5 (scala grande)                 256 / 128               1x1                    32768  

Conv. 6 (scala grande)                 128 / 256               3x3                    294912  

Conv. pre-yolo                         256 / 255               1x1                    65280

(scala grande)

----------------------------------------------------------------------------------------------------------------

Table: Parametri per ogni layer della rete YOLOv3, in funzione del numero di canali in input e output del layer e delle dimensioni del filtro di convoluzione utilizzato.  \label{tab_yolo}
