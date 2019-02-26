# Super-risoluzione

## Le origini

 La super-risoluzione è una tecnica in generale utilizzata per migliorare la risoluzione spaziale di un'immagine. Può essere applicata nella sua forma base solo a immagini che contengono aliasing, ovvero che sono state sotto-campionate o ridotte in dimensione. In queste immagini il contenuto ad alta frequenza dell'immagine ad alta risoluzione desiderata è nascosto nel contenuto a bassa frequenza dell'immagine di input. Di conseguenza applicare algoritmi per ripristinare questo contenuto (eventualmente dopo aver ingrandito l'immagine alle giuste dimensioni) consente di riprodurre immagini abbastanza simili all'output desiderato.

 I primi metodi di super-risoluzione di immagini digitali consistevano nella stima del contenuto ad alta frequenza tramite associazione di patch di immagine a bassa risoluzione con la corrispetiva immagine ad alta risoluzione. Queste patch venivano solitamente prese dopo aver filtrato l'immagine con un filtro di edge detection o dalla trasformata di Fourier dell'immagine per avere direttamente il contenuto in frequenza. Una volta imparato un "dizionario" di queste associazioni, da un insieme di coppie di immagini alta e bassa risoluzione note, era possibile applicarlo su un'immagine a bassa risoluzione e ottenerne una versione ad alta risoluzione. Si noti che in questo caso non necessariamente le immagini di input e output avevano dimensioni diverse: la risoluzione spaziale infatti dipende non solo dalle dimensioni dell'immagine ma anche dal passo di campionamento.

 Nel 2014, grazie al lavoro del Department of Information Engineering dell'università di Hong Kong, è nata l'idea di utilizzare i layer di convoluzione delle ormai sempre più popolari reti neurali per imparare in maniera automatizzata un analogo del dizionario delle patch, che in questo caso diventava un insieme di moltissime feature. Nacque così la prima ANN per super-risoluzione, chiamata SRCNN (Super Resolution Convolutional Neural Network), che consisteva semplicemente in tre layer di convoluzione. Il primo estraeva le patch a bassa risoluzione dall'immagine, il secondo collegava queste patch organizzate in un vettore a molte dimensioni a un altro vettore che idealmente rappresentava le patch ad alta risoluzione. Infine l'ultimo layer riorganizzava le patch ad alta risoluzione e le combinava per ottenere le immagini ad alta risoluzione di output.

 Da allora sono stati fatti molti progressi nella ricerca in questo campo, ma le idee fondamentali non sono cambiate. Semplicemente i modelli sono molto più profondi e utilizzano quindi contromisure per riuscire a gestire l'addestramento con un numero enorme di parametri imparabili.

## Preprocessing

Ogni rete a super-risoluzione deve ovviamente essere addestrata ed ha quindi bisogno di un dataset di immagini di input e di output attesi. Nelle reti di cui mi sono occupato in questo lavoro di tesi, l'immagine di input consiste di una versione ridimensionata dell'immagine di output. Questo introduce un fattore di aliasing che la rete dovrà quindi imparare a nullificare quando ingrandisce l'immagine, ripristinando i contenuti ad alta frequenza e quindi la risoluzione dell'output ottenuto. Queste reti sono molto grandi e hanno milioni di parametri: di conseguenza se l'input fosse modificato con dei filtri per simulare i problemi realistici delle immagini reali come rumore e sfocature, la rete imparerebbe in parte anche a risolvere questi problemi. Anche se i modelli di queste reti sono pensati per uno scopo diverso, ciò non evita che le feature estratte dai numerosi layer di convoluzione presenti risentano dei filtri applicati all'input e di conseguenza durante l'addestramento i pesi verranno aggiustati diversamente.

Per le analisi svolte in questa tesi ho considerato per semplicità (e anche per la facilità nel reperire i pesi dei modelli già precedentemente addestrati) solamente un fattore di scala di 4 e un metodo di riduzione dell'immagine bicubico. Inoltre seguendo la procedura standard viene rimossa la media (RGB) del dataset utilizzato durante l'addestramento prima di processare un'immagine con la rete, e dopo aver ottenuto l'immagine di output viene sommata nuovamente. Altre accortezze come riscalare tutti i valori dell'immagine per normalizzarli, che dipendono dal modello della rete e dai pesi usati, sono state tenute in considerazione per avere risultati ottimali.

### Ricampionamento bicubico

Il ricampionamento bicubico, che solitamente viene diviso in upsampling e downsampling o analogamente chiamati upscaling e downscaling, consiste in un metodo di interpolazione per determinare i pixel di un'immagine dopo che questa è stata cambiata di dimensioni (rispettivamente aumentata o diminuita). Il nome deriva dalla complessità massima dell'algoritmo di interpolazione usato, in cui l'operazione più complicata eseguita in questo caso è appunto il cubo del valore di un pixel. L'interpolazione viene eseguita in un intorno di 4 pixel. In generale i filtri scelti per il ricampionamento bicubico appartengono a una famiglia con la seguente forma:

$$ k(x) = \frac{1}{6}\left\{\begin{matrix}
(12-9B-6C)|x|^{3} + (-18+12B+6C)|x|^{2} + & \\
+ (6-2B) & se \ |x| < 1 \\
 (-B-6C)|x|^{3} + (6B+30C)|x|^{2} + & \\
(-12B-48C)|x| + (8B+24C) & se \ 1 \leq |x| <2 \\
0 & altrimenti
\end{matrix}\right.$$

Tra i più popolari cito le opzioni B=0, C=0.75 usata da OpenCV e Photoshop o B=0, C=0.5 solitamente chiamato filtro di Catmull-Rom usato da GIMP e Matlab.
Byron utilizza di default l'opzione (0, 0.75) ma può essere agevolmente cambiata secondo necessità. Inoltre in Byron ho anche implementato il filtro di Lanczos, che garantisce risultati migliori (soprattutto quando si parla di upsampling dove si vede effettivamente il miglioramento di qualità rispetto a un upsample lineare) e ha un intorno di 8 pixel.

Per applicare questi filtri bisogna ridimensionarli in modo che siano larghi quanto 4 pixel dell'immagine più piccola (quindi quella di partenza nel caso dell'upsampling e quella ridimensionata in caso di downsampling). Poi per ogni pixel dell'immagine obiettivo, bisogna calcolare tutti i contributi dei pixel che rientrano nel range del filtro per il pixel obiettivo e pesarli per il rispettivo valore del filtro a quella distanza dall'origine. Bisogna solitamente anche normalizzare i pesi del filtro in modo da non avere un aumento o una diminiuzione della intensità complessiva dell'immagine.

## EDSR

Il primo modello che ho analizzato e riprodotto in Byron si chiama Enhanced Deep Super Resolution (EDSR) ed è appunto un DNN per la super-risoluzione. Questa rete si è classificata al primo posto nella NTIRE (New Trends in Image Restoration and Enhancement) challenge del 2017, in cui vari team di ricerca proponevano modelli di network con l'obiettivo di migliorare la risoluzione di un'immagine ricampionata bicubicamente. La squadra che riusciva ad avere il PSNR (Peak Signal to Noise Ratio) medio sulle immagini di validazione del dataset DIV2K più alto si aggiudicava il primo posto. Delle misure della qualità delle immagini e delle performance parlerò nel prossimo capitolo.

La struttura di base dell'EDSR è la SRResNet, una modifica della ResNet (famosa rete neurale a blocchi residui nel campo dell'elaborazione immagini) pensata per la super-risoluzione, con ulteriori modifiche pensate per velocizzare l'addestramento e aumentare la qualità dell'immagine ottenuta. In particolare vengono rimossi i layer di batch normalization che risultano non solo poco efficaci per velocizzare l'addestramento ma anzi richiedono molto tempo in più per effettuare i calcoli di normalizzazione necessari. Infatti è stato dimostrato (ref) che per task di cosiddetta low-level vision come la super-risoluzione, dove non è necessario svolgere compiti difficili come l'object detection, mantenere una dinamicità del range di output è benefico per i risultati e non ha ripercussioni sull'addestramento.

La struttura della rete EDSR è illustrata nell'immagine \ref{edsr}. Essa consiste in:

- Un layer di convoluzione che prende l'immagine downsampled come input, con 256 filtri
- Un gruppo di 32 blocchi residui, ognuno a sua volta composto da:
  * Un layer di convoluzione con 256 filtri
  * Un layer di attivazione ReLU
  * Un altro layer di convoluzione con 256 filtri
  * Una moltiplicazione del risultato ottenuto per il fattore di scala, in questo caso equivalente a 0.1 , prima di sommare l'output del blocco residuo al suo input e continuare l'elaborazione nella rete
- Un layer di convoluzione con 256 filtri, a cui viene sommato l'output del primo layer di convoluzione della rete
- Un blocco per l'upsample dell'immagine, che nel caso del fattore di scala (x4) utilizzato è composto da:
  * Un layer di convoluzione con 1024 filtri
  * Un layer di pixel-shuffle con scala r = 2
  * Un layer di convoluzione con 1024 filtri
  * Un layer di pixel shuffle con scala r = 2
- Un layer finale di convoluzione che ha come output l'immagine super-risoluta, con 3 filtri

![Struttura dei layer che compongono la rete EDSR. \label{edsr}](immagini/edsr.png){ width=70% }

I vari blocchi residui con i rispettivi layer di convoluzione hanno la funzione di trovare le feature ed il contenuto ad alta frequenza nell'immagine a bassa risoluzione di input, mentre il primo layer di convoluzione crea una versione con il contenuto a bassa frequenza dell'immagine, che poi viene sommata alla compoente ad alta frequenza estratta dai blocchi residui. Infine l'immagine così elaborata attraversa ulteriori layer di convoluzione e pixel-shuffle per venire ridimensionata a dimensioni 4 volte superiori a quelle di partenza. A causa del grande numero di filtri e delle dimensioni delle immagini in input nei layer di convoluzione del blocco di upsample, questi ultimi risultano di gran lunga i più lenti della rete e occupano buona parte del tempo di calcolo. Nella tabella \ref{tab_edsr} riporto il numero di parametri utilizzati nei layer del modello. In totale sono oltre 43 milioni di pesi imparabili dalla rete.


----------------------------------------------------------------------------------------------------------------
Layer                                  Canali input/ouput      Dimensione filtri       Parametri
--------------                         -------------------     -------------------     -------------------
Convoluzione input                     3 / 256                 3x3                     6912

Convoluzione 1 e 2 (blocco residuo)    256 / 256               3x3                     589824  

Convoluzione (pre shuffle)             256 / 256               3x3                     589824

Convoluzione 1 e 2 (blocco upsample)   256 / 1024              3x3                     2359296

Convoluzione output                    256 / 3                 3x3                     6912

----------------------------------------------------------------------------------------------------------------

Table: Parametri per ogni layer della rete EDSR.  \label{tab_edsr}

\newpage

## WDSR

Il secondo modello che ho analizzato e riprodotto in Byron si chiama Wide Deep Super Resolution (WDSR) ed è un miglioramento della EDSR. Questa rete si è classificata al primo posto nella NTIRE challenge del 2018, nelle track con metodo di downsample sconosciuto, e ha performato molto bene anche nella track con metodo di downsample bicubico. Rispetto alla EDSR, la WDSR modifica principalmente due aspetti:

- **La struttura del network**: Come mostrato in figura \ref{wdsr}, la struttura della rete WDSR è leggermente più semplice della rete EDSR. Essa infatti non ha i layer di convoluzione dopo il pixel-shuffle e inoltre nel caso del fattore di scala x4 laddove il blocco di upsample della EDSR consiste in multipli layer di pixel-shuffle con r = 2 e layer di convoluzione, nella WDSR l'upsample è formato unicamente da un layer di pixel-shuffle con r = 4. Ciò permette un notevole risparmio di tempo in quanto i layer di convoluzione nel blocco di upsample della EDSR sono i più pesanti in termini di tempi di calcolo e parametri. Inoltre a differenza della EDSR dove una scorciatoia dopo il primo layer di convoluzione viene aggiunta prima del blocco di upsample per aggiungere il contenuto a bassa frequenza all'immagine di output, nella WDSR il contenuto a bassa frequenza viene processato in un ramo completamente separato della rete che viene sommato solamente alla fine al contenuto ad alta frequenza. Un esempio di ciò può essere visto nella figura \ref{wdsr_freq}.
- **I blocchi residui**: Aumentare la profondità e i parametri delle reti neurali generalmente migliora le performance a discapito dei tempi di calcolo. Per migliorare effettivamente le performance senza cambiare complessità computazionale (e quindi senza aggiungere parametri), la WDSR propone dei blocchi residui leggermente diversi basati sulla congettura (ref) che i layer di attivazione ReLU, sebbene garantiscano la non-linearità della rete e la stabilità durante l'addestramento, impediscono in parte il flusso di informazione dai layer meno profondi, che nel caso delle reti per super-risoluzione sono quelli con il contenuto a bassa frequenza da cui deve venire estrapolato quello ad alta frequenza. Per ovviare a questo problema senza aumentare il numero di parametri, nella WDSR c'è il cosiddetto "allargamento del passaggio". Esso consiste semplicemente nel ridurre il numero di canali in input ed aumentare il numero di canali in output (quest'ultimo dato dal numero di filtri) ai layer di convoluzione prima dei layer ReLU. Ciò consente di avere un maggior numero di canali su cui viene applicata l'attivazione, consentendo un migliore passaggio dell'informazione lungo la rete ma mantenendo la non-linearità necessaria. Per compensare a questo aumento di parametri pre-attivazione ovviamente i layer di convoluzione dopo i layer ReLU hanno un aumento dei canali in input e una riduzione dei canali in output, in modo da conservare il numero totale dei parametri (che in un layer di convoluzione è dato dal prodotto tra numero di canali in input e in output e le dimensioni del filtro).

In questa tesi ho utilizzato, come precedente specificato, dei modelli già addestrati e quindi non mi è stato possibile modificare la struttura delle reti. Al momento sono reperibili soltanto versioni ridotte della rete WDSR, e quella utilizzata nel mio caso ha un numero di parametri notevolmente inferiore rispetto alla rete EDSR: in totale sono poco più di 3,5 milioni di parametri, meno di 1/10 della precedente rete utilizzata. Nella tabella \ref{tab_wdsr} ho riportato il numero di parametri per ogni layer del modello della rete WDSR utilizzato. Come si può inoltre notare dal numero di filtri, è stato utilizzato un fattore di allargamento del passaggio pari a 6. Le performance che ho riscontrato dalla rete ovviamente risentono molto di questo numero ridotto di parametri, ma è stato dimostrato (ref) che in caso di parità di parametri la struttura della rete WDSR genera immagini di output di qualità nettamente superiore alla rete EDSR ed è anche più efficiente in termini di tempi di calcolo.

![Output dei due rami della rete WDSR. A sinistra il contenuto a bassa frequenza, ottenuto semplicemente con un layer di convoluzione e un pixel-shuffle. A destra il contenuto ad alta frequenza, ottenuto dopo layer di convoluzione, blocchi residui e pixel-shuffle. In basso l'immagine finale ottenuta dalla somma dei due contributi. \label{wdsr_freq}](immagini/WDSR.png){ width=100% }

![Confronto delle strutture delle reti EDSR e WDSR. \label{wdsr}](immagini/WDSR.png){ width=100% }

----------------------------------------------------------------------------------------------------------------
Layer                                  Canali input/ouput      Dimensione filtri       Parametri
--------------                         -------------------     -------------------     -------------------
Convoluzione 1 input                   3 / 32                  3x3                     864

Convoluzione 1 (blocco residuo)        32 / 192                3x3                     55296  

Convoluzione 2 (blocco residuo)        192 / 32                3x3                     55296  

Convoluzione (pre shuffle)             32 / 48                 3x3                     13824

Convoluzione 2 input (pre shuffle)     3 / 48                  5x5                     3600

----------------------------------------------------------------------------------------------------------------

Table: Parametri per ogni layer della rete WDSR.  \label{tab_wdsr}
