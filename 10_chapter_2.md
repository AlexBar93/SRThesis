# Layer fondamentali

I layer che compongono le reti neurali possono essere di varia natura, a seconda delle funzioni che devono svolgere. Nella libreria Byron ho implementato tutti quelli più comunemente utilizzati al momento, ma per lo scopo di questa tesi mi concentrerò solo su quelli necessari alla super-risoluzione.

## Immagini

Nel mondo del processing di immagini digitali, queste vengono solitamente rappresentate come tensori tridimensionali dove le tre dimensioni rappresentano l'altezza, la larghezza e il numero di canali dell'immagine. Ogni elemento del tensore ha un valore numerico che a seconda del formato dell'immagine può essere compreso nell'intervallo [0,255] o nell'intervallo [0,1]. Altri formati meno utilizzati hanno range diversi, ma la scelta di come rappresentare l'immagine digitale è solitamente lasciata all'utente. Byron si appoggia alla libreria OpenCV per caricare e salvare in memoria velocemente le immagini, ma ha un oggetto proprio per la loro elaborazione una volta che sono state importate, dotato di tutte le funzioni più comuni di elaborazione immagini.

## Convoluzione

La convoluzione consiste nell'applicazione di un filtro di dimensioni ridotte su tutta l'immagine. Ciò avviene scorrendo il filtro lungo tutta l'immagine, ed effettuando l'operazione di convoluzione ad ogni passo. Nel processing di segnali questa operazione è in realtà chiamata correlazione incrociata. L'unica differenza tra le due operazioni è che nella seconda il filtro viene rovesciato prima di essere applicato all'immagine, ma visto che i pesi del filtro vengono imparati dalla rete durante l'addestramento, questo rovesciamento è superfluo. Di conseguenza nel campo delle ANN si usa il termine convoluzione intercambiandolo con correlazione incrociata.
Matematicamente l'operazione di convoluzione è rappresentabile come:
$$ G = h * F $$
$$ G[i,j] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} h[u,v] \cdot F[i-u,j-v] $$
Di base quindi abbiamo come parametri le dimensioni dell'immagine e quelle del filtro (entrambi rappresentati come tensori numerici tridimensionali).
Nel campo del deep learning poi vengono solitamente usati alcuni parametri addizionali, tra cui:

* Padding: Definisce se aggiungere dei pixel ai bordi dell'immagine prima di applicare la convoluzione. Solitamente vengono aggiunti pixel neri o riflessi rispetto al bordo. Ciò permette di gestire le dimensioni dell'immagine in output dalla convoluzione.
* Striding: Definisce se applicare il filtro su tutta l'immagine o se saltare alcuni pixel. Per esempio uno stride di 2 equivale a applicare il filtro prendendo solo 1 pixel dell'immagine ogni 2. Questo parametro permette di ridurre le dimensioni dell'immagine in output.
* Numero di filtri: Ogni filtro deve avere tanti canali quante le dimensioni delle immagini in partenza. In questo modo la convoluzione verrà effettuata su ogni canale, e il numero di canali dell'output dipenderà dal numero di filtri che scegliamo di applicare.

I parametri della convoluzione vengono scelti prima dell'addestramento e di solito sono invariabili e caratteristici della struttura della rete. Quello che invece la rete impara e modifica durante l'addestramento sono i pesi dei filtri: ciò permette di insegnare alla rete ad applicare trasformazioni anche molto complesse, a seconda del numero di filtri, alla nostra immagine.

### Ottimizzazione

Applicare direttamente la formula della convoluzione passando il filtro su tutta l'immagine è un processo molto lento. Per migliorare le performance ci sono vari algoritmi di convoluzione che sfruttano caratteristiche tipiche del calcolo parallelo e della vettorizzazione. Uno dei più utilizzati, che ho implementato in Byron, consiste in due fasi:

#### Im2col

Questa trasformazione "appiattisce" l'immagine originale trasformandola in una enorme matrice, dove ogni colonna contiene tutti gli elementi a cui deve essere applicato un singolo filtro in un singolo step. Il numero di colonne di questa matrice dipende quindi da quante volte il filtro deve scorrere sull'immagine per coprirla interamente. Questa fase consiste solo nella copia di dati dell'immagine in una matrice delle giuste dimensioni, dove alcuni elementi sono solitamente ripetuti perchè rientrano nelle finestre di applicazione del filtro più volte man mano che si sposta. Di conseguenza non richiede calcoli (escluse eventuali conversioni tra gli indici) ed è molto veloce. Tuttavia è molto onerosa dal punto di vista della memoria, in quanto l'immagine trasformata è sempre molto più grande dell'originale. Ciò non è un problema nell'ottica di implementazione di Byron, in quanto i server e i computer tipicamente usati in bioinformatica hanno grandi quantità di memoria disponibile (per esempio quello utilizzato per i miei test dispone di 126 GB di ram). In figura \ref{im2col} è rappresentato uno schema dell'operazione di im2col (image to columns) seguita dalla gemm (general matrix multiply).

![Schema esplicativo dell'im2col con un filtro 2x2 su un'immagine a 3 canali. \label{im2col}](immagini/im2col.png){ width=100% }

#### GEMM

Una volta eseguito l'im2col sull'immagine, per ottenere il risultato della convoluzione è necessario moltiplicare tra di loro le matrici dell'immagine appiattita e del filtro. I filtri vengono anche essi appiatiti in modo da essere delle matrici in cui ogni riga rappresenta un filtro, e avremo quindi tante righe quanti i filtri da applicare. Questo passaggio rappresenta il vero vantaggio di questo modo di eseguire la convoluzione, in quanto per molti anni anche prima dell'avvento delle reti neurali l'operazione di moltiplicazione tra matrici è stata ottimizzata per essere il più veloce possibile. Per fare ciò è necessario seguire una serie di tecnicismi, tra cui per esempio preservare il Single Input Multiple Data stream (SIMD) del processore. Ogni processore moderno ha infatti una memoria chiamata cache (solitamente divisa in vari livelli chiamati L1, L2 etc) molto piccola ma molto vicina al processore che permette di risparmiare tempo rispetto all'accesso alla ram. Su questa memoria è possibile eseguire la stessa istruzione in parallelo su tutti gli elementi, il che permette un certo grado di parallelizzazione (che dipende dal processore) a livello di singolo core.

Ovviamente poi le operazioni necessarie per le moltiplicazioni tra matrici vengono divise tra i vari core, garantendo la parallelizzazione massima possibile e dividendo così il carico di lavoro in parti molto più piccole. A questo proposito è facile notare come le GPU siano così superiori alle CPU in termini di performance nel campo delle reti neurali: sebbene abbiano core molto meno potenti, sono immensamente più numerosi e generalmente parlando la struttura hardware delle GPU è molto più specifica e pensata appositamente per favorire il calcolo parallelo. Ne consegue che operazioni come la convoluzione risultano molto più veloci sulle GPU rispetto alle CPU.

### Problemi di concurrency

Per addestrare (in modo supervisionato) una rete neurale solitamente viene usata una procedura nota come backward propagation. Ciò permette di partire da una funzione di errore scelta a priori che valuti quanto sono differenti l'output atteso e quello ottenuto, e propagare all'indietro nella rete questo errore, correggendo i pesi di un layer alla volta. Per fare ciò ovviamente i layer hanno bisogno di funzioni che dicono alla rete come aggiustare i pesi in base all'errore. Solitamente queste funzioni, chiamate appunto backward, sono simili alle rispettive forward utilizzate quando usiamo una rete per inferenza, ma più complesse o problematiche. Un esempio di ciò è la funzione inversa dell'im2col, chiamata banalmente col2im. Questa funzione ovviamente deve trasformare la matrice dell'immagine appiatita e ricostruire l'immagine originale. Tuttavia come ho fatto notare precedentemente, ogni pixel dell'immagine originale viene mappato in più di una posizione nella matrice! Ne consegue che quando dobbiamo tornare indietro, vari numeri dalla matrice vanno a contribuire a un singolo pixel dell'immagine. Fin qui nessun problema: infatti basta semplicemente sommare i vari contributi per determinare il valore finale del pixel. I problemi sorgono, tuttavia, quando usiamo più di un core (o in generale più di un thread) per eseguire questa operazione: se due core cercano di scrivere nella stessa locazione di memoria insieme, non possiamo essere sicuri che il risultato sia quello che ci aspettiamo. Questo problema è chiamato concurrency. In Darknet la funzione di col2im per le CPU non risolve questo problema: ne consegue che durante l'addestramento su CPU possono nascere degli errori numerici che, sebbene piccoli, spesso si propagano in tutta l'immagine e in tutta la rete al susseguirsi dei cicli di addestramento. Questo problema ovviamente non si presenta solamente per la funzione di col2im ma è ricorrente in varie funzioni di backward di Darknet, che tuttavia non starò a specificare in quanto non saranno usate nelle reti di interesse.

In Byron questo problema è stato risolto garantendosi che l'accesso ad ogni locazione di memoria in scrittura fosse sempre sequenziale. Di conseguenza non capita mai che due o più core vadano a scrivere nella stessa locazione contemporaneamente.

## Problema della scomparsa del gradiente

L'addestramento delle reti neurali esula dallo scopo di questa trattazione e quindi non approfondirò l'argomento. Tuttavia per capire il motivo per cui i layer di cui sto per parlare servano effettivamente alla rete, è importante parlare dell problema della scomparsa del gradiente. Questo fenomeno si presenta durante l'addestramento di reti neurali con molti layer in cui l'errore viene propagato seguendo la regola della discesa del gradiente. In tale metodo, ogni parametro del modello riceve ad ogni iterazione un aggiornamento proporzionale alla derivata parziale della funzione di errore sull'output rispetto al parametro stesso. Solitamente durante il forward, dopo aver calcolato quella che è l'effettiva funzione del layer, viene applicata al risultato una funzione di attivazione (che rappresenta in biologia il firing dei neuroni). Le funzioni comunemente usate nelle ANN sono la tangente iperbolica e la funzione logistica, che hanno un gradiente nell'intervallo di valori [0;1]. Ciò significa che durante la backward propagation i vari gradienti che vengono moltiplicati per determinare la correzione dei parametri dei primi layer della rete, il cui numero dipende appunto da quanti layer è profonda la rete stessa, tendono a 0. Di conseguenza i layer più vicini agli input sono molto più difficili da addestrare di quelli vicini agli output e ciò può bloccare l'avanzamento dell'apprendimento della rete.

Due soluzioni sempre più comuni a questo problema sono le attivazioni ReLU e l'utilizzo di blocchi residui.

### Blocci residui

Nelle strutture delle reti che presenterò sono presenti dei collegamenti a livelli precedenti della rete, solitamente chiamati blocchi residui. Ciò perchè gli output di alcuni layer vengono passati in avanti senza essere modificati, come se prendessero una scorciatoia, e vengono poi sommati a layer successivi della rete. Questi layer avranno quindi un residuo da un livello meno profondo della rete. Durante l'addestramento ci saranno quindi alcuni layer speciali che, oltre a imparare a svolgere il compito necessario per ottenere l'output desiderato, si occupano di dare un contributo ai layer precedenti della rete a cui magari il gradiente della funzione di errore non arriva abbastanza grande da essere significativo nell'aggiustamento dei pesi.

### Attivazioni ReLU

Le funzioni di attivazione ReLU (Rectified Linear Unit) hanno la seguente forma:

$$ f(x) = max(0,x) $$

e sono sempre più utilizzate nelle DNN in quanto svolgono due compiti:

- introducono un grado di non linearità nel sistema, che permette solitamente alla rete di apprendere funzioni complesse in minor tempo;
- riducono il problema di scomparsa del gradiente in quanto non sono limitate a un intervallo che, se moltiplicato molte volte, tende a zero.

Nei modelli analizzati per questo motivo molti layer avranno attivazioni ReLU mentre altri avranno semplici attivazioni lineari (ovvero il risultato dell'attivazione è identico all'output del layer).

## Pixel-shuffle

Molte delle prime reti neurali per super-risoluzione preprocessavano l'immagine a bassa risoluzione di input con un upsample bicubico, di cui parlerò più avanti. In seguito l'immagine già delle dimensioni uguali a quella dell'output atteso veniva passata alla rete che cercava di migliorarne appunto la risoluzione. Questo rendeva l'addestramento più semplice per la rete ma molto più lento, in quanto l'immagine di input aveva già dimensioni notevoli, aumentando esponenzialmente i calcoli richiesti durante i forward dei vari layer. Per ovviare a questo problema è stato introdotto qualche anno fa un layer chiamato pixel-shuffle, anche noto come layer di convoluzione sub-pixel. Questo layer mescola appunto i canali di un'immagine a bassa risoluzione per generarne una con meno canali ma con dimensioni maggiori. Praticamente consiste nel riorganizzare le dimensioni del tensore dell'immagine, ma anche nel mescolare tra loro i vari pixel durante l'operazione. Matematicamente la funzione applicata è la seguente:

$$ PS(T)_{x,y,c} = T_{x//r , y//r , C\cdot r\cdot x\%r + C\cdot y\%r + c } $$

e trasforma un'immagine [H x W x Crr] in una immagine  [rH x rW x C].
Nella formula il simbolo " // " rappresenta il quoziente della divisione intera mentre " % " rappresenta il resto.
Se l'immagine invece è in formato channel-first, ovvero ordinata come [C,H,W], la funzione di pixel-shuffle è leggermente diversa ma analoga. In Byron sono state implementate entrambe le versioni, in modo da garantire la massima versatilità possibile.

Faccio notare che la funzione PixelShuffle della libreria Pytorch opera su immagini [C,H,W] mentre la funzione depth_to_space della libreria Tensorflow esegue il pixel-shuffle su immagini [H,W,C].

Il vantaggio principale nell'utilizzo di questo layer è l'incremento di velocità della rete: infatti utilizzando questo layer alla fine (o comunque in uno dei layer finali) della rete, è possibile estrarre tutte le feature necessarie per la super-risoluzione (che in questo caso sono i vari filtri imparati) direttamente dall'immagine a bassa risoluzione di input, applicando vari layer di convoluzione, per poi riorganizzarle nell'immagine finale di dimensioni volute.

![Schema esplicativo del layer di pixel shuffle applicato dopo un layer di convoluzione. \label{pixshuff}](immagini/pixelshuffle.png){ width=100% }

