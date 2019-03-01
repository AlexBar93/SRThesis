# Framework

Durante gli anni sono state sviluppate molte librerie per l'implementazione delle reti neurali, che si differenziano tra loro per performance, semplicità di uso, linguaggio di programmazione usato e hardware supportato. Di seguito elencherò i principali framework utilizzati durante questo lavoro di tesi.

## Darknet

Darknet [@darknet] è un framework per reti neurali scritto in Ansi C da Joseph Redmon, dottorando all'Università di Washington in computer vision, con supporto nativo solo per sistemi operativi Linux (anche se sono stati effettuati vari porting per altre piattaforme tra cui Windows) e ottimizzato per GPU (solo CUDA [@cuda], quindi solo schede grafiche NVidia). Risulta una delle migliori librerie per reti neurali applicate al campo dell'object detection attualmente disponibili e open source, in termini di performance e risultati [@yolov3], grazie all'implementazione nativa in questo framework della rete YOLOv3. Tuttavia ha alcuni aspetti che possono essere migliorati, come la compatibilità tra piattaforme diverse e una migliore gestione del calcolo parallelo su CPU. Per questo motivo è nata l'idea di Byron, un porting in C++ di Darknet che per ora si concentra su questi punti e sull'estensione del framework con nuove funzioni. Un porting è una "traduzione" del codice da una piattaforma o linguaggio a un altro, che solitamente viene fatto per motivi di compatibilità o per migliorare le performance (come nel mio caso).

## PyTorch e Keras

Altri framework per reti neurali molto popolari al momento sono PyTorch [@pytorch] e Keras [@keras]. Entrambi sono scritti in Python, e di conseguenza sono pensati per essere di facile uso per l'utente e consentono di scrivere e impostare velocemente anche modelli complicati. PyTorch è un porting in Python della libreria Torch, scritta in Lua. Essendo scritto quasi completamente in Python, questo framework offre un tradeoff tra performance e semplicità d'uso, in quanto Python è un linguaggio di alto livello e di conseguenza generalmente meno performante in quanto gestisce alcune variabili, tra cui la memoria allocata, in maniera automatica e non sempre nel modo ottimale. Keras invece è un wrapping di un'altra libreria scritta in Python chiamata Tensorflow, che a sua volta è un wrapping della versione in C++ della stessa libreria. Un wrapping è una interfaccia di codice che permette di usare codice sorgente scritto in un altro linguaggio o in generale più complicato e complesso da utilizzare. Ciò solitamente permette all'utente di scrivere codice più facilmente e più velocemente, tuttavia in questo caso c'è un tempo di overhead poichè il computer deve "tradurre" le istruzioni dal livello più alto (in questo caso Keras) al livello più basso (in questo caso Tensorflow in C++). Faccio notare che comunque questo tempo di overhead è molto inferiore solitamente rispetto ai tempi di calcolo effettivamente necessari nei vari layer della rete, e quindi sia Keras che Tensorflow sono entrambe librerie molto performanti e molto generiche, in grado di consentire l'implementazione di modelli di reti neurali di vario tipo. Inoltre le considerazioni che ho fatto finora valgono per quanto riguarda l'utilizzo su CPU di queste librerie, in quanto per l'utilizzo su GPU tutti questi framework si appoggiano a librerie esterne come cuDNN[@gemm].

Parlo di queste librerie perchè sono state in parte utilizzate durante il mio lavoro di tesi. Visti i lunghi tempi richiesti per la scrittura di una libreria così vasta e per il debugging necessario ad assicurarsi che funzionasse correttamente, ho scelto di non addestrare di persona le reti di cui parlerò più avanti. Questo avrebbe richiesto molti altri test oltre che ovviamente il tempo di addestramento, che per queste reti solitamente è superiore a una settimana sulle GPU più performanti del momento [@edsr, @wdsr]. Di conseguenza ho preso i pesi delle reti pre-addestrate, che però erano disponibili solamente per l'implementazione in PyTorch (per la rete EDSR) e per quella in Keras (per la WDSR). Ciò ha reso necessaria ovviamente la scrittura di ulteriore codice per la conversione dei pesi tra i vari modelli. Ho inoltre riscontrato che la versione dell'EDSR messa a disposizione nella repository ufficiale non compila su CPU. Questo problema è noto ai programmatori che hanno fatto il porting della rete da Torch ma non è stato ancora risolto. Di conseguenza l'implementazione su Byron dell'EDSR è per ora l'unica (a mia conoscenza) funzionante su CPU.

## Byron

Come detto sopra, Byron è un framework in C++ (standard 2017) basato per la maggior parte sul codice sorgente di Darknet. Tuttavia essendo stata riscritta da zero, questa libreria ha numerose migliorie e inoltre per alcuni aspetti critici, come la gestione dei core, adotta strategie nuove permettendo delle performance nettamente superiori a Darknet. Sia in Byron che in Darknet la gestione dei core e dei thread del processore viene effettuata tramite la libreria OpenMP [@openmp]. Tuttavia questa libreria ha varie direttive per gestire la divisione dei compiti da svolgere durante il codice tra i vari thread. Per esempio, in Darknet la GEMM è implementata con il seguente codice:

```c
void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
```

dove la direttiva `#pragma omp parallel for` della libreria OpenMP apre una sessione parallela e implica che il ciclo `for` subito dopo di essa verrà svolto in parallelo ed ogni core si occuperà di una iterazione del ciclo. In problema è che la dichiarazione delle variabili di ciclo all'esterno della sessione parallela significa che tutti i thread effettueranno l'accesso alle stesse variabili, causando eventualmente problemi di concurrency.

Per risolvere questo problema, in Byron impiego diverse direttive di OpenMP:

- `#pragma omp parallel` viene usata solamente all'inizio del programma principale per aprire la sessione parallela che resterà aperta fino alla fine dell'esecuzione, in quanto tutti i vari loop della libreria saranno poi eseguiti in parallelo e non solo quelli della GEMM come avviene in Darknet;
- `#pragma omp taskgroup` e `#pragma omp taskloop`  permettono di gestire i cicli come il `for` specificando eventuali variabili che i thread devono vedere come private in modo da non sovrascrivere quelle di altri thread e permettono anche di scegliere quanti thread devono occuparsi di una data funzione, permettendo una divisione dei compiti della rete sui vari core in maniera ottimale a seconda della potenza di calcolo disponibile sulla macchina al momento dell'esecuzione del codice. Sebbene durante questo lavoro di tesi non abbia addestrato le reti neurali utilizzate, questa specifica implementazione che è presente in tutte le funzioni della libreria permette di gestire più liberamente la fase di addestramento dividendo per esempio i core del computer in vari compiti quali caricamento delle immagini, propagazione forward e backward nella rete e aggiornamento dei pesi.

Oltre a queste correzioni per quanto riguarda la parallelizzazione del codice, Byron ha anche alcune funzioni completamente assenti in Darknet, tra cui il layer di pixel-shuffle che vede uso sempre maggiore nei modelli di reti neurali che elaborano le immagini e che permette l'implementazione delle migliori reti per super-risoluzione utilizzate al momento.
Rispetto a Keras e Tensforflow il miglioramento principale consiste nell'implementazione del layer YOLO per la object detection. Infatti implementare questo layer direttamente in C++ all'interno di Tensorflow sarebbe parecchio arduo, a causa della struttura enorme e complessa del framework. Ed implementarlo in Python, per quanto leggermente più semplice, ridurrebbe drasticamente le performance.
Ciò rende Byron un framework ottimizzato per CPU multi-core e per le reti neurali che si occupano di object detection e super-risoluzione.
