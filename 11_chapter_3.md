# Super-risoluzione

## Le origini

 La super-risoluzione è una tecnica in generale utilizzata per migliorare la risoluzione spaziale di un'immagine. Può essere applicata nella sua forma base solo a immagini che contengono aliasing, ovvero che sono state sotto-campionate o ridotte in dimensione. In queste immagini il contenuto ad alta frequenza dell'immagine ad alta risoluzione desiderata è nascosto nel contenuto a bassa frequenza dell'immagine di input. Di conseguenza applicare algoritmi per ripristinare questo contenuto (eventualmente dopo aver ingrandito l'immagine alle giuste dimensioni) consente di riprodurre immagini abbastanza simili all'output desiderato.

 I primi metodi di super-risoluzione di immagini digitali consistevano nella stima del contenuto ad alta frequenza tramite associazione di patch di immagine a bassa risoluzione con la corrispetiva immagine ad alta risoluzione. Queste patch venivano solitamente prese dopo aver filtrato l'immagine con un filtro di edge detection o dalla trasformata di Fourier dell'immagine per avere direttamente il contenuto in frequenza. Una volta imparato un "dizionario" di queste associazioni, da un insieme di coppie di immagini alta e bassa risoluzione note, era possibile applicarlo su un'immagine a bassa risoluzione e ottenerne una versione ad alta risoluzione. Si noti che in questo caso non necessariamente le immagini di input e output avevano dimensioni diverse: la risoluzione spaziale infatti dipende non solo dalle dimensioni dell'immagine ma anche dal passo di campionamento.

 Nel 2014, grazie al lavoro del Department of Information Engineering dell'università di Hong Kong, è nata l'idea di utilizzare i layer di convoluzione delle ormai sempre più popolari reti neurali per imparare in maniera automatizzata un analogo del dizionario delle patch, che in questo caso diventava un insieme di moltissime feature. Nacque così la prima ANN per super-risoluzione, chiamata SRCNN (Super Resolution Convolutional Neural Network), che consisteva semplicemente in tre layer di convoluzione. Il primo estraeva le patch a bassa risoluzione dall'immagine, il secondo collegava queste patch organizzate in un vettore a molte dimensioni a un altro vettore che idealmente rappresentava le patch ad alta risoluzione. Infine l'ultimo layer riorganizzava le patch ad alta risoluzione e le combinava per ottenere le immagini ad alta risoluzione di output.

 Da allora sono stati fatti molti progressi nella ricerca in questo campo, ma le idee fondamentali non sono cambiate. Semplicemente i modelli sono molto più profondi e utilizzano quindi contromisure per riuscire a gestire l'addestramento con un numero enorme di parametri imparabili.

## Preprocessing

Ogni rete a super-risoluzione deve ovviamente essere addestrata ed ha quindi bisogno di un dataset di immagini di input e di output attesi. Nelle reti di cui mi sono occupato in questo lavoro di tesi, l'immagine di input consiste di una versione ridimensionata dell'immagine di output. Questo introduce un fattore di aliasing che la rete dovrà quindi imparare a nullificare quando ingrandisce l'immagine, ripristinando i contenuti ad alta frequenza e quindi la risoluzione dell'output ottenuto. Queste reti sono molto grandi e hanno milioni di parametri: di conseguenza se l'input fosse modificato con dei filtri per simulare i problemi realistici delle immagini reali come rumore e sfocature, la rete imparerebbe in parte anche a risolvere questi problemi. Anche se i modelli di queste reti sono pensati per uno scopo diverso, ciò non evita che le feature estratte dai numerosi layer di convoluzione presenti risentano dei filtri applicati all'input e di conseguenza durante l'addestramento i pesi verranno aggiustati diversamente.

Per le analisi svolte in questa tesi ho considerato per semplicità (e anche per la facilità nel reperire i pesi dei modelli già precedentemente addestrati) solamente un fattore di scala di 4 e un metodo di riduzione dell'immagine bicubico.

### Ricampionamento bicubico

Il ricampionamento bicubico, che solitamente viene diviso in upsampling e downsampling o analogamente chiamati upscaling e downscaling, consiste in un metodo di interpolazione per determinare i pixel di un'immagine dopo che questa è stata cambiata di dimensioni (rispettivamente aumentata o diminuita). Il nome deriva dalla complessità massima dell'algoritmo di interpolazione usato, in cui l'operazione più complicata eseguita in questo caso è appunto il cubo del valore di un pixel. L'interpolazione viene eseguita in un intorno di 4 pixel. In generale i filtri scelti per il ricampionamento bicubico appartengono a una famiglia con la seguente forma:

$$ k(x) = \frac{1}{6}\left\{\begin{matrix}
(12-9B-6C)|x|^{3} + (-18+12B+6C)|x|^{2} + & \\
+ (6-2B) & se \ |x| < 1 \\
 (-B-6C)|x|^{3} + (6B+30C)|x|^{2} + & \\
(-12B-48C)|x| + (8B+24C) & se \ 1 \leq |x| <2 \\
0 & altrimenti
\end{matrix}\right.$$

Tra i più popolari cito le opzioni B=0, C=0.75 usata da OpenCV e Photoshop o B=0, C=0.5 solitamente chiamato filtro di Catmull-Rom usato da GIMP.
Byron utilizza di default l'opzione (0, 0.75) ma può essere agevolmente cambiata secondo necessità. Inoltre in Byron ho anche implementato il filtro di Lanczos, che garantisce risultati migliori (soprattutto quando si parla di upsampling dove si vede effettivamente il miglioramento di qualità rispetto a un upsample lineare) e ha un intorno di 8 pixel.

## EDSR


## WDSR
