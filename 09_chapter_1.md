# Introduzione

## Riassunto dei capitoli

 In questo capitolo farò una breve introduzione al mondo delle reti neurali profonde e ai vari framework per implementarle.
 Nel secondo capitolo parlerò dei costituenti fondamentali delle reti che poi ho implementato, ovvero i layer di convoluzione e di pixel-shuffle, e di alcune altre funzioni necessarie per i miei scopi.
 Nel terzo capitolo introdurrò la super-risoluzione spiegando da dove è nata e a che punto è ora, con i vari modelli che ho scelto di replicare con la mia libreria.
 Nel quarto capitolo illustrerò i vari risultati ottenuti confrontando le performance e la bontà dei vari modelli e dei vari framework al variare di alcuni parametri come per esempio il numero di core utilizzati.
 Infine nell'ultimo capitolo mosterò alcune applicazioni delle tecniche esposte e alcuni possibili lavori futuri.

## Deep neural network

Le reti neurali artificiali (ANN) sono modelli di calcolo informatico-matematici composti da neuroni artificiali che si ispirano al funzionamento biologico del cervello umano e basati sull'interconnessione delle informazioni.

L’idea di poter replicare artificialmente il cervello, simulandone il funzionamento attraverso delle unità di calcolo, ha una storia che inizia dai primi anni Quaranta del secolo scorso (il primo neurone artificiale fu proposto da W.S. McCulloch e W. Pitts). Da allora le ANN si sono sviluppate diventando un fenomeno emergente della realtà odierna ed evolvendosi nelle cosiddette reti neurali profonde (DNN). Esse consistono semplicemente in reti neurali che però hanno degli strati nascosti (comunemente chiamati hidden layer) tra il livello degli input e quello degli outpu della rete. Questi layer possono essere di vario tipo, a seconda delle funzioni che svolgono.

Il campo di applicazione principale delle DNN è il machine learning, ovvero l'apprendimento di informazioni dall'esperienza guidato da algoritmi matematici adattivi e automatici. In parole povere, le reti neurali possono imparare a risolvere problemi molto complessi, se strutturate e addestrate nel modo giusto. Per fare ciò ovviamente devono prima apprendere: ciò avviene nella cossidetta fase di training, che può essere di vario tipo:

- **Supervisionato**: All'algoritmo vengono forniti sia i dati in input che i dati in output attesi, in modo che la rete aggiusti i suoi parametri per avvicinarsi sempre di più ad ottenere il risulatato desiderato, imparando una o più regole o in generale funzioni molto complesse che collegano una classe di input simili a quello dato con i rispettivi output.
- **Non supervisionato**: Al sistema vengono formiti solamente i dati in input, sperando che la rete stessa trovi qualche connessione logica o schema sottostante alla struttura dei dati. In questo caso gli output della rete non sono sempre di facile interpretazione ed è anche difficile capirne la validità.
- **Semi-supervisionato**: Questo è un modello ibrido in cui alcuni dati di input hanno i corrispetivi dati di output attesi mentre altri non sono etichettati. L'obiettivo è sempre quello di identificare le regole per trasformare gli input in modo da ottenere qualcosa il più simile possibile agli output. Si noti che il concetto di "similarità" dipende dalla rete e viene scelto da chi crea il suo modello. Alcuni esempi abbastanza usati sono le norme L1 e L2.
- **Per rinforzo**: Il sistema in questo caso interagisce con un ambiente dinamico e, una volta elaborati i dati in input, deve raggiungere un obiettivo. A seconda del risultato ottenuto verrà fornita una "ricompensa" o una "punizione", per far capire alla rete in quale direzione sta procedendo. Come anche in tutti gli altri casi, le routine di addestramento vengono ripetute moltissime volte finchè la rete non svolge le funzioni desiderate o smette di apprendere.

Ovviamente nel caso in cui la rete smetta di apprendere prima di raggiungere il suo funzionamento previsto, potrebbe essere il caso di cambiare metodo e parametri scelti durante l'addestramento oppure di rivedere la struttura della rete stessa.

## I vari framework

Durante gli anni sono state sviluppate moltissime librerie per l'implementazione delle reti neurali, che si differenziano tra loro per performance, semplicità di uso, linguaggio di programmazione usato e hardware supportato. Di seguito elencherò le principali utilizzate durante questo lavoro di tesi.

### Darknet

Darknet è un framework per reti neurali scritto in C da Joseph Redmon, con supporto nativo solo per sistemi operativi Linux (anche se è possibile utilizzarlo anche in ambiente Windows, con delle modifiche e una gran dose di pazienza) e ottimizzato per GPU (solo CUDA, quindi solo schede grafiche NVidia) e CPU (tuttavia la libreria di default per il calcolo parallelo a cui si appoggia è esclusiva Linux). Risulta una delle migliori librerie per reti neurali attualmente disponibili e open source, in termini di performance. Tuttavia ha alcuni aspetti che possono essere migliorati, come la compatibilità tra piattaforme diverse e una migliore ottimizzazione per il calcolo parallelo su CPU. Per questo motivo è nata l'idea di Byron, un porting in C++ di Darknet che per ora si concentra su questi punti. Un porting è una "traduzione" del codice da una piattaforma o linguaggio a un altro, che solitamente viene fatto per motivi di compatibilità o per migliorare le performance (come nel mio caso).

### Byron

Come detto sopra, Byron è un framework in C++ basato per la maggior parte sul codice sorgente di Darknet. Tuttavia essendo stata riscritta da zero, questa libreria ha innumerevoli miglioramenti (molti permessi dallo standard del C++ che contiene funzioni molto più avanzate rispetto allo standard del C) e inoltre per alcuni aspetti critici (tra cui la gestione dei core per il calcolo parallelo) adotta strategie nuove permettendo delle performance nettamente superiori a Darknet. Inoltre Byron ha anche alcune funzioni completamente assenti in Darknet, tra cui il layer di pixel-shuffle di cui parlerò più avanti che vede uso sempre maggiore nei modelli di reti neurali che elaborano le immagini e che permette l'implementazione delle migliori reti per super-risoluzione utilizzate al momento.

### Pytorch e Keras

Altri framework per reti neurali molto popolari al momento sono Pytorch e Keras. Entrambi sono scritti in Python, e di conseguenza sono pensati per essere di facile uso per l'utente e consentono di scrivere e impostare velocemente anche modelli complicati. Come svantaggio hanno tuttavia la lentezza dovuta al linguaggio stesso, che essendo di alto livello gestisce molti parametri automaticamente e non sempre nel modo ottimale. Pytorch è un porting in Byron della libreria Torch, scritta in Lua. Keras invece è un wrapping di un'altra libreria sempre scritta in python e C++ chiamata Tensorflow, pensato per essere più user-friendly senza perdere in termini di performance. Un wrapping è una interfaccia di codice che permette di usare codice sorgente scritto in un altro linguaggio o in generale più complicato e complesso da utilizzare.

Parlo di queste librerie perchè sono state in parte utilizzate durante il mio lavoro di tesi. Visti i lunghi tempi richiesti per la scrittura di una libreria così vasta e per il debugging necessario ad assicurarsi che funzionasse correttamente, ho scelto di non addestrare di persona le reti di cui parlerò più avanti. Questo avrebbe richiesto molti altri test oltre che ovviamente il tempo di addestramento, che per queste reti solitamente è superiore a una settimana sulle GPU più performanti del momento. Di conseguenza ho preso i pesi delle reti pre-addestrate, che però erano disponibili solamente per l'implementazione in Pytorch (per la rete EDSR) e per quella in Keras (per la WDSR). Ciò ha reso necessaria ovviamente la scrittura di ulteriore codice per la conversione dei pesi tra i vari modelli. Ho inoltre riscontrato che la versione dell'EDSR messa a disposizione nella repository ufficiale non funziona su CPU. Questo problema è noto ai programmatori che hanno fatto il porting della rete da Torch ma non è stato ancora risolto. Di conseguenza l'implementazione su Byron dell'EDSR è per ora l'unica (a mia conoscenza) funzionante su CPU.

Per la rete YOLO, che è nativa su Darknet, la conversione dei pesi è invece stata molto più semplice, poichè il formato dei dati è simile.
