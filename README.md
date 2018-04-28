# ikr-classifier
Projekt do předmětu IKR 2017/2018

## Klasifikace řeči
Výsledky v souboru `results_speech.txt`.
Spoštění pomocí `audio_GMM_run.py`.

## Lineární klasifikátor obrazu
Využívá základní lineární klasifikaci z ikrlib.py.
Výsledky v souboru `image_linear_results.txt`.
Spouští se soubor `image_linear.py**.

## Pokus o konvoluční neuronovou síť
Používá framework Keras pro implementaci konvoluční neuronové sítě
**Požadavky** Na strukturu dat.
  Data jsou uložena ve složce data, kde jsou podsložky train, validation, eval
  V prvních dvou jsou dále 2 složky ve kterých jsou oddělena data daného člověka
  a ostatních. Ve složce eval je jedna složka s obrázky. Tato struktura je nutná pro použití funkce z frameworku Keras

Výsledky jsou uloženy v souboru `results_cnn.txt`
Spouštění skriptu `python3 convnet.py`

