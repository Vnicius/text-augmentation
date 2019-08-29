# Aumento da Bases de Dados de Texto

## Pré-processamento

`preprocess.py`

```
usage: preprocess.py [-h] [-o OUTPUT] [-l LENGTH] [--for-augment] input_file

positional arguments:
  input_file            Input file path

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory
  -l LENGTH, --length LENGTH
                        Max text length
  --for-augment         Preprocess the data for future augmentation
```

### Exemplo

- Pré-processamento de arquivos para treinamento

```
    $   python preprocess.py file.txt -o prep
```

- Pré-processamento de arquivos para treinamento limitando o tamanho das sentenças

```
    $   python preprocess.py file.txt -o prep -l 100
```

- Pré-processamento de arquivos para augmentation

```
    $   python preprocess.py file.txt -o output --for-augment
```

## Treinamento

`train.py`

```
usage: train.py [-h] [-o OUTPUT] [--embedding_size EMBEDDING_SIZE]
                [--lstm_size LSTM_SIZE] [-s STEPS] [-e EPOCHS]
                [-c CHECKPOINT_DIR]
                data_dir

positional arguments:
  data_dir              Data directory

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory
  --embedding_size EMBEDDING_SIZE
                        Embedding Size
  --lstm_size LSTM_SIZE
                        LSTM size
  -s STEPS, --steps STEPS
                        Steps per epoch
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -c CHECKPOINT_DIR, --checkpoint_dir CHECKPOINT_DIR
                        Model directory
```

### Exemplo

- Treinameto do Language Model

```
    $   python train.py prep -e 10 -o model
```

- Treinameto do Language Model a partir de um checkpoint

```
    $   python train.py prep -e 10 -o model -c model
```

## Aumento

`augment.py`

```
usage: augment.py [-h] [--max-freq MAX_FREQ] [--top-k TOP_K]
                  [--n-augment N_AUGMENT] [--max-length MAX_LENGTH]
                  original_data original_data_preprocessed model_path
                  output_file

positional arguments:
  original_data         Original data file
  original_data_preprocessed
                        Original data file with preprocess
  model_path            Path of the model file
  output_file           Output file with the augmented data

optional arguments:
  -h, --help            show this help message and exit
  --max-freq MAX_FREQ   Max frequence needed to a word be augmented
  --top-k TOP_K         Top K words considered in the augmentations
  --n-augment N_AUGMENT
                        Number of augmentations of a word
  --max-length MAX_LENGTH
                        Max length of the line
```

### Exemplo

- Aumento de um arquivo de texto

```
    $   python augment.py texto.txt prep/prep_texto.txt model/model.pkl augmented.txt --max-freq 50 --top-k 20 --n-augment 10 --max-length 5
```