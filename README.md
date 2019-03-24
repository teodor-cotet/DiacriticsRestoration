# Diacritics restoration for Romanian

The full article can be found [here](https://bitbucket.org/teodor_cotet/diacritics/src/master/article/Diacritics%20ConsILR%202018.pdf).  
Implementation was done using [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/).   
The core of the implementation is done in [model_diacritice.py](https://bitbucket.org/teodor_cotet/diacritics/src/master/model_diacritice.py).  
As a corpora we used parliamentary debates (see *Corpora* section).  
 
## Corpora

| **PAR**          |                                        | ROWIKI |         |       |         |
|--------------|----------------------------------------|--------|---------|-------|---------|
| Letters      | Letters with diacritics                | 15M    | 6.39%   | 38M   | 4.09%   |
|              | Letters that accept/contain diacritics | 84M    | 35.28%  | 296M  | 31.78%  |
|              | All letters                            | 239M   | 100.00% | 933M  | 100.00% |
| Words        | Words with diacritics                  | 13M    | 26.37%  | 33M   | 16.40%  |
|              | Words that accept/contain diacritics   | 35M    | 70.41%  | 118M  | 57.98%  |
|              | All words                              | 50M    | 100.00% | 204M  | 100.00% |
| Sentences    |                                        | 2.6M   |         | 22.0M |         |
| Unique words |                                        | 0.21M  |         | 2.62M |         |

Only PAR corpus was used in the end because of its higher quality.

## Models 

The architecutre of the models is briefly described in the picutre below. Several models were tried, using different branches of the architecture.
See the *Results* section for more details, or you can read the full [article](https://bitbucket.org/teodor_cotet/diacritics/src/master/article/Diacritics%20ConsILR%202018.pdf)

![architecture](https://bitbucket.org/teodor_cotet/diacritics/raw/0811ae10fdae3da2b2f07cb014c8d5055d1ec812/imgs/architecture.png)

## Results 

| Model                   | Char Embedding | Char LSTM | Hidden   | Epochs | Dev char acc (%) | Test char acc (%) | Test word acc (%) |
|-------------------------|----------------|-----------|----------|--------|------------------|-------------------|-------------------|
| Chars                   | 16             | 32        | 32       | 5      | 98.865           | 98.864            | 97.413            |
| Chars                   | 20             | 64        | 256      | 5      | 99.012           | 99.017            | 97.750            |
| Chars (5 classes)       | 16             | 32        | 32       | 5      | 99.048           | 99.068            | 97.867            |
| Chars                   | 24             | 64        | 64       | 4      | 99.064           | 99.057            | 97.856            |
| Chars + sentence        | 20             | 64        | 256      | 3      | 99.068           | 99.065            | 97.881            |
| Chars + word            | 20             | 64        | 256      | 4      | 99.309           | 99.329            | 98.453            |
| Chars + word + sentence | 20             | 64        | 256      | 5      | 99.365           | **99.378**            | **98.573**            |
| Chars + word + sentence | 20             | 64        | 256, 128 | 5      | **99.380**           | 99.366            | 98.553            |


Best models performance per letter:

| Model       | Letter | Precision (%) | Recall (%) | F-Score (%) |
|-------------|--------|---------------|------------|-------------|
| All-256-128 | “a”    | 99.16         | 98.86      | 99.01       |
|             | “ă”    | 96.29         | 97.31      | 96.80       |
|             | “â”    | 99.17         | 98.80      | 98.99       |
|             | “i”    | 99.97         | 99.96      | 99.97       |
|             | “î”    | 99.65         | 99.72      | 99.69       |
|             | “s”    | 99.84         | 99.84      | 99.84       |
|             | “ș”    | 99.44         | 99.43      | 99.43       |
|             | “t”    | 99.84         | 99.77      | 99.80       |
|             | “ț”    | 98.97         | 99.29      | 99.13       |



## Run the model for restoration

* Example usage:
```sh
    python3 model_diacritice.py -no_fast -load saved_models_diacritice/chars24-64 -no_word -no_sent -classes 4 -no_dep -no_tag -restore input.txt
```

The model will read text from input.txt (utf-8) and restore the diacritics in tmp_res.txt file. Only the pre-trained models that use characters are available on this git. 
