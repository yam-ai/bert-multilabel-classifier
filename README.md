# Multilabel Classification with BERT

Bidirectional Encoder Representations from Transformers (BERT) is a recent Natural Language Processing (NLP) technique proposed by the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). As the paper describes:
> Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

Serveral BERT pretrained models have been opensourced by Google in the [BERT](https://github.com/google-research/bert) repository and can be further trained to more fine-tuned models for downstream tasks.

This project adapts [BERT](https://github.com/google-research/bert) to perform a specific task: multilabel classification on texts. The training and inference procedures are packaged in containers and can be called separately.

## Usage


### 1. Prepare the dataset as a sqlite database  
The training data is expected to be given as a [sqlite](https://www.sqlite.org/index.html) database. It consists of two tables, `texts` and `labels`, storing the texts and their associated labels:
```SQL
CREATE TABLE IF NOT EXISTS texts (
    id TEXT NOT NULL PRIMARY KEY,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    label TEXT NOT NULL,
    text_id text NOT NULL,
    FOREIGN KEY (text_id) REFERENCES texts(id)
);
CREATE INDEX label_index ON labels (label);
```
An empty example sqlite file is in `example/data.db`.

Let us take the [toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) as an example. The data file `train.csv` in this dataset (not included in this repository) has the following columns: `"id"`, `"comment_text"`, `"toxic"`, `"severe_toxic"`, `"obscene"`, `"threat"`, `"insult"`, `"identity_hate"`. The last six columns represent the labels of the `comment_text`.

The python script in `example/csv2sqlite.py` can process `train.csv` and save the data in a sqlite file.


### 2. Download pretrained models  
Download and extract pretrained models from [BERT](https://github.com/google-research/bert), such as the [BERT-Base, Multilingual Cased](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) model.


### 3. Modify parameters in `train.sh`  
The training parameters such as `train_batch_size`, `learning_rate`, `num_train_epochs`, `max_seq_length` can be modified in `train.sh`.


### 4. Train  
Build the docker image for training  
```sh
docker build -f train.Dockerfile -t classifier-train .
```  
Assume the paths of the pretrained model, the sqlite file and the desired output model directory are `$BERT_DIR`, `$DATA_SQLITE` and `$OUTPUT_DIR` respectively. Run the training container by mounting the above volumes:
```sh
docker run -v $BERT_DIR:/bert -v $DATA_SQLITE:/data.db -v $OUTPUT_DIR:/output classifier-train
```

At the end of training, `$OUTPUT_DIR` should contain a bunch of files, including a directory with number (a timestamp) as its name. For example, it has the form `$OUTPUT_DIR/1564483298/`. This is the output directory used for serving.


### 5. Serve  
Build the docker image for serving  
```sh
docker build -f serve.Dockerfile -t classifier-serve .
```

Run the serving container by mounting the output directory above and expose a port:
```sh
docker run -v $OUTPUT_DIR/1564483298/:/model -p 8000:8000 classifier-serve
```


### 6. Make a post call to `http://localhost:8000/classifier` with a JSON body:
```json
{
    "texts": [
        {
            "id": 0,
            "text": "Testing comment"
        }
    ]
}
```
Then in reply we should get back a list of scores, indicating the likelihood of the labels for the input texts:
```json
[
    {
        "scores": {
            "threat": 0.0047899698838591576,
            "obscene": 0.015161020681262016,
            "identity_hate": 0.0059020197950303555,
            "toxic": 0.9870702028274536,
            "insult": 0.015693070366978645,
            "severe_toxic": 0.003254803130403161
        },
        "id": 0,
        "text": "Testing comment"
    }
]
```


### 7. GPU
If GPU is available, acceleration of training and serving can be acheived by running `nvidia-docker`. The base image in `train.Dockerfile` and `serve.Dockerfile` should also be changed to the GPU version.