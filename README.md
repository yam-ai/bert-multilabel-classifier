# Multilabel Text Classification with BERT

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
CREATE INDEX IF NOT EXISTS label_index ON labels (label);
CREATE INDEX IF NOT EXISTS text_id_index ON labels (text_id);
```
An empty example sqlite file is in [`example/data.db`](https://github.com/yam-ai/bert-multilabel-classifier/blob/master/example/data.db).

Let us take the [toxic comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) published on [kaggle](https://www.kaggle.com/) as an example. (Note: you will need to create a kaggle account in order to download the dataset.) The training data file `train.csv` (not provided by this repository) in the downloaded dataset has the following columns: `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. The last six columns represent the labels of the `comment_text`.

The python script in [`example/csv2sqlite.py`](https://github.com/yam-ai/bert-multilabel-classifier/blob/master/example/csv2sqlite.py) can process `train.csv` and save the data in a sqlite file `data.db`.

To convert `train.csv` to `data.db`, run the following commands:
```sh
python3 csv2sqlite.py -i /downloads/toxic-comment/train.csv -o /repos/bert-multilabel-classifier/example/data.db
```
You can also use the `-n` flag to convert only a subset of examples in the training csv file to reduce the training database size. For example, you can use `-n 1000` to convert only the first 1,000 examples in the csv file into the training database. This may be necessary if there is not enough memory to train the model with the entire raw training set or you want to shorten the training time.

### 2. Download the pretrained model
Download and extract pretrained models from [BERT](https://github.com/google-research/bert), such as the [BERT-Base, Multilingual Cased](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) model.


### 3. Tune the hyperparameters
The training hyperparameters such as `num_train_epochs`, `learning_rate`, `max_seq_length`, `train_batch_size` can be modified in [`settings.py`](settings.py).


### 4. Train the model
Build the docker image for training:
```sh
docker build -f train.Dockerfile -t classifier-train .
```  

Run the training container by mounting the above volumes:
```sh
docker run -v $BERT_DIR:/bert -v $TRAIN_DIR:/train -v $MODEL_DIR:/model classifier-train
```

* `$BERT_DIR` is the full path where the downloaded BERT pretrained model is unzipped to, e.g., `/downloads/multi_cased_L-12_H-768_A-12/`.
* `$TRAIN_DIR` is the full path of the input directory that contains the sqlite DB `train.db` storing the training set, e.g., `/data/example/train/`.
* `$MODEL_DIR` is the full path of the output directory, e.g., `/data/example/model/`.
After training, it will contain a bunch of files, including a directory with number (a timestamp) as its name. For example, the directory `$MODEL_DIR/1564483298/` stores the trained model to be used for serving.

If you want to override the default settings with your modified settings, for example, in `/data/example/settings.py`, you can add the flag `-v /data/example/settings.py:/src/settings.py`.


### 5. Serve the model
Build the docker image for the classification server:
```sh
docker build -f serve.Dockerfile -t classifier-serve .
```

Run the serving container by mounting the output directory above and exposing the HTTP port:
```sh
docker run -v $MODEL_DIR/1564483298/:/model -p 8000:8000 classifier-serve
```


### 6. Post an inference HTTP request

Make an HTTP POST request to `http://localhost:8000/classifier` with a JSON body like the following:
```json
{ 
   "texts":[ 
      { 
         "id":0,
         "text":"Three great forces rule the world: stupidity, fear and greed."
      },
      { 
         "id":1,
         "text":"Put your hand on a hot stove for a minute, and it seems like an hour. Sit with a pretty girl for an hour, and it seems like a minute. That's relativity."
      }
   ]
}
```
Then in reply you will get back a list of scores, indicating the likelihoods of the labels for the input texts (e.g., two Albert Einstein quotes as follows):
```json
[ 
   { 
      "id": 0,
      "scores": {
         "toxic": 0.9373089075088501,
         "severe_toxic": 0.0010259747505187988,
         "obscene": 0.013565391302108765,
         "insult": 0.03743860125541687,
         "identity_hate": 0.006350785493850708,
         "threat": 0.0046683549880981445
      }
   },
   { 
      "id": 1,
      "scores": {
         "toxic": 0.00045806169509887695,
         "severe_toxic": 0.000041812658309936523,
         "obscene": 0.0001443624496459961,
         "insult": 0.00014069676399230957,
         "identity_hate": 0.000054776668548583984,
         "threat": 0.000050067901611328125
      }
   }
]
```

You can test the API using `curl` as follows:

```sh
curl -X POST http://localhost:8000/classifier -H "Content-Type: application/json" -d $'{"texts":[{"id":0,"text":"Three great forces rule the world: stupidity, fear and greed."},{"id":1,"text":"Put your hand on a hot stove for a minute, and it seems like an hour. Sit with a pretty girl for an hour, and it seems like a minute. That\'s relativity."}]}'
```
You will get the response like the following:
```sh
[{"id": 0, "scores": {"toxic": 0.9373089075088501, "severe_toxic": 0.0010259747505187988, "obscene": 0.013565391302108765, "insult": 0.03743860125541687, "identity_hate": 0.006350785493850708, "threat": 0.0046683549880981445}}, {"id": 1, "scores": {"toxic": 0.00045806169509887695, "severe_toxic": 4.1812658309936523e-05, "obscene": 0.0001443624496459961, "insult": 0.00014069676399230957, "identity_hate": 5.4776668548583984e-05, "threat": 5.0067901611328125e-05}}]
```

### 7. Using a GPU
If GPU is available, acceleration of training and serving can be acheived by running [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker). The base image in [`train.Dockerfile`](https://github.com/yam-ai/bert-multilabel-classifier/blob/master/train.Dockerfile) and [`serve.Dockerfile`](https://github.com/yam-ai/bert-multilabel-classifier/blob/master/serve.Dockerfile) should also be changed to the GPU version, i.e., `tensorflow:1.1x.y-gpu-py3`.

After building the docker image, run `docker` using the `nvidia` runtime:

```sh
docker run --runtime nvidia -v $BERT_DIR:/bert -v $TRAIN_DIR:/train -v $MODEL_DIR:/model classifier-train
```
or 
```sh
docker run --runtime nvidia -v $MODEL_DIR/1564483298/:/model -p 8000:8000 classifier-serve
```

If you are building the project from the source code directly (i.e., not using Docker), you also need to modify [`requirements.txt`](https://github.com/yam-ai/bert-multilabel-classifier/blob/master/requirements.txt) to use `tensorflow-gpu`.


### 8. Pull docker images from Docker Hub
We have published the docker images on the [Docker Hub](https://hub.docker.com/r/yamai/bert-multilabel-classifier). You can pull them directly from the Docker Hub as follows:
```sh
docker pull yamai/bert-multilabel-classifier:train-latest
docker pull yamai/bert-multilabel-classifier:serve-latest
```

After these images are successfully pulled, you can run the training or serving container as follows:
```sh
docker run -v $TRAIN_DIR:/train -v $MODEL_DIR:/model yamai/bert-multilabel-classifier:train-latest
```
or
```sh
docker run -v $MODEL_DIR:/model -p 8000:8000 yamai/bert-multilabel-classifier:serve-latest
```


If GPU is available, you can pull and use the images with tags `gpu-train-latest` and `gpu-serve-latest` instead:
```sh
docker pull yamai/bert-multilabel-classifier:gpu-train-latest
docker pull yamai/bert-multilabel-classifier:gpu-serve-latest
```


### 9. Profesional services

If you need any consultancy and support services from [YAM AI Machinery](https://www.yam.ai), please find us at:
* https://www.yam.ai
* https://github.com/yam-ai
* https://twitter.com/theYAMai
* https://www.linkedin.com/company/yamai
* https://www.facebook.com/theYAMai
