#!/usr/bin/env bash

python run_classifier.py \
--do_train=true \
--do_eval=true \
--data_dir=$DATA_DIR \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=5 \
--eval_batch_size=5 \
--learning_rate=3e-5 \
--num_train_epochs=4.0 \
--output_dir=$OUTPUT_DIR