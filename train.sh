#!/usr/bin/env bash
# coding=utf-8
# Copyright 2019 YAM AI Machinery Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

python run_classifier.py \
--do_train=true \
--do_eval=true \
--data_sqlite=$DATA_SQLITE \
--train_eval_test_ratio=7,3,0 \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--init_checkpoint=$BERT_DIR/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=5 \
--eval_batch_size=5 \
--learning_rate=3e-5 \
--num_train_epochs=4.0 \
--output_dir=$OUTPUT_DIR