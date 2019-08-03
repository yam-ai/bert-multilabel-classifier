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

import sqlite3
import csv

# Create tables
schema = '''
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
    '''
conn = sqlite3.connect('data.db')
cur = conn.cursor()
cur.executescript(schema)

# Insert data
with open('train.csv', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    for row in reader:
        id = row[0]
        text = row[1]
        cur.execute('INSERT INTO texts (id, text) VALUES (?,?)', (id, text))
        for i in range(2, 8):
            if row[i] == '1':
                cur.execute(
                    'INSERT INTO labels (label, text_id) VALUES (?,?)', (header[i], id))

conn.commit()
