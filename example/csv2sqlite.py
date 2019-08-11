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

import sys
import sqlite3
import csv
import getopt

dbfile = 'data.db'
trainfile = 'train.csv'
numitems = -1

# Create tables
schema = '''
        DROP TABLE IF EXISTS texts;
        CREATE TABLE texts (
            id TEXT NOT NULL PRIMARY KEY,
            text TEXT NOT NULL
        );
        DROP TABLE IF EXISTS labels;
        CREATE TABLE labels (
            label TEXT NOT NULL,
            text_id text NOT NULL,
            FOREIGN KEY (text_id) REFERENCES texts(id)
        );
        DROP INDEX IF EXISTS label_index;
        CREATE INDEX label_index ON labels (label);
    '''
try:
    opts, _ = getopt.getopt(sys.argv[1:], 'i:o:n:')
except getopt.GetoptError as e:
    print('Invalid argument: {}'.format(e), file=sys.stderr)
    sys.exit(1)
for opt, arg in opts:
    if opt == '-o':
        dbfile = arg
    if opt == '-i':
        trainfile = arg
    if opt == '-n':
        try:
            numitems = int(arg)
        except:
            print('Invalid number of entries for -n', file=sys.stderr)
            sys.exit(1)

print('number of entries = {}\nsource training csv file = {}\ntarget training sqlite file = {}'.format(numitems, trainfile, dbfile))

try:
    conn = sqlite3.connect(dbfile)
except:
    print('Failed to open sqlite file {}'.format(dbfile), file=sys.stderr)
    sys.exit(1)
cur = conn.cursor()
cur.executescript(schema)

# Insert data
try:
    with open(trainfile, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        print('header = {}'.format(header))
        print()
        l = 0
        for row in reader:
            l = l + 1
            if numitems >= 0:
                if l > numitems:
                    print('Loaded {} entries.'.format(l-1, dbfile))
                    break;
            if l % 1000 == 0:
                print('Loading {} entries...'.format(l))
            id = row[0]
            text = row[1]
            cur.execute('INSERT INTO texts (id, text) VALUES (?,?)', (id, text))
            for i in range(2, 8):
                if row[i] == '1':
                    cur.execute(
                        'INSERT INTO labels (label, text_id) VALUES (?,?)', (header[i], id))
except IOError as e:
    print('Failed to open training csv file {}: {}'.format(trainfile, e), file=sys.stderr)
    sys.exit(1)
except sqlite3.Error as e:
    print('Failed to write training data to database file {}: e'.format(dbfile, e), file=sys.stderr)
    sys.exit(1)

try:
    conn.commit()
except sqlite3.Error as e:
    print('Failed to commit writes to database file {}: e'.format(dbfile, e), file=sys.stderr)
    sys.exit(1)
