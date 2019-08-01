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
    '''
conn = sqlite3.connect('labeled_text.db')
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
