#!/usr/bin/env sh

mkdir -p data

if [ -f data/enwik9 ]; then
    exit 0
fi

curl -o data/enwik9.zip http://mattmahoney.net/dc/enwik9.zip

unzip data/enwik9.zip -d data

rm data/enwik9.zip

head -c 100000000 data/enwik9 > data/enwik8
