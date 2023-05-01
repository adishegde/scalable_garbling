#!/usr/bin/env bash
prog=$1
num=$2
args=${@:3}
for i in $(seq 1 $(( num - 1 )))
do
    $prog --id $i $args >"./data/tmp/party_${i}.txt" 2>&1 &
done
$prog --id 0 $args | tee "./data/tmp/party_0.txt"
