#!/bin/bash

function logi() {
    echo -e "run.sh	`date +"%Y-%m-%d %H:%M:%S"`	INFO	$1"
}

function logw() {
    echo -e "run.sh	`date +"%Y-%m-%d %H:%M:%S"`	WARN	$1"
}

function preprocess() {
    python src/prepare.py --data_dir=$1
}

function train() {
    python src/trainer.py conf/model.config
}

function main() {
    cmd=$1
    args=${@:2}
    logi "------------start. $cmd $args---------"
    $cmd $args
    logi "------------finished. $cmd $args------"
}
main $@
