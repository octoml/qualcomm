#!/bin/bash

help()
{
    echo "Usage: run_benchmark.sh [args]"
    echo ""
    echo "Arguments:"
    echo "-d, --device            Device hash"
    echo "-m, --model             Path to model file"
    echo "-t, --type              Key of device in tracker (default: FP16)"
    echo "-h, --help              Print this help message"
    exit 1
}

SHORT=d:,m:,t:,h
LONG=device:,model:,type:,help
OPTS=$(getopt -a -n run_benchmark --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0  ]; then
      help
fi

eval set -- "$OPTS"
echo $OPTS

while :
do
  case "$1" in
    -d | --device )
      DEVICE="$2"
      shift 2
      ;;
    -m | --model )
      MODEL="$2"
      shift 2
      ;;
    -t | --type )
      TYPE="$2"
      shift 2
      ;;
    -h | --help)
      help
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
done


file_name=${MODEL##*/}

if [ "$DEVICE" == "" ]; then
    preamble="adb"
else
    preamble="adb -s ${DEVICE}"
fi

if [ "$TYPE" != "FP32" ]; then
    echo "Using FP16 precision"
    infer_type="--gpu_precision_loss_allowed=true"
else
    echo "Using FP32 precision"
    infer_type="--gpu_precision_loss_allowed=false"
fi

$preamble push $MODEL /data/local/tmp
$preamble shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/${file_name} --num_runs=500 --use_gpu=true --gpu_backend=cl $infer_type
$preamble shell rm -f /data/local/tmp/${file_name}
