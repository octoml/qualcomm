#!/bin/bash

RPC_KEY="android"
PATH_PREFIX="/data/local/tmp"

help()
{
    echo "Usage: launch_rpc.sh [args]"
    echo ""
    echo "Arguments:"
    echo "-d, --device            Device hash"
    echo "-t, --tracker           Tracker address (format: ip:port)"
    echo "-k, --key               Key of device in tracker (default: ${RPC_KEY})"
    echo "-p, --path              Path to directory with executable (default: ${PATH_PREFIX})"
    echo "-h, --help              Print this help message"
    exit 1
}

SHORT=d:,t:,k:,p:,h
LONG=device:,tracker:,key:,path:,help
OPTS=$(getopt -a -n launch_rpc --options $SHORT --longoptions $LONG -- "$@")

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
    -t | --tracker )
      TRACKER="$2"
      shift 2
      ;;
    -k | --key )
      RPC_KEY="$2"
      shift 2
      ;;
    -p | --path )
      PATH_PREFIX="$2"
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

if [ "$TRACKER" == "" ] || [ "$DEVICE" == "" ]; then
      help
fi

while true
do
   preamble='adb -s'
   body="shell cd ${PATH_PREFIX}; LD_LIBRARY_PATH=${PATH_PREFIX} ${PATH_PREFIX}/tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 --tracker=${TRACKER} --key=${RPC_KEY}"
    echo $preamble ${DEVICE} $body
    $preamble ${DEVICE} $body

    sleep 20
done
