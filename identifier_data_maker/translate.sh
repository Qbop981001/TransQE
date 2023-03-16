#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup  python translate.py mono.zh zh-translated.en zh en &
CUDA_VISIBLE_DEVICES=1 nohup  python translate.py mono.ja ja-translated.en jap en &
CUDA_VISIBLE_DEVICES=2 nohup python translate.py mono.de de-translated.en de en &
CUDA_VISIBLE_DEVICES=3 nohup python translate.py mono.en en-translated.ja en jap &