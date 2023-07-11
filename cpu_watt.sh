#!/bin/env bash

powerstat -D -z 0.5 10000000 > $1
#powerstat -z 0.5 1000000 > $1
#sudo powerstat -D -z 0.5 10000000 > $1
