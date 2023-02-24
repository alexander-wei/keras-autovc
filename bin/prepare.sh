#!/usr/bin/bash

/opt/conda/bin/python3 ./preprocessor.py \
		       -o ./bin/spects.pk \
		       -e 32,4,./bin/embs.pk \
		       -u .165 -v 23.97 -s 0. -t 1.778 \
		       -I p111,p112,p113,p114 \
		       /path/to/wav/files/rootdir
