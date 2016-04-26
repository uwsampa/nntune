FANNDIR=../fann-snnap
CC=gcc
CWDIR=cluster-workers
CFLAGS += -O3 -I$(FANNDIR)/src/include -I$(FANNDIR)/src/
FLOATFLAGS += $(FANNDIR)/src/floatfann.c
FIXEDFLAGS += -DFIXEDFANN $(FANNDIR)/src/fixedfann.c
LDFLAGS += -lm

.PHONY: all train recall

all: train recall

train: train.o
	$(CC) $(CFLAGS) $(FLOATFLAGS) $(LDFLAGS) $^ -o $@

recall: recall.o
	$(CC) $(CFLAGS) $(FLOATFLAGS) $(LDFLAGS) $^ -o $@

recall_fix: recall.o
	$(CC) $(CFLAGS) $(FIXEDFLAGS) $(LDFLAGS) $^ -o $@


# Build cluster-workers (slurm-only)
.PHONY: cluster-workers

cluster-workers: $(CWDIR)/cw/*.py
	cd $(CWDIR); sudo python setup.py install

CLEANME := train train.o recall recall.o \
	*.out *.log *.csv *.data *.nn ann.pdf

.PHONY: clean
clean:
	rm -f $(CLEANME)
	rm -rf ann/
