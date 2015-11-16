FANNDIR=../fann-snnap
CC=gcc
CWDIR=cluster-workers
WEIGHTLIM=8
CFLAGS += -O3 -I$(FANNDIR)/src/include -I$(FANNDIR)/src/ -DWEIGHT_LIM=$(WEIGHTLIM)
FLOATFLAGS += $(FANNDIR)/src/floatfann.c
FIXEDFLAGS += -DFIXEDFANN $(FANNDIR)/src/fixedfann.c
# LDFLAGS += -L$(FANNDIR)/src -lfann -lm
LDFLAGS += -lm

ifeq ($(shell uname -s),Darwin)
		LIBEXT := dylib
else
		LIBEXT := so
endif

LIBFANN=$(FANNDIR)/src/libfann.$(LIBEXT)

.PHONY: all
all: train recall recall_fix

install: cluster-workers

train: train.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(FLOATFLAGS) $^ -o $@

recall: recall.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(FLOATFLAGS) $^ -o $@

recall_fix: recall.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(FIXEDFLAGS) $^ -o $@

cluster-workers: $(CWDIR)/cw/*.py
	cd $(CWDIR); sudo python setup.py install

CLEANME := train train.o recall_fix recall recall.o \
	*.out *.log *.csv *.data *.nn ann.pdf

.PHONY: clean
clean:
	rm -f $(CLEANME)
