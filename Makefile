FANNDIR=../fann
CWDIR=cluster-workers
CFLAGS += -I$(FANNDIR)/src/include
LDFLAGS += -L$(FANNDIR)/src -lfann -lm

ifeq ($(shell uname -s),Darwin)
		LIBEXT := dylib
else
		LIBEXT := so
endif

LIBFANN=$(FANNDIR)/src/libfann.$(LIBEXT)

.PHONY: all
all: train recall

install: cluster-workers

train: train.o
	$(CC) $(LDFLAGS) $^ -o $@

recall: recall.o
	$(CC) $(LDFLAGS) $^ -o $@

cluster-workers: $(CWDIR)/cw/*.py
	cd $(CWDIR); sudo python setup.py install

CLEANME := train train.o recall recall.o \
	*.out *.log *.csv

.PHONY: clean
clean:
	rm -f $(CLEANME)
