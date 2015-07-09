FANNDIR=FANN-2.2.0-Source
CFLAGS += -I$(FANNDIR)/src/include
LDFLAGS += -L$(FANNDIR)/src -lfann -lm

ifeq ($(shell uname -s),Darwin)
		LIBEXT := dylib
else
		LIBEXT := so
endif

LIBFANN=$(FANNDIR)/src/libfann.$(LIBEXT)

.PHONY: all
all: fann train recall

train: train.o
	$(CC) $(LDFLAGS) $^ -o $@

recall: recall.o
	$(CC) $(LDFLAGS) $^ -o $@

fann: $(FANNDIR)/src/*.c
	cd $(FANNDIR) ; cmake .
	cd $(FANNDIR) ; sudo make install

.PHONY: clean
clean:
	rm -f train train.o recall recall.o
	cd $(FANNDIR) ; make clean