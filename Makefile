.PHONY: all
all: recall

FANDIR=FANN-2.2.0-Source

CFLAGS += -I$(FANNDIR)/src/include
LDFLAGS += -L$(FANNDIR)/src -lfann

train: train.o
	$(CC) $(LDFLAGS) $^ -o $@

jmeint.nn: train test/jmeint.data
	./$^

.PHONY: clean
clean:
	rm -f train train.o
