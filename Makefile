FANNDIR=FANN-2.2.0-Source
CFLAGS += -I$(FANNDIR)/src/include
LDFLAGS += -L$(FANNDIR)/src -lfann

ifeq ($(shell uname -s),Darwin)
		LIBEXT := dylib
else
		LIBEXT := so
endif

LIBFANN=$(FANNDIR)/src/libfann.$(LIBEXT)

.PHONY: all
all: $(LIBFANN) train

train: train.o
	$(CC) $(LDFLAGS) $^ -o $@

$(LIBFANN): $(FANNDIR)/src/*.c
	cd $(FANNDIR) ; cmake .
	cd $(FANNDIR) ; make

.PHONY: clean
clean:
	rm -f train train.o
