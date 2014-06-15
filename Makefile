.PHONY: all
all: recall

FANNDIR=FANN-2.2.0-Source

CFLAGS += -I$(FANNDIR)/src/include
LDFLAGS += -L$(FANNDIR)/src -lfann

jmeint_train: jmeint_train.o
	$(CC) $(LDFLAGS) $^ -o $@

jmeint: jmeint.o tritri.o
	$(CC) $^ -o $@

jmeint_recall: jmeint_recall.o
	$(CC) $(LDFLAGS) $^ -o $@

jmeint.all.data: jmeint
	./$< 100000 > $@

jmeint.training.fann jmeint.testing.fann: divide_data.py jmeint.all.data
	./$^

jmeint.training.arff jmeint.testing.arff: divide_data.py jmeint.all.data
	./$^ arff

jmeint.nn: jmeint_train jmeint.training.fann
	./$^

.PHONY: recall
NNFILE ?= jmeint.nn
TESTDATA ?= jmeint.testing.fann
recall: jmeint_recall $(NNFILE) $(TESTDATA)
	./$^

.PHONY: weka wekagui
WEKAARGS := -H 32,8 -M 0.4 -N 1000
weka: jmeint.training.arff jmeint.testing.arff
	java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron \
		-t jmeint.training.arff -T jmeint.testing.arff \
		$(WEKAARGS)
wekagui: WEKAARGS += -G
wekagui: weka

.PHONY: clean
clean:
	rm -f jmeint_train.o jmeint.o jmeint_recall.o tritri.o jmeint.all.data jmeint.training.fann jmeint.testing.fann jmeint.training.arff jmeint.testing.arff jmeint_train jmeint_recall jmeint jmeint.nn
