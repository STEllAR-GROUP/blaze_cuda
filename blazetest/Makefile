.POSIX:
.SUFFIXES: .h .cu .o .d .test .run

include config.mk

# Sources
CUSRC = $(shell find src -name "*.cu")

# Objects
TESTS = $(CUSRC:.cu=.test)
RUNS	= $(TESTS:.test=.run)

# Dependency files
DEPS  = $(OBJ:.o=.d)

# Compilation
.cu.test:
	$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<

# Run
.test.run:
	$<

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(NAME)

clean:
	rm -f $(TESTS)

run: $(RUNS)

.PHONY: all clean run
