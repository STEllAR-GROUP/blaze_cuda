.POSIX:
.SUFFIXES: .h .cu .o .d .test .run

include config.mk

CUSRC = $(shell find src -name "*.cu")	# Sources
DEPS  = $(TESTS:.test=.d)					# Dependency files
TESTS = $(CUSRC:.cu=.test)					# Test executables
RUNS	= $(TESTS:.test=.run)				# Runs

# Compilation
.cu.test:
	$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<

# Run
.test.run:
	./$<
	@touch $@
	@echo "$<: Passed."

tests: $(TESTS)
run: $(RUNS)

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(TESTS)

clean:
	rm -f $(TESTS) $(RUNS) $(DEPS)

.PHONY: all clean run
