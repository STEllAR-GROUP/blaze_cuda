.POSIX:
.SUFFIXES: .h .cu .o .d .test .run

include config.mk

CUSRC = $(shell find src -name "*.cu")	# Sources
TESTS = $(CUSRC:.cu=.test)					# Test executables
RUNS	= $(TESTS:.test=.run)				# Runs
LOGS	= $(TESTS:.test=.test.run)			# Logs

# Dependency files
DEPS  = $(TESTS:.test=.d)

# Compilation
.cu.test:
	$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<
#	$(CU) $(CXXFLAGS) $(CUFLAGS) -MMD -c -o $@ $<

# Run
.test.run:
	./$<
	touch $@	# Renamed if test passed

tests: $(TESTS)

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(NAME)

clean:
	rm -f $(TESTS) $(RUNS) $(LOGS) $(DEPS)

run: $(RUNS)

.PHONY: all clean run
