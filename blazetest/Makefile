.POSIX:
.SUFFIXES: .h .cu .o .d .test .run

include config.mk

CUSRC = $(shell find src -name "*.cu")	# Sources
TESTS = $(CUSRC:.cu=.test)					# Test executables
RUNS	= $(TESTS:.test=.run)				# Runs

# Dependency files
DEPS  = $(OBJ:.o=.d)

# Compilation
.cu.test:
	mv $< $<.disable
#	@echo $<
#	$(CU) $(CXXFLAGS) $(CUFLAGS) -MMD -c -o $@ $<

# Run
.test.run:
	$<

tests: $(TESTS)

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(NAME)

clean:
	rm -f $(TESTS)

run: $(RUNS)

.PHONY: all clean run
