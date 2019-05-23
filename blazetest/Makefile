.POSIX:
.SUFFIXES: .h .cu .o .d .test .run .remote

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

.test.remote:
	ssh $(REMOTE_EXEC_HOST) mkdir -p $(REMOTE_EXEC_PATH)
	scp $< $(REMOTE_EXEC_HOST):$(REMOTE_EXEC_PATH)
	ssh $(REMOTE_EXEC_HOST) $(REMOTE_EXEC_PATH)/$(notdir $<)
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
