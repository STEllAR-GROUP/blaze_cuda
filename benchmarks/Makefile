.POSIX:
.SUFFIXES: .cpp .hpp .h .cu .d .bench .score

include config.mk

# Sources
CXXSRC  = $(shell find blaze_cuda -name "*.cpp")
CUSRC   = $(shell find blaze_cuda -name "*.cu")

# Objects
CXXBENCH = $(CXXSRC:.cpp=.bench)
CUBENCH  = $(CUSRC:.cu=.bench)

BENCH    = $(CXXBENCH) $(CUBENCH)
SCORE    = $(BENCH:.bench=.score)

# Dependency files
DEPS   = $(BENCH:.bench=.d)

all: $(BENCH)

# Compilation
.cpp.bench:
	@echo '  · Compiling $@...'
	@$(CXX) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<
.cu.bench:
	@echo '  · Compiling $@...'
	@$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<

# Running...
.bench.score:
ifeq ($(REMOTE),)
	@echo '  · Running $<...'
	@echo
	@./$< | tee $@
	@echo
else
	@echo '  · Running $<...'
	@ssh $(REMOTE_EXEC_HOST) mkdir -p $(REMOTE_EXEC_PATH)
	@scp $< $(REMOTE_EXEC_HOST):$(REMOTE_EXEC_PATH)
	@ssh $(REMOTE_EXEC_HOST) $(REMOTE_EXEC_PATH)/$(notdir $<) | tee $@
	@echo
endif

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(BENCH)

clean:
	rm -f $(SCORE) $(BENCH) $(DEPS)

run: $(SCORE)

.PHONY: all clean remote run dump
