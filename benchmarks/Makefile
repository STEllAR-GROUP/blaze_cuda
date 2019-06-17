.POSIX:
.SUFFIXES: .cpp .hpp .h .cu .d .bench .score .profile

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
	$(CXX) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $< $(HPXFLAGS)
.cu.bench:
	@echo '  · Compiling $@...'
	$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $< $(HPXFLAGS)

# Running...
.bench.score:
ifeq ($(REMOTE_EXEC_HOST),)
	@echo '  · Running $<...'
	@echo
	@hpxrun.py ./$< | tee $@
	@echo
else
	@echo '  · Running $< on $(REMOTE_EXEC_HOST)...'
	@ssh $(REMOTE_EXEC_HOST) mkdir -p $(REMOTE_EXEC_PATH)
	@scp $< $(REMOTE_EXEC_HOST):$(REMOTE_EXEC_PATH)
	@ssh $(REMOTE_EXEC_HOST) hpxrun.py $(REMOTE_EXEC_PATH)/$(notdir $<) | tee $@
	@echo
endif

# Profiling...
.bench.profile:
ifeq ($(REMOTE_EXEC_HOST),)
	@echo '  · Running $<...'
	@echo
	@nvprof ./$< | tee $@
	@echo
else
	@echo '  · Running $< on $(REMOTE_EXEC_HOST)...'
	@ssh $(REMOTE_EXEC_HOST) mkdir -p $(REMOTE_EXEC_PATH)
	@scp $< $(REMOTE_EXEC_HOST):$(REMOTE_EXEC_PATH)
	@ssh $(REMOTE_EXEC_HOST) nvprof -o /tmp/nvprof_output $(REMOTE_EXEC_PATH)/$(notdir $<) | tee $@
	@scp $(REMOTE_EXEC_HOST):/tmp/nvprof_output $@
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
