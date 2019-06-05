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
	$(CXX) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<
.cu.bench:
	$(CU) $(CXXFLAGS) $(CUFLAGS) $(LDFLAGS) -MMD -o $@ $<

# Running...
.bench.score:
	@echo
	@echo '  Running $<'
	./$< | tee $@

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(BENCH)

clean:
	rm -f $(SCORE) $(BENCH) $(DEPS)

run: $(SCORE)

#remote: $(NAME)
#	ssh $(REMOTE_EXEC_HOST) mkdir -p $(REMOTE_EXEC_PATH)
#	scp $(NAME) $(REMOTE_EXEC_HOST):$(REMOTE_EXEC_PATH)/$(NAME)
#	ssh $(REMOTE_EXEC_HOST) $(REMOTE_EXEC_PATH)/$(NAME)

.PHONY: all clean remote run dump
