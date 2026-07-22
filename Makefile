####
# Makefile rules:
# Indentation for real build commands should be TABS (not spaces)
# but, directives / conditionals (ifeq, ifneq, else, endif, include, $(info ...), $(warning ...) etc.)
# should NOT be indented with tabs, but with spaces
####

.DEFAULT_GOAL := all

# Compiler
CC  := /opt/homebrew/opt/llvm/bin/clang
CXX := /opt/homebrew/opt/llvm/bin/clang++

# Build mode: release (default) or debug with make MODE=debug
MODE ?= release

# CPU tuning for Apple Silicon
# This can be overridden e.g. CPU_FLAGS="-march=native" or leave empty
CPU_FLAGS ?= -mcpu=apple-m2 -mtune=apple-m2

ifeq ($(MODE),release)
  OPT_FLAGS := -O3 -DNDEBUG
  DBG_FLAGS :=
  FP_FLAGS  :=                                # omit frame pointer (default)
  LTO_FLAGS := -flto
else
  OPT_FLAGS := -O0
  DBG_FLAGS := -g
  FP_FLAGS  := -fno-omit-frame-pointer        # easier backtraces in debug
  LTO_FLAGS :=                                # avoid slow LTO in debug
endif

CFLAGS   := -Wall $(DBG_FLAGS) $(OPT_FLAGS) $(CPU_FLAGS) $(LTO_FLAGS) -fno-sanitize=address $(FP_FLAGS)
CXXFLAGS := -Wall $(DBG_FLAGS) $(OPT_FLAGS) $(CPU_FLAGS) $(LTO_FLAGS) -fno-sanitize=address $(FP_FLAGS)
LDFLAGS  := $(LTO_FLAGS)

# Optional sanitizers: enable with `make SAN=1`
SAN ?= 0
ifeq ($(SAN),1)
  CFLAGS   += -fsanitize=address -fno-omit-frame-pointer -O1
  CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1
  LDFLAGS  += -fsanitize=address
endif

# Sources and target
SRC  := $(wildcard *.c)
OBJ  := $(SRC:.c=.o)
BIN  := ccubes

# Try OpenMP first, then pthreads, then serial fallback.
# Disable OpenMP with DISABLE_OMP=1; disable pthread fallback with USE_PTHREAD=0.
DISABLE_OMP ?= 0
USE_PTHREAD ?= 1
ifneq ($(shell test $(DISABLE_OMP) -eq 1; echo $$?),0)
  ifneq ($(shell $(CC) -fopenmp -dM -E - < /dev/null 2>/dev/null | grep _OPENMP),)
    CFLAGS   += -DHAVE_OPENMP -fopenmp
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
    $(info OpenMP found -> enabling)
  else
    ifeq ($(USE_PTHREAD),1)
      CFLAGS  += -DHAVE_PTHREAD -pthread
      CXXFLAGS += -pthread
      LDFLAGS += -pthread
      $(info OpenMP not found -> enabling pthread fallback)
    else
      $(info OpenMP not found and pthread disabled -> serial fallback)
    endif
  endif
else
  ifeq ($(USE_PTHREAD),1)
    CFLAGS  += -DHAVE_PTHREAD -pthread
    CXXFLAGS += -pthread
    LDFLAGS += -pthread
    $(info OpenMP disabled -> enabling pthread fallback)
  else
    $(info OpenMP and pthread disabled -> serial fallback)
  endif
endif

# Try Gurobi. Override GUROBI_HOME/GUROBI_LIC/GUROBI_LIBNAME when needed.
GUROBI_HOME ?= $(shell ls -d /Library/gurobi*/macos_universal2 2>/dev/null | sort | tail -n 1)
GUROBI_LIC  ?= $(if $(GRB_LICENSE_FILE),$(GRB_LICENSE_FILE),$(HOME)/gurobi.lic)
GUROBI_LIB  ?= $(shell if [ -n "$(GUROBI_HOME)" ] && [ -d "$(GUROBI_HOME)/lib" ]; then ls "$(GUROBI_HOME)"/lib/libgurobi[0-9]*.dylib 2>/dev/null | grep -E '/libgurobi[0-9]+\.dylib$$' | head -n 1; fi)
GUROBI_LIBNAME ?= $(patsubst lib%.dylib,%,$(notdir $(GUROBI_LIB)))

ifneq ("$(wildcard $(GUROBI_HOME))","")
  ifneq ("$(wildcard $(GUROBI_LIC))","")
    ifneq ($(GUROBI_LIBNAME),)
      CFLAGS   += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
      CXXFLAGS += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
      LDFLAGS  += -L$(GUROBI_HOME)/lib -l$(GUROBI_LIBNAME)
      $(info Gurobi found at $(GUROBI_HOME), library $(GUROBI_LIBNAME), license $(GUROBI_LIC) -> enabling)
    else
      $(warning Gurobi found at $(GUROBI_HOME) but no libgurobi*.dylib was detected -> disabling)
    endif
  else
    $(warning Gurobi found at $(GUROBI_HOME) but no license at $(GUROBI_LIC) -> disabling)
  endif
else
  $(info Gurobi not found -> disabling)
endif

# Gurobi license check target and conditional dependency
.PHONY: check-gurobi
check-gurobi:
	@env GRB_LICENSE_FILE=$(GUROBI_LIC) $(GUROBI_HOME)/bin/grbprobe >/dev/null 2>&1 || { echo "Gurobi license invalid or expired"; exit 1; }

# If we built with -DHAVE_GUROBI in CFLAGS, ensure we verify license at build time
ifeq ($(filter -DHAVE_GUROBI,$(CFLAGS)),-DHAVE_GUROBI)
  $(BIN): check-gurobi
endif

# Build rules: indent using TABS
all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: test-pool-selection
test-pool-selection:
	$(CC) -Wall -O2 -I. tests/test_pool_selection.c pool_selection.c -o /tmp/ccubes_test_pool_selection
	/tmp/ccubes_test_pool_selection
