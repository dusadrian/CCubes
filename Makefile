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

# try OpenMP
# Allow disabling OpenMP with DISABLE_OMP=1 if needed for debugging
DISABLE_OMP ?= 0
ifneq ($(shell test $(DISABLE_OMP) -eq 1; echo $$?),0)
  ifneq ($(shell $(CC) -fopenmp -dM -E - < /dev/null 2>/dev/null | grep _OPENMP),)
    CFLAGS   += -fopenmp
    CXXFLAGS += -fopenmp
    LDFLAGS  += -fopenmp
    $(info OpenMP found -> enabling)
  else
    $(info OpenMP not found -> disabling)
  endif
else
  $(info OpenMP disabled by user -> DISABLE_OMP=$(DISABLE_OMP))
endif

# try Gurobi (hardcoded path for now)
GUROBI_HOME ?= /Library/gurobi1202/macos_universal2
GUROBI_LIC  ?= $(HOME)/gurobi.lic

ifneq ("$(wildcard $(GUROBI_HOME))","")
  ifneq ("$(wildcard $(GUROBI_LIC))","")
    CFLAGS   += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
    CXXFLAGS += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
    LDFLAGS  += -L$(GUROBI_HOME)/lib -lgurobi120
    $(info Gurobi found at $(GUROBI_HOME) and license $(GUROBI_LIC) -> enabling)
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