####
# Makefile rules:
# Indentation for real build commands should be TABS (not spaces)
# but, directives / conditionals (ifeq, ifneq, else, endif, include, $(info ...), $(warning ...) etc.)
# should NOT be indented with tabs, but with spaces
####

.DEFAULT_GOAL := all

# Host detection.  Everything platform specific below keys off these two.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Compiler.  An explicit CC always wins (`make CC=gcc`, or CC=gcc in the
# environment); the defaults below only apply when make supplies its own.
BREW_CLANG := /opt/homebrew/opt/llvm/bin/clang
BREW_CLANGXX := /opt/homebrew/opt/llvm/bin/clang++

ifeq ($(origin CC),default)
  ifeq ($(UNAME_S),Darwin)
    CC := $(if $(wildcard $(BREW_CLANG)),$(BREW_CLANG),clang)
  else
    CC := cc
  endif
endif

ifeq ($(origin CXX),default)
  ifeq ($(UNAME_S),Darwin)
    CXX := $(if $(wildcard $(BREW_CLANGXX)),$(BREW_CLANGXX),clang++)
  else
    CXX := c++
  endif
endif

# Build mode: release (default) or debug with make MODE=debug
MODE ?= release

# CPU tuning.  Override with e.g. CPU_FLAGS="-march=native" or CPU_FLAGS= to
# disable.  Apple Silicon keeps the original -mcpu=apple-m2 default.
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    CPU_FLAGS ?= -mcpu=apple-m2 -mtune=apple-m2
  else
    CPU_FLAGS ?= -march=native
  endif
else
  ifeq ($(UNAME_M),aarch64)
    CPU_FLAGS ?= -mcpu=native
  else
    CPU_FLAGS ?= -march=native
  endif
endif

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

# libm is part of libSystem on macOS but a separate library elsewhere.
ifneq ($(UNAME_S),Darwin)
  LDLIBS += -lm
endif

# Optional sanitizers: enable with `make SAN=1`
SAN ?= 0
ifeq ($(SAN),1)
  CFLAGS   += -fsanitize=address -fno-omit-frame-pointer -O1
  CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -O1
  LDFLAGS  += -fsanitize=address
endif

# Sources and target
# Explicit membership prevents stray experimental files entering production.
SRC := binomial.c bounded_mmcs.c ccubes_threads.c certified_stop.c \
  checkpoint.c cover_validation.c debug.c gurobi.c \
  lagrangian.c lock_stats.c main.c plateau_probe.c \
  pool_selection.c prime_check.c \
  progressive_column_pool.c progressive_cover.c progressive_dual_calibration.c \
  progressive_global_bound.c progressive_greedy_price.c progressive_input.c progressive_literal_polish.c \
  progressive_native_cover.c progressive_native_master.c progressive_native_phase_policy.c progressive_native_reduce_expand.c \
  progressive_native_repair.c progressive_rwls.c progressive_seed.c progressive_validation.c \
  subsumption_index.c utils.c cli.c \
  runtime_options.c
OBJ  := $(SRC:.c=.o)
DEP  := $(OBJ:.o=.d)
BIN  := ccubes

# Track project-header dependencies so incremental builds cannot mix object
# files compiled against different structure layouts.
CFLAGS   += -MMD -MP
CXXFLAGS += -MMD -MP

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
# Layout differs per platform: macOS ships macos_universal2/*.dylib under
# /Library, Linux ships linux64/*.so under /opt.
ifeq ($(UNAME_S),Darwin)
  GUROBI_PLATFORM := macos_universal2
  GUROBI_LIBEXT   := dylib
  GUROBI_ROOTS    := /Library/gurobi*
else
  GUROBI_PLATFORM := linux64
  GUROBI_LIBEXT   := so
  GUROBI_ROOTS    := /opt/gurobi* $(HOME)/gurobi*
endif

GUROBI_HOME ?= $(shell ls -d $(addsuffix /$(GUROBI_PLATFORM),$(GUROBI_ROOTS)) 2>/dev/null | sort | tail -n 1)
GUROBI_LIC  ?= $(if $(GRB_LICENSE_FILE),$(GRB_LICENSE_FILE),$(HOME)/gurobi.lic)
GUROBI_LIB  ?= $(shell if [ -n "$(GUROBI_HOME)" ] && [ -d "$(GUROBI_HOME)/lib" ]; then ls "$(GUROBI_HOME)"/lib/libgurobi[0-9]*.$(GUROBI_LIBEXT) 2>/dev/null | grep -E '/libgurobi[0-9]+\.$(GUROBI_LIBEXT)$$' | head -n 1; fi)
GUROBI_LIBNAME ?= $(patsubst lib%.$(GUROBI_LIBEXT),%,$(notdir $(GUROBI_LIB)))

ifneq ("$(wildcard $(GUROBI_HOME))","")
  ifneq ("$(wildcard $(GUROBI_LIC))","")
    ifneq ($(GUROBI_LIBNAME),)
      CFLAGS   += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
      CXXFLAGS += -I$(GUROBI_HOME)/include -DHAVE_GUROBI
      LDFLAGS  += -L$(GUROBI_HOME)/lib -l$(GUROBI_LIBNAME)
      # macOS resolves the dylib by install name; ELF needs the path recorded.
      ifneq ($(UNAME_S),Darwin)
        LDFLAGS += -Wl,-rpath,$(GUROBI_HOME)/lib
      endif
      $(info Gurobi found at $(GUROBI_HOME), library $(GUROBI_LIBNAME), license $(GUROBI_LIC) -> enabling)
    else
      $(warning Gurobi found at $(GUROBI_HOME) but no libgurobi*.$(GUROBI_LIBEXT) was detected -> disabling)
    endif
  else
    $(warning Gurobi found at $(GUROBI_HOME) but no license at $(GUROBI_LIC) -> disabling)
  endif
else
  $(info Gurobi not found -> disabling)
endif

$(info Building for $(UNAME_S)/$(UNAME_M) with $(CC))

# Gurobi license check target and conditional dependency
.PHONY: check-gurobi
check-gurobi:
	@env GRB_LICENSE_FILE=$(GUROBI_LIC) $(GUROBI_HOME)/bin/grbprobe >/dev/null 2>&1 || { echo "Gurobi license invalid or expired"; exit 1; }

# If we built with -DHAVE_GUROBI in CFLAGS, ensure we verify license at build time
ifeq ($(filter -DHAVE_GUROBI,$(CFLAGS)),-DHAVE_GUROBI)
  $(BIN): check-gurobi
endif

# Build rules: indent using TABS
.PHONY: all clean
all: $(BIN)

# Header dependencies alone cannot detect a changed compiler, backend or flags.
# Keep the stamp's timestamp unchanged when the effective configuration matches.
build_shell_quote = '$(subst ','"'"',$(1))'
BUILD_CONFIG_SUM := $(shell printf '%s\n' $(call build_shell_quote,CC=$(CC)) $(call build_shell_quote,CFLAGS=$(CFLAGS)) $(call build_shell_quote,LDFLAGS=$(LDFLAGS)) $(call build_shell_quote,LDLIBS=$(LDLIBS)) | cksum)
SAVED_BUILD_CONFIG_SUM := $(shell sed -n '1p' .build-config 2>/dev/null)
.PHONY: FORCE_BUILD_CONFIG
FORCE_BUILD_CONFIG:

# Older make versions compare timestamps only to whole seconds. Force a
# rebuild on a changed signature even when two builds share that second.
ifneq ($(BUILD_CONFIG_SUM),$(SAVED_BUILD_CONFIG_SUM))
$(OBJ) $(BIN): FORCE_BUILD_CONFIG
endif

.build-config: FORCE_BUILD_CONFIG
	@printf '%s\n' $(call build_shell_quote,$(BUILD_CONFIG_SUM)) $(call build_shell_quote,CC=$(CC)) $(call build_shell_quote,CFLAGS=$(CFLAGS)) $(call build_shell_quote,LDFLAGS=$(LDFLAGS)) $(call build_shell_quote,LDLIBS=$(LDLIBS)) > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(OBJ): .build-config

$(BIN): $(OBJ) .build-config
	$(CC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(DEP) $(BIN) .build-config .build-config.tmp

-include $(DEP)
