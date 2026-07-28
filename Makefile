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
SRC  := $(wildcard *.c)
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

$(BIN): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(DEP) $(BIN)

-include $(DEP)

# Unit tests link only the module under test, so they stay fast and do not
# need the Gurobi or OpenMP configuration above.
TEST_CFLAGS := -Wall -O2 -I.
TEST_LIBS   := -lm

.PHONY: test test-cover-validation test-pool-selection test-pool-policy test-pi-finalization test-projection-cache test-wildcard-off-masks test-nchoosek test-certified-stop test-projected-cube-prime test-subsumption-index test-plateau-probe test-bounded-mmcs test-mmcs-generator test-effort-policy
test: test-cover-validation test-pool-selection test-pool-policy test-pi-finalization test-projection-cache test-wildcard-off-masks test-nchoosek test-certified-stop test-projected-cube-prime test-subsumption-index test-plateau-probe test-bounded-mmcs test-mmcs-generator test-effort-policy

test-cover-validation:
	$(CC) $(TEST_CFLAGS) tests/test_cover_validation.c cover_validation.c -o /tmp/ccubes_test_cover_validation $(TEST_LIBS)
	/tmp/ccubes_test_cover_validation

test-pool-selection:
	$(CC) $(TEST_CFLAGS) tests/test_pool_selection.c pool_selection.c -o /tmp/ccubes_test_pool_selection $(TEST_LIBS)
	/tmp/ccubes_test_pool_selection

test-pool-policy: $(BIN)
	sh tests/test_pool_policy.sh ./$(BIN) examples/pool_positive.pla

test-pi-finalization:
	$(CC) $(TEST_CFLAGS) tests/test_pi_finalization.c utils.c binomial.c ccubes_threads.c checkpoint.c cover_validation.c debug.c lagrangian.c lock_stats.c pool_selection.c prime_check.c subsumption_index.c -o /tmp/ccubes_test_pi_finalization $(TEST_LIBS)
	/tmp/ccubes_test_pi_finalization

test-projection-cache:
	$(CC) $(TEST_CFLAGS) -DCCUBES_TESTING tests/test_projection_cache.c utils.c binomial.c ccubes_threads.c checkpoint.c cover_validation.c debug.c lagrangian.c lock_stats.c pool_selection.c prime_check.c subsumption_index.c -o /tmp/ccubes_test_projection_cache $(TEST_LIBS)
	/tmp/ccubes_test_projection_cache

test-wildcard-off-masks:
	$(CC) $(TEST_CFLAGS) -DCCUBES_TESTING tests/test_wildcard_off_masks.c utils.c binomial.c ccubes_threads.c checkpoint.c cover_validation.c debug.c lagrangian.c lock_stats.c pool_selection.c prime_check.c subsumption_index.c -o /tmp/ccubes_test_wildcard_off_masks $(TEST_LIBS)
	/tmp/ccubes_test_wildcard_off_masks

test-nchoosek:
	$(CC) $(TEST_CFLAGS) tests/test_nchoosek.c binomial.c -o /tmp/ccubes_test_nchoosek $(TEST_LIBS)
	/tmp/ccubes_test_nchoosek

test-projected-cube-prime:
	$(CC) $(TEST_CFLAGS) tests/test_projected_cube_prime.c prime_check.c utils.c binomial.c ccubes_threads.c checkpoint.c cover_validation.c debug.c lagrangian.c lock_stats.c pool_selection.c subsumption_index.c -o /tmp/ccubes_test_projected_cube_prime $(TEST_LIBS)
	/tmp/ccubes_test_projected_cube_prime

test-certified-stop:
	$(CC) $(TEST_CFLAGS) tests/test_certified_stop.c certified_stop.c -o /tmp/ccubes_test_certified_stop $(TEST_LIBS)
	/tmp/ccubes_test_certified_stop

test-subsumption-index:
	$(CC) $(TEST_CFLAGS) tests/test_subsumption_index.c subsumption_index.c -o /tmp/ccubes_test_subsumption_index $(TEST_LIBS)
	/tmp/ccubes_test_subsumption_index

test-plateau-probe:
	$(CC) $(TEST_CFLAGS) tests/test_plateau_probe.c plateau_probe.c -o /tmp/ccubes_test_plateau_probe $(TEST_LIBS)
	/tmp/ccubes_test_plateau_probe

test-bounded-mmcs:
	$(CC) $(TEST_CFLAGS) tests/test_bounded_mmcs.c bounded_mmcs.c -o /tmp/ccubes_test_bounded_mmcs $(TEST_LIBS)
	/tmp/ccubes_test_bounded_mmcs

test-mmcs-generator: $(BIN)
	sh tests/test_mmcs_generator.sh ./$(BIN) examples/mmcs_forced_100x1.pla examples/certified_F2.pla examples/rnd_20x10x40.pla

test-effort-policy: $(BIN)
	sh tests/test_effort_policy.sh ./$(BIN) examples/certified_F2.pla
