#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

// Debug levels (incremental thresholds)
enum {
    DBG_ERROR = 1,  // errors + warnings
    DBG_INFO  = 2,  // errors + warnings + info
    DBG_TRACE = 3   // everything
};

// Global debug configuration
extern int debug_enabled;   // 0 = off, 1 = on
extern int debug_level;     // current threshold
extern FILE *debug_out;     // output stream (stderr or file)

// Initialize debug system
//   filename = NULL → stderr
//   level    = threshold level
void debug_init(const char *filename, int level);

// Clean up (close file if needed)
void debug_close(void);

// Macro for logging
#define DBG(level, fmt, ...) \
    do { if (debug_enabled && (level) <= debug_level) { \
        fprintf(debug_out, "[%s] %s:%d: " fmt "\n", \
            (level)==DBG_ERROR ? "ERROR" : \
            (level)==DBG_INFO  ? "INFO " : "TRACE", \
            __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(debug_out); \
    } } while (0)

// Run a block only if debug is enabled and at least 'level'
#define DBG_BLOCK(level) if (debug_enabled && (level) <= debug_level)

// Convenience wrappers for common levels
#define DBG_ERROR_BLOCK   DBG_BLOCK(DBG_ERROR)
#define DBG_INFO_BLOCK    DBG_BLOCK(DBG_INFO)
#define DBG_TRACE_BLOCK   DBG_BLOCK(DBG_TRACE)

#endif // DEBUG_H
