#ifndef EDGEAI_PASSES_H
#define EDGEAI_PASSES_H

namespace edgeai {

// Register builtin.module passes. Dialects are declared by the passes; the
// embedding driver remains responsible for external bufferization models.
void registerPreprocessPasses();

// Opt-in M4 schedule experiments: bounded tiling and explicit vectorization.
void registerScheduleExperimentPasses();

} // namespace edgeai

#endif // EDGEAI_PASSES_H
