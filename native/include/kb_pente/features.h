#pragma once

#include "kb_pente/position.h"

namespace kb_pente {

// Write four NCHW planes into caller-owned storage. Each plane has
// position.action_count() contiguous float32 entries, so the required output
// size is 4 * position.action_count().
void write_features(const Position& position, float* output);

}  // namespace kb_pente
