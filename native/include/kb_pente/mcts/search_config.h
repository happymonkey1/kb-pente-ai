#pragma once

#include <cmath>
#include <stdexcept>

namespace kb_pente {

// SearchConfig contains deterministic PUCT settings owned by one Tree.
// Root noise and policy-temperature settings belong to later lifecycle layers.
struct SearchConfig final {
    explicit SearchConfig(float c_puct_value = 1.5F)
        : c_puct(c_puct_value) {
        validate();
    }

    void validate() const {
        if (!std::isfinite(c_puct) || c_puct <= 0.0F) {
            throw std::invalid_argument(
                "SearchConfig c_puct must be finite and positive");
        }
    }

    float c_puct;
};

}  // namespace kb_pente
