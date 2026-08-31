#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace kb_pente {

// SearchConfig contains search settings owned by one Tree, including the
// persistent random seed used by sessions over that tree.
struct SearchConfig final {
    explicit SearchConfig(
        float c_puct_value = 1.5F,
        std::uint32_t simulation_budget_value = 400U,
        float root_noise_epsilon_value = 0.25F,
        float root_dirichlet_alpha_value = 0.03F,
        std::uint64_t seed_value = 0U)
        : c_puct(c_puct_value),
          simulation_budget(simulation_budget_value),
          root_noise_epsilon(root_noise_epsilon_value),
          root_dirichlet_alpha(root_dirichlet_alpha_value),
          seed(seed_value) {
        validate();
    }

    void validate() const {
        if (!std::isfinite(c_puct) || c_puct <= 0.0F) {
            throw std::invalid_argument(
                "SearchConfig c_puct must be finite and positive");
        }
        if (simulation_budget == 0U) {
            throw std::invalid_argument(
                "SearchConfig simulation budget must be positive");
        }
        if (!std::isfinite(root_noise_epsilon) ||
            root_noise_epsilon < 0.0F || root_noise_epsilon > 1.0F) {
            throw std::invalid_argument(
                "SearchConfig root noise epsilon must be finite and in [0, 1]");
        }
        if (!std::isfinite(root_dirichlet_alpha) ||
            root_dirichlet_alpha <= 0.0F) {
            throw std::invalid_argument(
                "SearchConfig root Dirichlet alpha must be finite and positive");
        }
    }

    float c_puct;
    std::uint32_t simulation_budget;
    float root_noise_epsilon;
    float root_dirichlet_alpha;
    std::uint64_t seed;
};

}  // namespace kb_pente
