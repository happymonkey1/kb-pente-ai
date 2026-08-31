#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kb_pente/mcts/search_session.h"

namespace {

class TestFailure final : public std::runtime_error {
public:
    explicit TestFailure(const std::string& message)
        : std::runtime_error(message) {}
};

void expect(bool condition, const char* message) {
    if (!condition) {
        throw TestFailure(message);
    }
}

template <typename Exception, typename Function>
void expect_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw TestFailure(message);
    }

    throw TestFailure(message);
}

void expect_close(float left, float right, float tolerance, const char* message) {
    if (std::fabs(left - right) > tolerance) {
        throw TestFailure(message);
    }
}

kb_pente::Action action_at(
    std::uint8_t board_size,
    std::uint8_t row,
    std::uint8_t column) {
    return static_cast<kb_pente::Action>(row * board_size + column);
}

kb_pente::SearchConfig tree_config(
    std::uint32_t simulation_budget,
    std::uint64_t seed = 0U) {
    return kb_pente::SearchConfig(
        1.5F, simulation_budget, 0.25F, 0.03F, seed);
}

std::array<float, kb_pente::kMaxActions> uniform_policy() {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy.fill(1.0F);
    return policy;
}

std::array<float, kb_pente::kMaxActions> policy_with(
    kb_pente::Action action,
    float probability) {
    std::array<float, kb_pente::kMaxActions> policy{};
    policy[action] = probability;
    return policy;
}

kb_pente::Position make_draw_root() {
    constexpr std::array<std::array<std::int8_t, 5>, 5> pattern{{
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, 1, -1, -1, 1}},
        {{-1, -1, 1, 1, -1}},
        {{1, -1, 1, -1, 1}},
    }};

    kb_pente::Position position = kb_pente::Position::initial(5);
    for (std::uint8_t row = 0; row < 5; ++row) {
        for (std::uint8_t column = 0; column < 5; ++column) {
            position.stones[action_at(5, row, column)] = pattern[row][column];
        }
    }
    position.stones[24] = 0;
    position.ply = 24;
    position.current_player = kb_pente::Player::One;
    position.last_action = kb_pente::kInvalidAction;
    position.validate();
    expect(!kb_pente::check_terminal(position).is_terminal(),
           "draw fixture starts in progress");
    return position;
}

std::size_t run_with_uniform_evaluator(kb_pente::SearchSession& session) {
    const auto policy = uniform_policy();
    std::size_t evaluator_requests = 0;
    while (true) {
        const auto leaf = session.select_evaluation_leaf();
        if (!leaf.has_value()) {
            break;
        }
        ++evaluator_requests;
        expect(!session.tree().leaf_terminal(*leaf).is_terminal(),
               "session returns only nonterminal evaluator leaves");
        const std::size_t action_count =
            session.tree().leaf_position(*leaf).action_count();
        session.accept_evaluation(
            *leaf, policy.data(), action_count, 0.0F);
    }
    return evaluator_requests;
}

void test_configuration_and_ownership() {
    expect_throws<std::invalid_argument>(
        [] { (void)kb_pente::SearchConfig(1.5F, 0U); },
        "zero simulation budget is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchConfig(
                1.5F, 1U, std::numeric_limits<float>::quiet_NaN());
        },
        "NaN root noise epsilon is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchConfig(1.5F, 1U, 2.0F);
        },
        "out-of-range root noise epsilon is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchConfig(
                1.5F, 1U, 0.25F,
                std::numeric_limits<float>::infinity());
        },
        "infinite Dirichlet alpha is rejected");

    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchSessionConfig(-1.0F);
        },
        "negative temperature is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchSessionConfig(
                std::numeric_limits<float>::quiet_NaN());
        },
        "NaN temperature is rejected");
    expect_throws<std::invalid_argument>(
        [] {
            (void)kb_pente::SearchSessionConfig(
                std::numeric_limits<float>::infinity());
        },
        "infinite temperature is rejected");

    kb_pente::SearchConfig mutated_tree_config;
    mutated_tree_config.simulation_budget = 0U;
    expect_throws<std::invalid_argument>(
        [&mutated_tree_config] {
            kb_pente::Tree tree(
                kb_pente::Position::initial(5),
                kb_pente::Ruleset::Freestyle,
                mutated_tree_config);
        },
        "Tree validates a mutated simulation budget");

    kb_pente::SearchSessionConfig mutated_session_config;
    mutated_session_config.temperature =
        std::numeric_limits<float>::quiet_NaN();
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    expect_throws<std::invalid_argument>(
        [&tree, &mutated_session_config] {
            kb_pente::SearchSession session(tree, mutated_session_config);
        },
        "session validates a mutated temperature");

    {
        kb_pente::SearchSession first(tree);
        expect(!first.root_priors_initialized(),
               "root priors stay uninitialized before root expansion");
        expect_throws<std::logic_error>(
            [&first] { (void)first.root_search_priors(); },
            "root priors cannot be inspected before root expansion");
        expect_throws<std::logic_error>(
            [&tree] {
                kb_pente::SearchSession second(tree);
            },
            "a Tree rejects a second live session");
        expect_throws<std::logic_error>(
            [&tree] { (void)tree.select_leaf(); },
            "direct Tree selection is rejected while a session is live");
        const auto policy = uniform_policy();
        expect_throws<std::logic_error>(
            [&tree, &policy] {
                tree.accept_evaluation(
                    tree.root_id(), policy.data(), 25U, 0.0F);
            },
            "direct Tree evaluation is rejected while a session is live");
        expect_throws<std::logic_error>(
            [&tree] { tree.resolve_terminal(tree.root_id()); },
            "direct Tree terminal resolution is rejected while a session is live");
    }

    const auto released_leaf = tree.select_leaf();
    expect(released_leaf == tree.root_id(),
           "Tree accepts direct selection after session destruction");
    const auto released_policy = uniform_policy();
    tree.accept_evaluation(
        released_leaf, released_policy.data(), 25U, 0.0F);

    kb_pente::Tree pending_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    (void)pending_tree.select_leaf();
    expect_throws<std::logic_error>(
        [&pending_tree] {
            kb_pente::SearchSession session(pending_tree);
        },
        "a session rejects a Tree with a pending evaluation");

    kb_pente::Tree movable_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    kb_pente::Tree moved_tree(std::move(movable_tree));
    expect(moved_tree.root_position() == kb_pente::Position::initial(5),
           "session-free Trees remain movable");

    kb_pente::Tree live_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    {
        kb_pente::SearchSession live_session(live_tree);
        expect_throws<std::logic_error>(
            [&live_tree] {
                kb_pente::Tree moved_live_tree(std::move(live_tree));
            },
            "moving a Tree with a live session is rejected");
    }
}

void test_exact_completion_and_terminal_progress() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(6U));
    kb_pente::SearchSession session(tree);
    expect(!session.complete(), "new session is incomplete");
    expect(session.completed_simulations() == 0U,
           "new session has no completed simulations");
    expect(run_with_uniform_evaluator(session) == 6U,
           "ordinary search evaluates each nonterminal simulation");
    expect(session.complete(), "session reaches its exact budget");
    expect(session.completed_simulations() == 6U,
           "completed simulations equal the configured budget");
    expect(!session.has_pending_evaluation(),
           "completed session has no pending evaluator request");
    expect(!session.select_evaluation_leaf().has_value(),
           "completed session returns no further leaf");

    kb_pente::Tree terminal_tree(
        make_draw_root(), kb_pente::Ruleset::Freestyle, tree_config(4U));
    kb_pente::SearchSession terminal_session(terminal_tree);
    const auto root_leaf = terminal_session.select_evaluation_leaf();
    expect(root_leaf.has_value() && *root_leaf == terminal_tree.root_id(),
           "terminal-only fixture first requests root evaluation");
    const auto final_action = policy_with(24U, 1.0F);
    terminal_session.accept_evaluation(
        *root_leaf, final_action.data(), 25U, 0.0F);
    expect(terminal_session.completed_simulations() == 1U,
           "root evaluation counts exactly one simulation");

    expect(!terminal_session.select_evaluation_leaf().has_value(),
           "terminal leaves are resolved internally through the budget");
    expect(terminal_session.completed_simulations() == 4U,
           "terminal-only progress reaches the exact budget");
    expect(!terminal_session.has_pending_evaluation(),
           "terminal-only progress leaves no pending evaluation");
    const auto root = terminal_tree.root_id();
    const auto child = terminal_tree.arena().child(root, 24U);
    expect(child != kb_pente::kInvalidNode,
           "terminal-only search creates the selected child once");
    expect(terminal_tree.arena().node(child).terminal.status ==
               kb_pente::GameStatus::Draw,
           "terminal-only child is the expected draw");
    expect(terminal_tree.arena().visit_count(root, 24U) == 3U,
           "each internally resolved terminal path increments its edge");
}

struct NoiseObservation final {
    std::array<float, kb_pente::kMaxActions> base{};
    std::array<float, kb_pente::kMaxActions> search{};
};

NoiseObservation observe_noise(std::uint64_t seed) {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U, seed));
    kb_pente::SearchSessionConfig session_config(1.0F, true);
    kb_pente::SearchSession session(tree, session_config);
    const auto leaf = session.select_evaluation_leaf();
    expect(leaf.has_value() && *leaf == tree.root_id(),
           "noise observation starts with root evaluation");
    const auto policy = uniform_policy();
    session.accept_evaluation(
        *leaf, policy.data(), tree.root_position().action_count(), 0.0F);
    expect(session.root_priors_initialized(),
           "root priors initialize after root expansion");
    expect(session.root_noise_initialized(),
           "noise session reports initialized root noise");

    NoiseObservation observation{};
    observation.search = session.root_search_priors();
    const auto row = tree.arena().edge_row(tree.root_id());
    for (std::size_t index = 0; index < tree.root_position().action_count();
         ++index) {
        observation.base[index] = row.prior(static_cast<kb_pente::Action>(index));
    }
    return observation;
}

void test_seeded_noise_and_base_immutability() {
    const NoiseObservation first = observe_noise(1234U);
    const NoiseObservation same_seed = observe_noise(1234U);
    const NoiseObservation different_seed = observe_noise(5678U);
    expect(first.search == same_seed.search,
           "same-seed trees reproduce root search priors");

    bool differs = false;
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        if (first.search[index] != different_seed.search[index]) {
            differs = true;
            break;
        }
    }
    expect(differs, "different seeds diversify root search priors");

    const float expected_base = 1.0F / 25.0F;
    float search_mass = 0.0F;
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        expect_close(
            first.base[index], index < 25U ? expected_base : 0.0F, 1.0e-7F,
            "root arena base priors remain unchanged by noise");
        search_mass += first.search[index];
        if (index >= 25U) {
            expect(first.search[index] == 0.0F,
                   "noise does not populate root padding actions");
        }
    }
    expect_close(search_mass, 1.0F, 1.0e-6F,
                 "root search priors remain normalized");

    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U, 999U));
    std::array<float, kb_pente::kMaxActions> first_stream{};
    std::array<float, kb_pente::kMaxActions> second_stream{};
    {
        kb_pente::SearchSession session(
            tree, kb_pente::SearchSessionConfig(1.0F, true));
        const auto leaf = session.select_evaluation_leaf();
        const auto policy = uniform_policy();
        session.accept_evaluation(
            *leaf, policy.data(), tree.root_position().action_count(), 0.0F);
        first_stream = session.root_search_priors();
    }
    {
        kb_pente::SearchSession session(
            tree, kb_pente::SearchSessionConfig(1.0F, true));
        second_stream = session.root_search_priors();
    }
    expect(first_stream != second_stream,
           "successive sessions advance the Tree-owned RNG stream");
}

struct TemperatureObservation final {
    std::array<std::uint32_t, kb_pente::kMaxActions> visits{};
    std::array<float, kb_pente::kMaxActions> policy{};
};

TemperatureObservation observe_temperature(float temperature) {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(4U));
    kb_pente::SearchSession session(
        tree, kb_pente::SearchSessionConfig(temperature, false));
    (void)run_with_uniform_evaluator(session);

    TemperatureObservation observation{};
    const auto row = tree.arena().edge_row(tree.root_id());
    for (std::size_t index = 0; index < kb_pente::kMaxActions; ++index) {
        observation.visits[index] =
            row.visit_count(static_cast<kb_pente::Action>(index));
    }
    observation.policy = session.root_policy();
    return observation;
}

void test_temperature_and_fallbacks() {
    const TemperatureObservation zero = observe_temperature(0.0F);
    const TemperatureObservation positive = observe_temperature(1.0F);
    std::uint32_t maximum_visits = 0U;
    kb_pente::Action first_max_action = kb_pente::kInvalidAction;
    std::uint32_t total_visits = 0U;
    for (std::size_t index = 0; index < 25U; ++index) {
        const auto action = static_cast<kb_pente::Action>(index);
        total_visits += positive.visits[index];
        if (positive.visits[index] > maximum_visits) {
            maximum_visits = positive.visits[index];
            first_max_action = action;
        }
    }
    expect(maximum_visits > 0U, "temperature fixture visits a root action");
    expect(zero.policy[first_max_action] == 1.0F,
           "temperature zero chooses the first maximum action");
    for (std::size_t index = 0; index < 25U; ++index) {
        const float expected = total_visits == 0U
                                   ? 0.0F
                                   : static_cast<float>(positive.visits[index]) /
                                         static_cast<float>(total_visits);
        expect_close(
            positive.policy[index], expected, 1.0e-6F,
            "temperature one normalizes root visit counts");
    }
    for (std::size_t index = 25U; index < kb_pente::kMaxActions; ++index) {
        expect(zero.policy[index] == 0.0F && positive.policy[index] == 0.0F,
               "root policy keeps padding actions zero");
    }

    const TemperatureObservation tiny =
        observe_temperature(std::numeric_limits<float>::denorm_min());
    std::size_t maximum_count = 0U;
    for (std::size_t index = 0; index < 25U; ++index) {
        if (tiny.visits[index] == maximum_visits) {
            ++maximum_count;
        }
    }
    expect(maximum_count > 0U,
           "tiny positive temperature sees a maximum root visit");
    for (std::size_t index = 0; index < 25U; ++index) {
        if (tiny.visits[index] == maximum_visits) {
            expect_close(
                tiny.policy[index], 1.0F / static_cast<float>(maximum_count),
                1.0e-6F,
                "tiny positive temperature shares tied maximum visits");
        } else {
            expect(tiny.policy[index] == 0.0F,
                   "tiny positive temperature suppresses lower visits");
        }
    }

    kb_pente::Tree fallback_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    kb_pente::SearchSession fallback_session(
        fallback_tree, kb_pente::SearchSessionConfig(1.0F, false));
    const auto fallback_leaf = fallback_session.select_evaluation_leaf();
    const auto sparse_policy = policy_with(7U, 1.0F);
    fallback_session.accept_evaluation(
        *fallback_leaf, sparse_policy.data(), 25U, 0.0F);
    const auto sparse_root_policy = fallback_session.root_policy();
    expect(sparse_root_policy[7U] == 1.0F,
           "zero-visit fallback uses normalized legal base priors");
    expect(fallback_session.zero_visit_fallbacks() == 1U,
           "zero-visit fallback increments its counter");

    kb_pente::Tree uniform_fallback_tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(1U));
    kb_pente::SearchSession uniform_fallback_session(uniform_fallback_tree);
    const auto uniform_leaf = uniform_fallback_session.select_evaluation_leaf();
    std::array<float, kb_pente::kMaxActions> zero_policy{};
    uniform_fallback_session.accept_evaluation(
        *uniform_leaf, zero_policy.data(), 25U, 0.0F);
    const auto uniform_root_policy = uniform_fallback_session.root_policy();
    expect(uniform_fallback_tree.invalid_policy_fallbacks() == 1U,
           "zero evaluator policy uses the core uniform fallback");
    expect(uniform_fallback_session.zero_visit_fallbacks() == 1U,
           "uniform zero-visit fallback increments its counter");
    for (std::size_t index = 0; index < 25U; ++index) {
        expect_close(
            uniform_root_policy[index], 1.0F / 25.0F, 1.0e-7F,
            "uniform fallback stays legal and normalized");
    }
}

void test_invalid_lifecycle_and_retry() {
    kb_pente::Tree tree(
        kb_pente::Position::initial(5), kb_pente::Ruleset::Freestyle,
        tree_config(3U));
    kb_pente::SearchSession session(tree);
    const auto policy = uniform_policy();
    const auto root = tree.root_id();

    expect_throws<std::logic_error>(
        [&session, root, &policy] {
            session.accept_evaluation(root, policy.data(), 25U, 0.0F);
        },
        "accepting without selection is rejected");
    expect_throws<std::logic_error>(
        [&session] { (void)session.root_policy(); },
        "root policy before completion is rejected");

    const auto leaf = session.select_evaluation_leaf();
    expect(leaf.has_value() && *leaf == root,
           "invalid lifecycle fixture selects root");
    expect_throws<std::logic_error>(
        [&session] { (void)session.select_evaluation_leaf(); },
        "selecting twice is rejected");
    expect(session.has_pending_evaluation() &&
               session.completed_simulations() == 0U,
           "selection retry state remains unchanged");
    expect_throws<std::logic_error>(
        [&session, &policy] {
            session.accept_evaluation(
                kb_pente::kInvalidNode, policy.data(), 25U, 0.0F);
        },
        "wrong pending leaf is rejected");
    expect_throws<std::invalid_argument>(
        [&session, leaf, &policy] {
            session.accept_evaluation(*leaf, policy.data(), 24U, 0.0F);
        },
        "wrong evaluator policy length is rejected");

    auto invalid_policy = policy;
    invalid_policy[0] = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>(
        [&session, leaf, &invalid_policy] {
            session.accept_evaluation(*leaf, invalid_policy.data(), 25U, 0.0F);
        },
        "NaN evaluator policy is rejected");
    invalid_policy = policy;
    invalid_policy[0] = -1.0F;
    expect_throws<std::invalid_argument>(
        [&session, leaf, &invalid_policy] {
            session.accept_evaluation(*leaf, invalid_policy.data(), 25U, 0.0F);
        },
        "negative evaluator policy is rejected");
    expect_throws<std::invalid_argument>(
        [&session, leaf, &policy] {
            session.accept_evaluation(
                *leaf, policy.data(), 25U,
                std::numeric_limits<float>::infinity());
        },
        "nonfinite evaluator value is rejected");
    expect_throws<std::invalid_argument>(
        [&session, leaf, &policy] {
            session.accept_evaluation(*leaf, policy.data(), 25U, 2.0F);
        },
        "out-of-range evaluator value is rejected");

    expect(session.has_pending_evaluation() &&
               session.completed_simulations() == 0U &&
               !tree.arena().node(root).expanded,
           "invalid evaluator calls preserve pending state and tree data");
    session.accept_evaluation(*leaf, policy.data(), 25U, 0.0F);
    expect(session.completed_simulations() == 1U,
           "valid retry increments the session exactly once");
    expect(session.root_priors_initialized(),
           "root priors initialize after successful retry");
    expect_throws<std::logic_error>(
        [&session, &policy] {
            session.accept_evaluation(
                kb_pente::kInvalidNode, policy.data(), 25U, 0.0F);
        },
        "duplicate evaluation without a pending leaf is rejected");
    expect_throws<std::logic_error>(
        [&session] { (void)session.root_policy(); },
        "root policy remains unavailable before completion");

    expect(run_with_uniform_evaluator(session) == 2U,
           "remaining session simulations can be evaluated");
    expect(session.complete() && session.completed_simulations() == 3U,
           "retryable session completes its remaining budget");
    expect_throws<std::logic_error>(
        [&session, &policy] {
            session.accept_evaluation(
                kb_pente::kInvalidNode, policy.data(), 25U, 0.0F);
        },
        "evaluation after completion is rejected");
}

}  // namespace

int main() {
    try {
        test_configuration_and_ownership();
        test_exact_completion_and_terminal_progress();
        test_seeded_noise_and_base_immutability();
        test_temperature_and_fallbacks();
        test_invalid_lifecycle_and_retry();
    } catch (const std::exception& error) {
        std::cerr << "search session test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "search session tests passed\n";
    return EXIT_SUCCESS;
}
