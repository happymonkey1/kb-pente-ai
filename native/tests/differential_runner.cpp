#include "kb_pente/game.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>

namespace {

constexpr std::size_t kHeaderSize = 8;
constexpr std::array<char, kHeaderSize> kRequestMagic{{
    'K', 'B', 'P', 'D', 'I', 'F', 'F', '3',
}};
constexpr std::array<char, kHeaderSize> kResponseMagic{{
    'K', 'B', 'P', 'R', 'E', 'S', '3', '\0',
}};

[[nodiscard]] bool read_bytes(void* destination, std::size_t size) {
    std::cin.read(static_cast<char*>(destination),
                  static_cast<std::streamsize>(size));
    return std::cin.good() &&
           static_cast<std::size_t>(std::cin.gcount()) == size;
}

[[nodiscard]] bool read_u8(std::uint8_t& value) {
    return read_bytes(&value, sizeof(value));
}

[[nodiscard]] bool read_i8(std::int8_t& value) {
    std::uint8_t raw = 0;
    if (!read_u8(raw)) {
        return false;
    }
    value = static_cast<std::int8_t>(raw);
    return true;
}

[[nodiscard]] bool read_u16(std::uint16_t& value) {
    std::array<std::uint8_t, 2> bytes{};
    if (!read_bytes(bytes.data(), bytes.size())) {
        return false;
    }
    value = static_cast<std::uint16_t>(bytes[0]) |
            (static_cast<std::uint16_t>(bytes[1]) << 8U);
    return true;
}

[[nodiscard]] bool read_u32(std::uint32_t& value) {
    std::array<std::uint8_t, 4> bytes{};
    if (!read_bytes(bytes.data(), bytes.size())) {
        return false;
    }
    value = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
            (static_cast<std::uint32_t>(bytes[3]) << 24U);
    return true;
}

void write_bytes(const void* source, std::size_t size) {
    std::cout.write(static_cast<const char*>(source),
                    static_cast<std::streamsize>(size));
}

void write_u8(std::uint8_t value) {
    write_bytes(&value, sizeof(value));
}

void write_i8(std::int8_t value) {
    write_u8(static_cast<std::uint8_t>(value));
}

void write_u16(std::uint16_t value) {
    const std::array<std::uint8_t, 2> bytes{{
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8U) & 0xffU),
    }};
    write_bytes(bytes.data(), bytes.size());
}

void write_u32(std::uint32_t value) {
    const std::array<std::uint8_t, 4> bytes{{
        static_cast<std::uint8_t>(value & 0xffU),
        static_cast<std::uint8_t>((value >> 8U) & 0xffU),
        static_cast<std::uint8_t>((value >> 16U) & 0xffU),
        static_cast<std::uint8_t>((value >> 24U) & 0xffU),
    }};
    write_bytes(bytes.data(), bytes.size());
}

void write_u64(std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] =
            static_cast<std::uint8_t>((value >> (8U * index)) & 0xffU);
    }
    write_bytes(bytes.data(), bytes.size());
}

struct WirePosition final {
    std::array<std::int8_t, kb_pente::kMaxActions> stones{};
    std::array<std::uint8_t, kb_pente::kPlayerCount> captures{};
    std::uint16_t ply = 0;
    kb_pente::Action last_action = kb_pente::kInvalidAction;
    std::uint8_t board_size = 0;
    std::int8_t current_player = 0;
};

[[nodiscard]] bool read_wire_position(WirePosition& wire) {
    if (!read_bytes(wire.stones.data(), wire.stones.size())) {
        return false;
    }
    for (auto& capture : wire.captures) {
        if (!read_u8(capture)) {
            return false;
        }
    }
    return read_u16(wire.ply) && read_u16(wire.last_action) &&
           read_u8(wire.board_size) && read_i8(wire.current_player);
}

void write_wire_position(const kb_pente::Position& position) {
    write_bytes(position.stones.data(), position.stones.size());
    for (const auto capture : position.captures) {
        write_u8(capture);
    }
    write_u16(position.ply);
    write_u16(position.last_action);
    write_u8(position.board_size);
    write_i8(static_cast<std::int8_t>(position.current_player));
}

[[nodiscard]] kb_pente::Position decode_position(const WirePosition& wire) {
    kb_pente::Position position{};
    position.stones = wire.stones;
    position.captures = wire.captures;
    position.ply = wire.ply;
    position.last_action = wire.last_action;
    position.board_size = wire.board_size;
    position.current_player =
        static_cast<kb_pente::Player>(wire.current_player);
    return position;
}

void write_terminal(const kb_pente::TerminalResult& terminal) {
    write_u8(static_cast<std::uint8_t>(terminal.status));
    write_i8(terminal.winner.has_value()
                 ? static_cast<std::int8_t>(*terminal.winner)
                 : static_cast<std::int8_t>(0));
    write_u8(static_cast<std::uint8_t>(terminal.reason));
}

void write_float(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(bits);
}

int protocol_error(const char* message) {
    std::cerr << "native differential protocol error: " << message << '\n';
    return 2;
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::array<char, kHeaderSize> request_magic{};
    if (!read_bytes(request_magic.data(), request_magic.size())) {
        return protocol_error("truncated request header");
    }
    if (request_magic != kRequestMagic) {
        return protocol_error("unexpected request magic");
    }

    std::uint32_t request_count = 0;
    if (!read_u32(request_count)) {
        return protocol_error("truncated request count");
    }

    write_bytes(kResponseMagic.data(), kResponseMagic.size());
    write_u32(request_count);

    for (std::uint32_t request_index = 0; request_index < request_count;
         ++request_index) {
        std::uint8_t ruleset_code = 0;
        WirePosition wire{};
        if (!read_u8(ruleset_code) || !read_wire_position(wire)) {
            return protocol_error("truncated request record");
        }

        try {
            const auto ruleset =
                static_cast<kb_pente::Ruleset>(ruleset_code);
            if (!kb_pente::is_valid_ruleset(ruleset)) {
                return protocol_error("unknown ruleset code");
            }

            const kb_pente::Position position = decode_position(wire);
            position.validate();
            const auto terminal = kb_pente::check_terminal(position);
            const auto legal = kb_pente::legal_action_mask(position, ruleset);
            std::array<float, 4 * static_cast<std::size_t>(kb_pente::kMaxActions)>
                features{};
            kb_pente::write_features(position, features.data());

            write_wire_position(position);
            write_terminal(terminal);
            for (const auto word : legal.words) {
                write_u64(word);
            }
            const auto area = position.action_count();
            for (std::size_t index = 0; index < 4 * area; ++index) {
                write_float(features[index]);
            }

            const auto successor_count =
                terminal.is_terminal()
                    ? static_cast<std::uint16_t>(0)
                    : static_cast<std::uint16_t>(legal.count());
            write_u16(successor_count);
            for (std::size_t action_index = 0;
                 action_index < position.action_count(); ++action_index) {
                const auto action = static_cast<kb_pente::Action>(action_index);
                if (successor_count == 0 || !legal.contains(action)) {
                    continue;
                }
                const auto transition =
                    kb_pente::apply_action(position, action, ruleset);
                write_u16(action);
                write_wire_position(transition.position);
                write_terminal(transition.terminal);
            }
        } catch (const std::exception& error) {
            std::cerr << "native differential request " << request_index
                      << " failed: " << error.what() << '\n';
            return 3;
        }
    }

    std::cout.flush();
    return std::cout.good() ? 0 : protocol_error("response write failed");
}
