#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace vllm {

class ScalarType {
public:
    enum NanRepr : uint8_t {
        NAN_NONE = 0,
        NAN_IEEE_754 = 1,
        NAN_EXTD_RANGE_MAX_MIN = 2,
    };

    constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_, int32_t bias,
            bool finite_values_only = false, NanRepr nan_repr = NAN_IEEE_754)
        : exponent(exponent),
          mantissa(mantissa),
          signed_(signed_),
          bias(bias),
          finite_values_only(finite_values_only),
          nan_repr(nan_repr) {
    }

    static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
        return ScalarType(0, size_bits - 1, true, bias);
    }

    static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
        return ScalarType(0, size_bits, false, bias);
    }

    static constexpr ScalarType float_IEEE754(uint8_t exponent, uint8_t mantissa) {
        return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
    }

    static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
            bool finite_values_only, NanRepr nan_repr) {
        return ScalarType(exponent, mantissa, true, 0, finite_values_only, nan_repr);
    }

    using Id = int64_t;

    uint8_t const exponent;
    uint8_t const mantissa;
    bool const signed_;
    int32_t const bias;
    bool const finite_values_only;
    NanRepr const nan_repr;

private:
    template <typename T_>
    static constexpr size_t member_id_field_width() {
        using T = std::decay_t<T_>;
        return std::is_same_v<T, bool> ? 1 : sizeof(T) * 8;
    }

    template <typename Fn, typename Init, typename Member, typename... Rest>
    static constexpr auto reduce_members_helper(Fn f, Init val, Member member, Rest... rest) {
        auto new_val = f(val, member);
        if constexpr (sizeof...(rest) > 0) {
            return reduce_members_helper(f, new_val, rest...);
        } else {
            return new_val;
        }
    }

    template <typename Fn, typename Init>
    constexpr auto reduce_members(Fn f, Init init) const {
        return reduce_members_helper(f, init, exponent, mantissa, signed_, bias,
                finite_values_only, nan_repr);
    }

    template <typename Fn, typename Init>
    static constexpr auto reduce_member_types(Fn f, Init init) {
        constexpr auto dummy_type = ScalarType(0, 0, false, 0, false, NAN_NONE);
        return dummy_type.reduce_members(f, init);
    }

    static constexpr auto id_size_bits() {
        return reduce_member_types(
                [](int acc, auto member) -> int {
                    return acc + member_id_field_width<decltype(member)>();
                },
                0);
    }

public:
    constexpr Id id() const {
        static_assert(id_size_bits() <= sizeof(Id) * 8, "ScalarType id is too large");

        auto or_and_advance = [](std::pair<Id, uint32_t> result, auto member) -> std::pair<Id, uint32_t> {
            auto [id, bit_offset] = result;
            constexpr auto bits = member_id_field_width<decltype(member)>();
            return {
                id | (int64_t(member) & ((uint64_t(1) << bits) - 1)) << bit_offset,
                bit_offset + bits,
            };
        };

        return reduce_members(or_and_advance, std::pair<Id, uint32_t>{}).first;
    }

    static constexpr ScalarType from_id(Id id) {
        auto extract_and_advance = [id](auto result, auto member) {
            using T = decltype(member);
            auto [tuple, bit_offset] = result;
            constexpr auto bits = member_id_field_width<T>();
            auto extracted_val = static_cast<T>((int64_t(id) >> bit_offset) &
                    ((uint64_t(1) << bits) - 1));
            auto new_tuple = std::tuple_cat(tuple, std::make_tuple(extracted_val));
            return std::pair<decltype(new_tuple), int>{new_tuple, bit_offset + bits};
        };

        auto [tuple_args, _] = reduce_member_types(extract_and_advance,
                std::pair<std::tuple<>, int>{});
        return std::apply([](auto... args) { return ScalarType(args...); }, tuple_args);
    }

    constexpr int64_t size_bits() const {
        return mantissa + exponent + is_signed();
    }

    constexpr bool is_signed() const {
        return signed_;
    }

    constexpr bool operator==(const ScalarType & other) const {
        return mantissa == other.mantissa &&
               exponent == other.exponent &&
               bias == other.bias &&
               signed_ == other.signed_ &&
               finite_values_only == other.finite_values_only &&
               nan_repr == other.nan_repr;
    }

    constexpr bool operator!=(const ScalarType & other) const {
        return !(*this == other);
    }
};

using ScalarTypeId = ScalarType::Id;

static inline constexpr auto kS4      = ScalarType::int_(4);
static inline constexpr auto kU4      = ScalarType::uint(4);
static inline constexpr auto kU4B8    = ScalarType::uint(4, 8);
static inline constexpr auto kS8      = ScalarType::int_(8);
static inline constexpr auto kU8      = ScalarType::uint(8);
static inline constexpr auto kU8B128  = ScalarType::uint(8, 128);
static inline constexpr auto kFE2M1f  = ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn = ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu = ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M7   = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10  = ScalarType::float_IEEE754(5, 10);
static inline constexpr auto kHalf     = kFE5M10;
static inline constexpr auto kFloat16  = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

} // namespace vllm
