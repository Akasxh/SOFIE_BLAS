// currently this contains the activvation functions:
// relu , sigmoid , tanh , GELU , ELU , Leaky relu

// currently writing tests for them, will be completed soon
// tests like we did for relu at https://gist.github.com/Akasxh/8f53220f27f23df9f3c36091654ff4ba

#include <alpaka/alpaka.hpp>
#include <cstddef>

namespace sofie_blas {

// -------- activation functions --------

// ReLU
template <typename T> struct OpRelu {
    ALPAKA_FN_HOST_ACC constexpr OpRelu() = default;
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const&, T x) const { return x > T{0} ? x : T{0}; }
};

// Sigmoid
template <typename T> struct OpSigmoid {
    ALPAKA_FN_HOST_ACC constexpr OpSigmoid() = default;
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const& acc, T x) const {
        if (x >= T{0}) {
            T e = alpaka::math::exp(acc, -x);
            return T{1} / (T{1} + e);
        } else {
            T e = alpaka::math::exp(acc,  x);
            return e / (T{1} + e);
        }
    }
};

// Tanh
template <typename T> struct OpTanh {
    ALPAKA_FN_HOST_ACC constexpr OpTanh() = default;
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const& acc, T x) const {
        return alpaka::math::tanh(acc, x);
    }
};

// Leaky ReLU
template <typename T> struct OpLeakyRelu {
    T alpha;
    ALPAKA_FN_HOST_ACC constexpr explicit OpLeakyRelu(T a=T(0.01)) : alpha(a) {}
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const&, T x) const { return x > T{0} ? x : alpha*x; }
};

// GELU (tanh approximation) : https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
template <typename T> struct OpGelu {
    ALPAKA_FN_HOST_ACC constexpr OpGelu() = default;
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const& acc, T x) const {
        const T k0 = T(0.7978845608028654); // sqrt(2/pi)
        const T k1 = T(0.044715);
        T x3 = x*x*x;
        T t  = k0*(x + k1*x3);
        T u  = alpaka::math::tanh(acc, t);
        return T(0.5)*x*(T(1)+u);
    }
};

// ELU
template <typename T> struct OpElu {
    T alpha;
    ALPAKA_FN_HOST_ACC constexpr explicit OpElu(T a=T(1)) : alpha(a) {}
    template <typename Acc>
    ALPAKA_FN_ACC T operator()(Acc const& acc, T x) const {
        return x >= T{0} ? x : alpha*(alpaka::math::exp(acc, x) - T{1});
    }
};

// -------- kernel --------
// y[i] = op(x[i]) for i in [0, N)

struct Activation_Functions {
    template <typename Acc, typename T, typename Op>
    ALPAKA_FN_ACC void operator()(Acc const& acc,
                                  T const* x,
                                  T*       y,
                                  std::size_t N,
                                  Op       op) const
    {
        for (auto i : alpaka::uniformElements(acc, N))
            y[i] = op(acc, x[i]);
    }
};

} 
