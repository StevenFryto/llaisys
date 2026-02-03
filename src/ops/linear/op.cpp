#include "op.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // TO_BE_IMPLEMENTED();
    if (bias == nullptr) {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
        ASSERT(bias->shape()[0] == weight->shape()[0], "Linear: bias.shape[0] must be equal to weight.shape[0].");
    }
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out.shape[0] must be equal to in.shape[0].");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out.shape[1] must be equal to weight.shape[0].");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in.shape[1] must be equal to weight.shape[1].");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        if (bias == nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, out->dtype(), out->shape(), in->shape(), weight->shape(), {});
        }
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), out->shape(), in->shape(), weight->shape(), bias->shape());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        if (bias == nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, out->dtype(), out->shape(), in->shape(), weight->shape(), {});
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), out->shape(), in->shape(), weight->shape(), bias->shape());
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
