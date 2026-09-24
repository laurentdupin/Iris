#include "iris_native.h"

int main(void) {
    iris_context* context = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (iris_native_abi_version() != IRIS_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (iris_create(0, 0, &context) != IRIS_INVALID_ARGUMENT) {
        return 2;
    }
    if (iris_inferbridge_image_shape(53, 41, &width, &height) != IRIS_OK ||
        width != 344 || height != 264) {
        return 3;
    }
    if (iris_inferbridge_image_shape(0, 41, &width, &height) !=
        IRIS_INVALID_ARGUMENT) {
        return 4;
    }
    iris_destroy(context);
    return 0;
}
