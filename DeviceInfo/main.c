// https://www.youtube.com/watch?v=dsZv82qfWRs
// https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clGetDeviceInfo.html

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 100
#include <CL/opencl.h>

int main() {
  cl_int err = 0;

  cl_device_id cpu = NULL, device = NULL;
  size_t returned_size = 0;

  // Find the CPU CL device as a fallback
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);
  // assert(err == CL_SUCCESS);
  // assert(cpu);

  // Find the GPU CL device. On failure fall back to CPU
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  assert(err == CL_SUCCESS);
  assert(device);

  // Get some information about the returned device
  cl_char vendor_name[1024] = {0};
  cl_char device_name[1024] = {0};
  err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name),
                        vendor_name, &returned_size);
  assert(err == CL_SUCCESS);

  err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                        device_name, &returned_size);
  assert(err == CL_SUCCESS);

  printf("Device %s : %s\n", vendor_name, device_name);

  cl_uint compute_units;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(compute_units), &compute_units, &returned_size);
  assert(err == CL_SUCCESS);

  printf("Max compute units: %d\n", compute_units);

  cl_uint clock_frequency;
  err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                        sizeof(clock_frequency), &clock_frequency,
                        &returned_size);
  assert(err == CL_SUCCESS);

  printf("Max clock frequency: %d MHz\n", clock_frequency);

  cl_ulong global_mem_size;
  err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(global_mem_size), &global_mem_size,
                        &returned_size);
  assert(err == CL_SUCCESS);

  printf("Global mem size: %ld bytes\n", global_mem_size);

  cl_bool image_support;
  err = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support),
                        &image_support, &returned_size);
  assert(err == CL_SUCCESS);

  printf("Image support: %d\n", image_support);

  cl_char device_extensions[2048] = {0};
  err |=
      clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_extensions),
                      &device_extensions, &returned_size);
  assert(err == CL_SUCCESS);

  printf("Device extensions: %s\n", device_extensions);

  printf("Done.\n");

  return EXIT_SUCCESS;
}
