// https://www.youtube.com/watch?v=yAz9Kj6zRcA
// https://www.youtube.com/watch?v=ROTE_yjRi9s
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 100
#include <CL/opencl.h>

void runCL(float* a, int n) {
  cl_program program;
  cl_kernel kernel;

  cl_command_queue cmd_queue;
  cl_context context;

  cl_device_id cpu = NULL, device = NULL;

  cl_int err = 0;
  size_t returned_size = 0;
  size_t buffer_size;

  cl_mem buffer;

  {
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
    err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                           device_name, &returned_size);
    assert(err == CL_SUCCESS);

    printf("Connecting to %s : %s ...\n", vendor_name, device_name);
  }

  {
    // Now create a context to perform our calculation with the specified device
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    assert(err == CL_SUCCESS);

    // Add a command queue to the context
    cmd_queue = clCreateCommandQueue(context, device, 0, &err);
    assert(err == CL_SUCCESS);
  }

  {
    // Load the program source from disk
    const char* filename = "example.cl";
    char* program_source = {
        "kernel void calcSin(global float* data) {\n"
        "  int id = get_global_id(0);\n"
        "  data[id] = sin(data[id]);\n"
        "}\n"};

    program = clCreateProgramWithSource(
        context, 1, (const char**)&program_source, NULL, &err);
    assert(err == CL_SUCCESS);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    assert(err == CL_SUCCESS);

    // Now create the kernel objects to be used in example file
    kernel = clCreateKernel(program, "calcSin", &err);
  }

  {
    // Allocate memory on the device for data and results
    buffer_size = sizeof(float) * n;

    // Create memory object
    buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, buffer, CL_TRUE, 0, buffer_size,
                               (void*)a, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    // Get all of the stuff written and allocated
    clFinish(cmd_queue);
  }

  {
    // Setup the arguments to our kernel
    err = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
    assert(err == CL_SUCCESS);
  }

  {
    // Run the calculation by enqueuing it and forcing
    // the command queue to complete the task
    size_t global_dimensions[] = {n, 0, 0};
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, global_dimensions,
                                 NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    clFinish(cmd_queue);

    // Once finished read back the results from the answer array
    // to the results array
    err = clEnqueueReadBuffer(cmd_queue, buffer, CL_TRUE, 0, buffer_size, a, 0,
                              NULL, NULL);
    assert(err == CL_SUCCESS);
    clFinish(cmd_queue);
  }

  {
    // Teardown
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);
  }
}

int main() {
  // Problem size
  int n = 32;

  // Allocate some memory and a place for the results
  float* a = (float*)malloc(n * sizeof(float));

  // Fill in the values
  for (int i = 0; i < n; i++) {
    a[i] = (float)i;
  }

  // Do the OpenCL calculation
  runCL(a, n);

  // Print out some results. The values of the elements should be n
  for (int i = 0; i < n; i++) {
    printf("%f\n", a[i]);
  }

  // Free up memory
  free(a);

  return EXIT_SUCCESS;
}
