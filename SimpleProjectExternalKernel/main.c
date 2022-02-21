// https://www.youtube.com/watch?v=yAz9Kj6zRcA
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 100
#include <CL/opencl.h>

char* load_program_source(const char* filename) {
  long int size = 0, res = 0;

  char* src = NULL;

  FILE* file = fopen(filename, "rb");

  if (!file) {
    return NULL;
  }

  if (fseek(file, 0, SEEK_END)) {
    fclose(file);
    return NULL;
  }

  size = ftell(file);
  if (0 == size) {
    fclose(file);
    return NULL;
  }

  rewind(file);

  src = (char*)calloc(size + 1, sizeof(char));
  if (!src) {
    fclose(file);
    return NULL;
  }

  res = fread(src, 1, sizeof(char) * size, file);
  if (res != sizeof(char) * size) {
    fclose(file);
    free(src);
    return NULL;
  }

  src[size] = '\0';
  fclose(file);
  return src;
}

void runCL(float* a, float* b, float* results, int n) {
  cl_program program[1];
  cl_kernel kernel[2];

  cl_command_queue cmd_queue;
  cl_context context;

  cl_device_id cpu = NULL, device = NULL;

  cl_int err = 0;
  size_t returned_size = 0;
  size_t buffer_size;

  cl_mem a_mem, b_mem, ans_mem;

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
    char* program_source = load_program_source(filename);
    program[0] = clCreateProgramWithSource(
        context, 1, (const char**)&program_source, NULL, &err);
    assert(err == CL_SUCCESS);

    err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
    assert(err == CL_SUCCESS);

    // Now create the kernel objects to be used in example file
    kernel[0] = clCreateKernel(program[0], "add", &err);

    free(program_source);
  }

  {
    // Allocate memory on the device for data and results
    buffer_size = sizeof(float) * n;

    // Input array a
    a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, a_mem, CL_TRUE, 0, buffer_size,
                               (void*)a, 0, NULL, NULL);

    // Input array b
    b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b_mem, CL_TRUE, 0, buffer_size,
                                (void*)b, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    // Results array
    ans_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);

    // Get all of the stuff written and allocated
    clFinish(cmd_queue);
  }

  {
    // Setup the arguments to our kernel
    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &a_mem);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &b_mem);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &ans_mem);
    assert(err == CL_SUCCESS);
  }

  {
    // Run the calculation by enqueuing it and forcing
    // the command queue to complete the task
    size_t global_work_size = n;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL,
                                 &global_work_size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    clFinish(cmd_queue);

    // Once finished read back the results from the answer array
    // to the results array
    err = clEnqueueReadBuffer(cmd_queue, ans_mem, CL_TRUE, 0, buffer_size,
                              results, 0, NULL, NULL);
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
  float* b = (float*)malloc(n * sizeof(float));
  float* results = (float*)malloc(n * sizeof(float));

  // Fill in the values
  for (int i = 0; i < n; i++) {
    a[i] = (float)i;
    b[i] = (float)n - i;
    results[i] = 0;
  }

  // Do the OpenCL calculation
  runCL(a, b, results, n);

  // Print out some results. The values of the elements should be n
  for (int i = 0; i < n; i++) {
    printf("%f\n", results[i]);
  }

  // Free up memory
  free(a);
  free(b);
  free(results);

  return EXIT_SUCCESS;
}
