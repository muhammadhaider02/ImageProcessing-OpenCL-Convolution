#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <string>

const int SIZE = 2048;
const int FILTER_SIZE = 3;
const int HALO = 1;

// Gaussian blur 3x3 kernel
const float filter[FILTER_SIZE][FILTER_SIZE] = {
    {0.0625f, 0.125f, 0.0625f},
    {0.125f,  0.25f,  0.125f},
    {0.0625f, 0.125f, 0.0625f}
};

std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), 
                        std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// CPU-based Gaussian blur
void gaussianBlurCpu(const float* inputImage, float* outputImage, const float* filter, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int fy = -1; fy <= 1; fy++) {
                for (int fx = -1; fx <= 1; fx++) {
                    int nx = x + fx;
                    int ny = y + fy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += inputImage[ny * width + nx] * 
                               filter[(fy + 1) * FILTER_SIZE + (fx + 1)];
                    }
                }
            }
            outputImage[y * width + x] = sum;
        }
    }
}

float computeMse(const float* reference, const float* test, int size) {
    float mse = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = reference[i] - test[i];
        mse += diff * diff;
    }
    return mse / size;
}

int main() {
    // Initialize image
    std::vector<float> inputImage(SIZE * SIZE);
    std::vector<float> outputNaive(SIZE * SIZE);
    std::vector<float> outputOptimized(SIZE * SIZE);
    std::vector<float> outputCpu(SIZE * SIZE);
    
    // Generate mock data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < SIZE * SIZE; i++) {
        inputImage[i] = dist(gen);
    }
    
    // CPU version
    gaussianBlurCpu(inputImage.data(), outputCpu.data(), &filter[0][0], SIZE, SIZE);
    
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program programNaive, programOptimized;
    cl_kernel kernelNaive, kernelOptimized;
    cl_mem bufInput, bufOutput, bufFilter;
    cl_int err;
    
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform: " << err << std::endl;
        return 1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device: " << err << std::endl;
        return 1;
    }
    
    char deviceName[128];
    size_t maxWorkGroupSize;
    cl_ulong deviceLocalMemSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &deviceLocalMemSize, nullptr);
    std::cout << "Device: " << deviceName << "\n";
    std::cout << "Max Work Group Size: " << maxWorkGroupSize << "\n";
    std::cout << "Local Memory Size: " << deviceLocalMemSize << " bytes\n";
    
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << err << std::endl;
        return 1;
    }
    
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create queue: " << err << std::endl;
        return 1;
    }
    
    // Read kernel source files
    std::string naiveKernelSource = readFile("gaussianBlurNaive.cl");
    std::string optimizedKernelSource = readFile("gaussianBlurOptimized.cl");
    if (naiveKernelSource.empty() || optimizedKernelSource.empty()) {
        std::cerr << "Failed to read kernel source files" << std::endl;
        return 1;
    }
    
    const char* naiveSource = naiveKernelSource.c_str();
    const char* optimizedSource = optimizedKernelSource.c_str();
    
    programNaive = clCreateProgramWithSource(context, 1, &naiveSource, nullptr, &err);
    programOptimized = clCreateProgramWithSource(context, 1, &optimizedSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << err << std::endl;
        return 1;
    }
    
    err = clBuildProgram(programNaive, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(programNaive, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(programNaive, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Naive program build failed: " << err << "\nBuild Log:\n" << log.data() << std::endl;
        return 1;
    }
    
    err = clBuildProgram(programOptimized, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(programOptimized, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(programOptimized, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Optimized program build failed: " << err << "\nBuild Log:\n" << log.data() << std::endl;
        return 1;
    }
    
    kernelNaive = clCreateKernel(programNaive, "gaussianBlurNaive", &err);
    kernelOptimized = clCreateKernel(programOptimized, "gaussianBlurOptimized", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel: " << err << std::endl;
        return 1;
    }
    
    bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              SIZE * SIZE * sizeof(float), inputImage.data(), &err);
    bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * SIZE * sizeof(float), nullptr, &err);
    bufFilter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               FILTER_SIZE * FILTER_SIZE * sizeof(float), (void*)filter, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffers: " << err << std::endl;
        return 1;
    }
    
    err = clSetKernelArg(kernelNaive, 0, sizeof(cl_mem), &bufInput);
    err |= clSetKernelArg(kernelNaive, 1, sizeof(cl_mem), &bufOutput);
    err |= clSetKernelArg(kernelNaive, 2, sizeof(cl_mem), &bufFilter);
    err |= clSetKernelArg(kernelNaive, 3, sizeof(int), &SIZE);
    err |= clSetKernelArg(kernelNaive, 4, sizeof(int), &SIZE);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set naive kernel args: " << err << std::endl;
        return 1;
    }
    
    // Work-group sizes
    const size_t workGroupSizes[][2] = {{8, 8}, {16, 16}, {32, 32}};
    const int numSizes = 3;
    size_t globalSize[2] = {SIZE, SIZE};
    
    std::cout << "\n--- Testing Naive and Optimized Kernels ---";
    for (int i = 0; i < numSizes; i++) {
        size_t localSize[2] = {workGroupSizes[i][0], workGroupSizes[i][1]};
        size_t localMemSize = (localSize[0] + 2 * HALO) * (localSize[1] + 2 * HALO) * sizeof(float);
        
        size_t maxWorkGroupSize;
        err = clGetKernelWorkGroupInfo(kernelOptimized, device, CL_KERNEL_WORK_GROUP_SIZE, 
                                       sizeof(size_t), &maxWorkGroupSize, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get work-group info: " << err << std::endl;
            continue;
        }
        
        if (localSize[0] * localSize[1] > maxWorkGroupSize) {
            std::cout << "\nSkipping work-group size " << localSize[0] << "x" << localSize[1] 
                      << " (exceeds max: " << maxWorkGroupSize << ")\n";
            continue;
        }
        
        if (localMemSize > deviceLocalMemSize) {
            std::cout << "\nSkipping work-group size " << localSize[0] << "x" << localSize[1] 
                      << " (local memory exceeds: " << deviceLocalMemSize << " bytes)\n";
            continue;
        }
        
        err = clSetKernelArg(kernelOptimized, 0, sizeof(cl_mem), &bufInput);
        err |= clSetKernelArg(kernelOptimized, 1, sizeof(cl_mem), &bufOutput);
        err |= clSetKernelArg(kernelOptimized, 2, sizeof(cl_mem), &bufFilter);
        err |= clSetKernelArg(kernelOptimized, 3, sizeof(int), &SIZE);
        err |= clSetKernelArg(kernelOptimized, 4, sizeof(int), &SIZE);
        err |= clSetKernelArg(kernelOptimized, 5, localMemSize, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set optimized kernel args: " << err << std::endl;
            continue;
        }
        
        // Naive kernel
        cl_event eventNaive;
        err = clEnqueueNDRangeKernel(queue, kernelNaive, 2, nullptr, globalSize, nullptr, 
                                     0, nullptr, &eventNaive);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue naive kernel: " << err << std::endl;
            continue;
        }
        clFinish(queue);
        
        cl_ulong start, end;
        clGetEventProfilingInfo(eventNaive, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        clGetEventProfilingInfo(eventNaive, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        double naiveTime = (end - start) / 1e6;
        
        err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, SIZE * SIZE * sizeof(float), 
                                  outputNaive.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read naive output: " << err << std::endl;
            continue;
        }
        
        // Optimized kernel
        cl_event eventOptimized;
        err = clEnqueueNDRangeKernel(queue, kernelOptimized, 2, nullptr, globalSize, localSize, 
                                     0, nullptr, &eventOptimized);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue optimized kernel: " << err << std::endl;
            continue;
        }
        clFinish(queue);
        
        clGetEventProfilingInfo(eventOptimized, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        clGetEventProfilingInfo(eventOptimized, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        double optimizedTime = (end - start) / 1e6;
        
        err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, SIZE * SIZE * sizeof(float), 
                                  outputOptimized.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to read optimized output: " << err << std::endl;
            continue;
        }
        
        float mseNaive = computeMse(outputCpu.data(), outputNaive.data(), SIZE * SIZE);
        float mseOptimized = computeMse(outputCpu.data(), outputOptimized.data(), SIZE * SIZE);
        
        std::cout << "\nWork-group size: " << localSize[0] << "x" << localSize[1] << "\n";
        std::cout << "Naive kernel time: " << naiveTime << " ms\n";
        std::cout << "Optimized kernel time: " << optimizedTime << " ms\n";
        std::cout << "Speedup: " << (naiveTime / optimizedTime) << "x\n";
        std::cout << "Naive MSE: " << mseNaive << "\n";
        std::cout << "Optimized MSE: " << mseOptimized << "\n";
        
        clReleaseEvent(eventNaive);
        clReleaseEvent(eventOptimized);
    }
    
    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufOutput);
    clReleaseMemObject(bufFilter);
    clReleaseKernel(kernelNaive);
    clReleaseKernel(kernelOptimized);
    clReleaseProgram(programNaive);
    clReleaseProgram(programOptimized);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
