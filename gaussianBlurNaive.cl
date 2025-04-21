__kernel void gaussianBlurNaive(
    __global const float* inputImage,
    __global float* outputImage,
    __constant float* filter,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    for (int fy = -1; fy <= 1; fy++) {
        for (int fx = -1; fx <= 1; fx++) {
            int nx = x + fx;
            int ny = y + fy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += inputImage[ny * width + nx] * 
                       filter[(fy + 1) * 3 + (fx + 1)];
            }
        }
    }
    outputImage[y * width + x] = sum;
}
