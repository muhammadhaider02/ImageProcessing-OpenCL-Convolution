__kernel void gaussianBlurOptimized(
    __global const float* inputImage,
    __global float* outputImage,
    __constant float* filter,
    const int width,
    const int height,
    __local float* tile)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lw = get_local_size(0);
    int lh = get_local_size(1);
    
    int tw = lw + 2;
    int th = lh + 2;
    
    int tx = lx + 1;
    int ty = ly + 1;
    
    tile[ty * tw + tx] = (gx < width && gy < height) ? 
                         inputImage[gy * width + gx] : 0.0f;
    
    // Load halo pixels
    if (lx == 0 && gy < height) {
        tile[ty * tw + (tx - 1)] = (gx > 0) ? 
                                   inputImage[gy * width + (gx - 1)] : 0.0f;
    }
    if (lx == lw - 1 && gy < height) {
        tile[ty * tw + (tx + 1)] = (gx < width - 1) ? 
                                   inputImage[gy * width + (gx + 1)] : 0.0f;
    }
    if (ly == 0 && gx < width) {
        tile[(ty - 1) * tw + tx] = (gy > 0) ? 
                                   inputImage[(gy - 1) * width + gx] : 0.0f;
    }
    if (ly == lh - 1 && gx < width) {
        tile[(ty + 1) * tw + tx] = (gy < height - 1) ? 
                                   inputImage[(gy + 1) * width + gx] : 0.0f;
    }
    
    // Load corner pixels
    if (lx == 0 && ly == 0) {
        tile[(ty - 1) * tw + (tx - 1)] = (gx > 0 && gy > 0) ? 
                                         inputImage[(gy - 1) * width + (gx - 1)] : 0.0f;
    }
    if (lx == lw - 1 && ly == 0) {
        tile[(ty - 1) * tw + (tx + 1)] = (gx < width - 1 && gy > 0) ? 
                                         inputImage[(gy - 1) * width + (gx + 1)] : 0.0f;
    }
    if (lx == 0 && ly == lh - 1) {
        tile[(ty + 1) * tw + (tx - 1)] = (gx > 0 && gy < height - 1) ? 
                                         inputImage[(gy + 1) * width + (gx - 1)] : 0.0f;
    }
    if (lx == lw - 1 && ly == lh - 1) {
        tile[(ty + 1) * tw + (tx + 1)] = (gx < width - 1 && gy < height - 1) ? 
                                         inputImage[(gy + 1) * width + (gx + 1)] : 0.0f;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (gx >= width || gy >= height) return;
    
    float sum = 0.0f;
    for (int fy = -1; fy <= 1; fy++) {
        for (int fx = -1; fx <= 1; fx++) {
            sum += tile[(ty + fy) * tw + (tx + fx)] *
                   filter[(fy + 1) * 3 + (fx + 1)];
        }
    }
    outputImage[gy * width + gx] = sum;
}
