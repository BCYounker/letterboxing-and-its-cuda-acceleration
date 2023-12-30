/**
 * Changelog for Version 3.0
 * 
 *  
 * This version use cuda programming to accelerate letterboxing.
 *
 * Enhancements in Version 3 include:

 * - Around 5x-6x faster for letterboxing when use 1000 cuda cores
 * - Have same result with cpp version 2
 *
 *  Related cuda preprocessing code is from line 22 to line 179
 */
// Include Libraries.
#include <fstream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;

// Declaration of a pointer for the image buffer in GPU memory.
static uint8_t *img_buffer_device = nullptr;

// Structure to store an affine transformation matrix.
struct AffineMatrix
{
  float value[6]; // Six values of the 2x3 affine transformation matrix.
};

// CUDA kernel to perform affine warp transformation on images.
__global__ void warpaffine_kernel(
    uint8_t *src, int src_line_size, int src_width,
    int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge)
{
  // Calculate the global position of the thread.
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  // Return if the position exceeds the number of pixels to be processed.
  if (position >= edge)
    return;

  // Extract affine matrix values from the structure.
  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  // Calculate the destination coordinates corresponding to this thread.
  int dx = position % dst_width;
  int dy = position / dst_width;

  // Compute the source coordinates using the affine transformation matrix.
  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2; // Variables to store color components.

  // Check if the source coordinates are within the bounds of the source image.
  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
  {
    // Assign a constant value to the pixel if it is outside the source image bounds.
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  }
  else
  {
    // Perform bilinear interpolation to compute pixel value at non-integer positions.
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    // Default values for interpolation are set to const_value_st.
    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    // Compute interpolation weights.
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    // Pointers to the four surrounding pixels for interpolation.
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;

    // Adjust pointers if the surrounding pixels are within the image bounds.
    if (y_low >= 0)
    {
      if (x_low >= 0)
        v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width)
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height)
    {
      if (x_low >= 0)
        v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width)
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    // Compute the interpolated color components.
    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // Pointer to the destination pixel in the output image.
  float *pdst = dst + (dy * dst_width + dx) * 3;

  // Store the computed RGB values in the destination image.
  pdst[0] = c0; // Red
  pdst[1] = c1; // Green
  pdst[2] = c2; // Blue
}

// High-level function to set up and launch the CUDA warp affine kernel.
void process_cuda(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height)
{
  // Copy the source image data from host to device memory.
  int img_size = src_width * src_height * 3;
  CUDA_CHECK(cudaMemcpy(img_buffer_device, src, img_size, cudaMemcpyHostToDevice));

  // Compute the affine transformation matrix for scaling and centering (letterboxing).
  AffineMatrix s2d, d2s;
  // Calculate scale factor to maintain aspect ratio.
  float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  // Populate the source-to-destination affine matrix.
  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

  // Convert the matrices to cv::Mat for easy inversion.
  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  // Invert the affine transformation to get the destination-to-source matrix.
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  // Copy the inverted matrix back to the AffineMatrix structure.
  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  // Calculate the number of threads and blocks for CUDA kernel execution.
  int jobs = dst_height * dst_width;
  int threads = 1000;
  int blocks = ceil(jobs / (float)threads);
  // Launch the warp affine kernel.
  warpaffine_kernel<<<blocks, threads>>>(
      img_buffer_device, src_width * 3, src_width,
      src_height, dst, dst_width,
      dst_height, 0, d2s, jobs);
}

// Function to preprocess the image using CUDA for YOLOv5.
void format_yolov5_cuda(cv::Mat &src, float *input_device_buffer)
{
  // Call the process function with the source image and destination buffer.
  process_cuda(src.ptr(), src.cols, src.rows, input_device_buffer, INPUT_WIDTH, INPUT_HEIGHT);
}

// Initialize the CUDA preprocessing, allocating memory for the input buffer.
void cuda_preprocess_init(int max_image_size)
{
  // Allocate memory on the GPU for the image buffer.
  CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, max_image_size * 3));
}

// Constants
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

//Colors
//const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

// Letterboxing v1
// cv::Mat format_yolov5(const cv::Mat &source) {
//     int col = source.cols;
//     int row = source.rows;
//     int _max = MAX(col, row); 
//     cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3); //Create a Blank Square Image
//     source.copyTo(result(cv::Rect(0, 0, col, row)));
//     return result;
// }

// Letterboxing V2
inline cv::Mat format_yolov5(cv::Mat &source) // use inline to improve efficiency
{
  // Calculate the scale ratio to fit the image within the input size while keeping aspect ratio
  float scale = std::min(INPUT_HEIGHT / (float)source.rows, INPUT_WIDTH / (float)source.cols);

  // Calculate offsets to center the image on the canvas
  int offsetx = (INPUT_WIDTH - source.cols * scale) / 2;
  int offsety = (INPUT_HEIGHT - source.rows * scale) / 2;

  // Define three points in the source image for affine transformation: top-left, top-right, bottom-left
  cv::Point2f sourceTri[3];
  sourceTri[0] = cv::Point2f(0.f, 0.f);
  sourceTri[1] = cv::Point2f(source.cols - 1.f, 0.f);
  sourceTri[2] = cv::Point2f(0.f, source.rows - 1.f);

  // Corresponding points in the destination image, adjusted for scale and offset
  cv::Point2f dstTri[3];
  dstTri[0] = cv::Point2f(offsetx, offsety);
  dstTri[1] = cv::Point2f(source.cols * scale - 1.f + offsetx, offsety);
  dstTri[2] = cv::Point2f(offsetx, source.rows * scale - 1.f + offsety);

  // Calculate the affine transformation matrix
  cv::Mat warp_mat = cv::getAffineTransform(sourceTri, dstTri);

  // Create the destination image canvas
  cv::Mat warp_dst = cv::Mat::zeros(INPUT_HEIGHT, INPUT_WIDTH, source.type());

  // Apply the affine transformation
  cv::warpAffine(source, warp_dst, warp_mat, warp_dst.size());

  return warp_dst;
}

// Draw the predicted bounding box.
void draw_label(cv::Mat& input_image, std::string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Getting Top left corner and bottom right corner to draw black rectangle
    cv::Point tlc = cv::Point(left, top);
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

// Loading class list 
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("../coco/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

// Loading ONNX(network model)
void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("../weights/yolov5s.onnx");
    if (is_cuda)
    {
        std::cout << "OpenCV Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "OpenCV Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

// Structure of single detected image
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

void detect(cv::Mat &input_image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;
    
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float * classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += 85;

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main(int argc, char **argv)
{

    std::vector<std::string> class_list = load_class_list();

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    std::vector<Detection> output;
    cv::Mat frame;
    cv::Mat frame_original;
    frame_original = cv::imread("../sample.jpg"); // Load the image directly
    if (frame_original.empty())
    {
        std::cerr << "Error opening image file\n";
        return -1;
    }

    // create buffer for original frame (WxHx3) on cuda
    int max_image_size=frame_original.rows*frame_original.cols;
    cuda_preprocess_init(max_image_size); 

    // create buffer for desired picture (after letterboxing) on cuda
    float *target_buffer = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&target_buffer, 640*640*3*sizeof(float)));

    // Cuda event to record the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

    // Running on GPU
    cudaEventRecord(start, 0);
    //GPU's letterboxing function
    format_yolov5_cuda(frame_original, target_buffer);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time spent on GPU: %f ms\n", time);

    // Running on CPU
    cudaEventRecord(start, 0);
    format_yolov5(frame_original);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time spent on CPU: %f ms\n", time);
    
    // Create a buffer in CPU's memory, and copy the result from GPU to CPU
    float* host_buffer = new float[640 * 640 * 3];
    CUDA_CHECK(cudaMemcpy(host_buffer, target_buffer, 640 * 640 * 3*sizeof(float), cudaMemcpyDeviceToHost));
    //print each element
    // for (int i = 0; i < 640 * 640 * 3; ++i) {
    //     if (host_buffer[i] > 1e-6) {
    //         std::cout << "Value at [" << i << "]: " << host_buffer[i] << std::endl;
    //     }
    // }
    // Convert the result from consecutive list to image
    cv::Mat image(640,640, CV_32FC3, host_buffer);
    image.convertTo(frame, CV_8UC3); 

    // release buffer on GPU
    cudaFree(target_buffer);
    cudaFree(img_buffer_device);

    detect(frame, net, output, class_list);

    int detections = output.size();

    for (int i = 0; i < detections; ++i)
    {

        auto detection = output[i];
        auto box = detection.box;
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        auto classId = detection.class_id;
        auto confidence= detection.confidence;

        // Draw bounding box.
        // different color: const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, cv::Point(left, top), cv::Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get class name and confidence
        std::string label = cv::format("%.2f", confidence);
        label = class_list[classId] + ":" + label;
        // Draw class labels.
        draw_label(frame, label, left, top);
    }
    
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time : %.2f ms", t);
    std::cout << label << std::endl;
    cv::putText(frame, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    std::string output_file = "../output_cpp_letterbox_cuda.jpg";
    if(cv::imwrite(output_file, frame)) {
        std::cout << "Output saved as " << output_file << std::endl;
    } else {
        std::cout << "Error saving the image." << std::endl;
    }
    // cv::imshow("output", frame);
    // cv::waitKey(0);

    return 0;
}