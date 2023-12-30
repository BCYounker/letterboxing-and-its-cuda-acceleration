/**
 * Changelog for Version 2.0
 * 
 *  
 * This version use improved letterboxing.
 * This advanced version improves upon the basic letterboxing approach of Version 1, which involved 
 * placing the original image in the top-left corner of a black square background and resizing it to 640x640.
 *
 * Enhancements in Version 2 include:
 * - Aspect ratio preservation through dynamic scaling.
 * - Centered image placement for better visual balance.
 * - Use of affine transformation for precise image scaling and positioning.
 * - Greater adaptability to various image sizes and aspect ratios.
 *
 *  Related function is format_yolov5
 */

// Include Libraries.
#include <fstream>
#include <opencv2/opencv.hpp>

// Constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
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
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
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

    // auto input_image = format_yolov5(image);
    // std::string output_file = "../output_cpp_letterbox_improved_test.jpg";
    // if(cv::imwrite(output_file, input_image))
    
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
    frame=format_yolov5(frame_original); // letterboxing
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

    std::string output_file = "../output_cpp_letterbox_improved.jpg";
    if(cv::imwrite(output_file, frame)) {
        std::cout << "Output saved as " << output_file << std::endl;
    } else {
        std::cout << "Error saving the image." << std::endl;
    }
    // cv::imshow("output", frame);
    // cv::waitKey(0);

    return 0;
}