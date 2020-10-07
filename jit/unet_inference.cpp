%%writefile unet_inference.cpp

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <time.h>
#include <codecvt>
#include <locale>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
torch::DeviceType device_type;

at::Tensor imageTransform(cv::Mat image)
{
    constexpr const int SCALE_SIZE = 256;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size{SCALE_SIZE, SCALE_SIZE}, 0.0, 0.0, cv::INTER_LINEAR);
    const int cropSize = 224;
    // const int offsetW = (resizedImage.cols - cropSize) / 2;
    // const int offsetH = (resizedImage.rows - cropSize) / 2;
    // const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    // resizedImage = resizedImage(roi).clone();
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    at::Tensor image_tensor = torch::from_blob(resizedImage.data, {resizedImage.rows, resizedImage.cols, 3}, at::kByte);
    image_tensor = image_tensor / 255.0f;

    at::Tensor mean = torch::tensor({0.485, 0.456, 0.406});
    at::Tensor std = torch::tensor({0.229, 0.224, 0.225});

    // image_tensor = (image_tensor - mean) / std;
    image_tensor = image_tensor.permute({2,0,1});   // C,H,W
    image_tensor = image_tensor.unsqueeze(0);       // 1,C,H,W

    return image_tensor;
}

cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    // std::cout << "converting tensor to cvmat\n";
    tensor = tensor.argmax(1);
    tensor = tensor.permute({1, 2, 0});
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kUInt8);
    tensor = tensor.to(torch::kCPU);
    int height = tensor.size(0);
    int width = tensor.size(1);
    cv::Mat mat{cv::Size2i{width, height}, CV_8UC1};
    std::memcpy((void*)mat.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());
    cv::resize(mat, mat, cv::Size{320, 240});
    return mat.clone();
}

int main(int argc, char** argv)
{

    if (torch::cuda::is_available())
    {
        device_type = torch::kCUDA;
    } 
    else
    {
        device_type = torch::kCPU;
    }
    
    torch::Device device(device_type);
    torch::jit::script::Module module;
    try {
        const std::string path{argv[1]};
        module = torch::jit::load(path);
        
        module.to(device);
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        std::cerr << e.msg();
        return -1;
    }
 
    // std::cout << "OK!";

    // Test forward ok
    const std::string image_path{argv[2]};
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
    at::Tensor image_tensor = imageTransform(image);
    // std::cout << image_tensor.sizes() << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor.to(device));
    
    time_t start;
    time_t end;
    start = time(NULL);
    at::Tensor output = module.forward(inputs).toTensor();
 
    for (int i = 0; i < 1; i++)
        output = module.forward(inputs).toTensor();
    end = time(NULL);
    std::cout << (end-start);
    // std::cout << output << std::endl;
    // std::cout << output.sizes() << std::endl;
    cv::Mat output_mat = TensorToCVMat(output);
    cv::imwrite("./output_cpp.jpg", output_mat);

    return 0;
}