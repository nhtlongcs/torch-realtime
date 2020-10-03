#include <torch/torch.h>
#include <torch/script.h>
#include <opencv4/opencv2/opencv.hpp>

#include <codecvt>
#include <locale>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

at::Tensor imageTransform(cv::Mat image)
{
    constexpr const int SCALE_SIZE = 256;
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size{image.cols * SCALE_SIZE / image.rows, SCALE_SIZE}, 0.0, 0.0, cv::INTER_LINEAR);
    const int cropSize = 224;
    const int offsetW = (resizedImage.cols - cropSize) / 2;
    const int offsetH = (resizedImage.rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    resizedImage = resizedImage(roi).clone();
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);
    at::Tensor image_tensor = torch::from_blob(resizedImage.data, {resizedImage.rows, resizedImage.cols, 3}, at::kByte);
    image_tensor = image_tensor / 255.0f;

    at::Tensor mean = torch::tensor({0.485, 0.456, 0.406});
    at::Tensor std = torch::tensor({0.229, 0.224, 0.225});

    image_tensor = (image_tensor - mean) / std;
    image_tensor = image_tensor.permute({2,0,1});   // C,H,W
    image_tensor = image_tensor.unsqueeze(0);       // 1,C,H,W

    return image_tensor;
}

cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    std::cout << "converting tensor to cvmat\n";
    tensor = tensor.squeeze(0).permute({1, 2, 0});
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
    torch::jit::script::Module module;
    try {
        const std::string path{argv[1]};
        module = torch::jit::load(path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        std::cerr << e.msg();
        return -1;
    }
    std::cout << "OK!";

    // Test forward ok
    const std::string image_path{argv[2]};
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
    at::Tensor image_tensor = imageTransform(image);
    std::cout << image_tensor.sizes() << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image_tensor.to("cpu"));


    at::Tensor output = module.forward(inputs).toTensor();
    // std::cout << output << std::endl;
    // std::cout << output.sizes() << std::endl;
    cv::Mat output_mat = TensorToCVMat(output);
    cv::imwrite("./output_cpp.jpg", output_mat);



    // int n = 100;
    // // int n = 1;
    // at::Tensor output;
    // std::cout << "Start DRY RUN!!!!" << std::endl;
    // for (int i = 0; i < n; i++)
    // {
    //     output = module.forward(inputs).toTensor();
    // }

    // std::cout << "Start!!!!" << std::endl;
    // auto start_time = std::chrono::system_clock::now();
    // // std::this_thread::sleep_for(std::chrono::seconds(2));
    // for (int i = 0; i < n; i++)
    // {
    //     output = module.forward(inputs).toTensor();
    // }
    // auto end_time = std::chrono::system_clock::now();
    // std::cout << "Time (total) = " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() << std::endl;
    // std::cout << "Time (avg) = " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / (n * 1.0f);
    // output = output.argmax(-1);
    // std::cout << std::endl << output;

    // const std::string vocab_path = "/home/vinhloiit/projects/vbhc_info_extraction_for_only_team/models/weights/textline_ocr/09212020/vocab.txt";
    // std::wstring alphabets = get_alphabets_from_file(vocab_path);

    // std::cout << "Alphabet length: " << alphabets.length() << std::endl;
    // auto x = alphabets.data();
    // FILE* outFile = fopen( "Serialize.txt", "w+,ccs=UTF-8");
    // fwrite(x, sizeof(wchar_t), wcslen(x), outFile);
    // for (int i = 0; i < alphabets.size(); i++)
    // {
    //     std::cout << alphabets[i] << std::endl;
    // }

    // long* ptr = output.data_ptr<long>();
    // long index = 0;
    // std::wstring result;
    // for (int row = 0; row < output.sizes()[0]; row++)
    // {
    //     for (int col = 0; col < output.sizes()[1]; col++)
    //     {
    //         index = (*ptr++);
    //         result += alphabets[index];
    //     }
    // }
    // // std::wcout << result << std::endl;
    // auto x = result.data();
    // FILE* outFile = fopen( "Serialize.txt", "w+,ccs=UTF-8");
    // fwrite(x,  wcslen(x) * sizeof(wchar_t), 1, outFile);
    // fclose(outFile);


    return 0;
}