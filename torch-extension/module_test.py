import torch
import torch.utils.cpp_extension

op_source = """
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>

void warp_perspective(std::string path, std::string m_path) {
        std::string model_path = m_path;

        std::ifstream images(path, std::ios::binary);

        auto tensor =
             torch::empty({1, 3, 224, 224}, torch::kByte);
        images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
        tensor = tensor.to(torch::kFloat32).div_(255);

        // TODO: Check if img was read properly or not
        // Resize image to the expected dimensions for the model
        // No need to do CUBIC interpolation - linear should be fine (performance)
        // cv::resize(img, img, cv::Size(224, 224), cv::INTER_LINEAR);

        // torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
        // img_tensor = img_tensor.permute({0, 3, 1, 2});
        // img_tensor = img_tensor.to(torch::kF32);

        torch::jit::script::Module model;
        try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                std::cout << "Model path: " << model_path << std::endl;
                model = torch::jit::load(model_path);
        }
        catch (const c10::Error& e) {
                std::cerr << "error loading the model";
                return;
        }

        std::vector<torch::jit::IValue> input;
        input.push_back(tensor);

        torch::Tensor prob = model.forward(input).toTensor();

        // TODO: get the max index out of prob.data
        // return that index (that will be the number)
        std::cout << "Probability of 0: " << *(prob.data<float>()) * 100.
                << "Probability of 1: " << *(prob.data<float>() + 1)*100. << std::endl;
}

TORCH_LIBRARY(my_ops, m) {
        m.def("warp_perspective", &warp_perspective);
}
"""

torch.utils.cpp_extension.load_inline(
    name="warp_perspective",
    cpp_sources=op_source,
    is_python_module=False,
    verbose=True,
)

print(torch.ops.my_ops.warp_perspective)
torch.ops.my_ops.warp_perspective("five.png", "model.pt")
