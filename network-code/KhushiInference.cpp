#include <torch/torch.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
// #include <pybind11/pybind11.h>

// namespace py = pybind11;

/* KhushiInference obj("/home/khushi/Downloads/zero.jpg", "net.pt");
 * obj.read_image();
 * obj.test();
 */
class KhushiInference {
        private:
                std::string img_path;
                std::string model_path;
                cv::Mat img;
        public:
                // TODO: Search why we did const std::string&
                KhushiInference(std::string path, std::string m_path) {
                        img_path = path;
                        model_path = m_path;
                }

                void read_image() {
                        img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
                        // TODO: Check if img was read properly or not
                        // Resize image to the expected dimensions for the model
                        // No need to do CUBIC interpolation - linear should be fine (performance)
                        cv::resize(img, img, cv::Size(224, 224), cv::INTER_LINEAR);
                }

                void test() {
                        torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
                        img_tensor = img_tensor.permute({0, 3, 1, 2});
                        img_tensor = img_tensor.to(torch::kF32);

                        torch::jit::script::Module model;
                        try {
                                // Deserialize the ScriptModule from a file using torch::jit::load().
                                std::cout << "Model path: " << model_path << std::endl;
                                model = torch::jit::load(model_path);
                        }
                        catch (const c10::Error& e) {
                                std::cerr << "error loading the model\n";
                                return;
                        }

                        std::vector<torch::jit::IValue> input;
                        input.push_back(img_tensor);

                        torch::Tensor prob = model.forward(input).toTensor();

                        // TODO: get the max index out of prob.data
                        // return that index (that will be the number)
                        std::cout << "Probability of 0: " << *(prob.data<float>()) * 100.
                                << "Probability of 1: " << *(prob.data<float>() + 1)*100. << std::endl;
                }
};
/*
PYBIND11_MODULE(khushi, m) {
        py::class_<KhushiInference>(m, "KhushiInference")
                .def(py::init<std::string, std::string>())
                .def("read_image", &KhushIinference::read_image)
                .def("test", &KhushiInference::test);
}*/
