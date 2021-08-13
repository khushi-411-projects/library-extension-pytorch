// https://github.com/krshrimali/Digit-Recognition-MNIST-SVHN-PyTorch-CPP/blob/master/training.cpp

#include <torch/torch.h>
#include <fstream> // No need to import iostream everytime ;) there is fstream!

/*
 * Load Dataset First
 * Make a data loader then
 * Create the network
 * Start training!
 */

int main(int argc, char** argv) {
        // Dataset path
        const std::string dataset_path = "/Users/91939/Downloads/datasets/pytorch/fashion_mnist/unzipped";
        const int batch_size = 64;
        const float learning_rate = 0.01;
        const int num_epochs = 10;
        const int print_every = 100;

        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(torch::data::datasets::MNIST(dataset_path).map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
                        torch::data::transforms::Stack<>())), batch_size);

        // Note: make_shared ----- owns and stores a pointer to a newly allocated object of type T
        auto net = std::make_shared<Net>();
        torch::optim::SGD optimizer(net->parameters(), learning_rate);

        for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
                size_t batch_index = 0;
                for (auto& batch : *data_loader) {
                        // Reset gradients
                        optimizer.zero_grad();
                        // Execute model
                        torch::Tensor prediction = net->forward(batch.data);
                        // Compute loss
                        torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                        // Compute gradients
                        loss.backward();
                        // Update parameters
                        optimizer.step();

                        // Saving checkpoints as net<batch_index>.pt
                        if (++batch_index % print_every == 0) {
                                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                                        << " | Loss: " << loss.item<float>() << std::endl;
                                torch::save(net, "net" + str(batch_index) + ".pt");
                        }
                }
        }
}
