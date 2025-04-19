# mini distributed Neural Network Inference Framework

This framework enables you to run neural network inference across multiple devices, distributing the computational load and potentially improving performance. It leverages gRPC for inter-node communication and allows you to split a neural network model into parts, assigning each part to a different node in the network.

## Key Features

*   **Distributed Inference:** Distribute the computational workload of a neural network across multiple devices.
*   **gRPC Communication:** Uses gRPC for efficient and reliable communication between nodes.
*   **Model Partitioning:** Supports splitting a neural network model into multiple parts, each handled by a separate node.
*   **Flexible Configuration:** Easily configure the network topology and model partitioning using a JSON configuration file.
*   **PyTorch Integration:** Built using PyTorch for defining and manipulating neural network models.

## Architecture

The framework consists of multiple nodes, each running a `node.py` script. Each node is responsible for:

1.  **Receiving Input:** Receiving a tensor (data) from the previous node in the pipeline (or from an initial client).
2.  **Performing Computation:** Executing a specific part of the neural network model on the received tensor.
3.  **Forwarding Output:** Sending the resulting tensor to the next node in the pipeline (or returning the final result to the client).

The `node_service.proto` file defines the gRPC service and message definitions used for communication between nodes.

## Setup and Installation

1.  **Install Dependencies:**



    Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Define Model Parts:**

    *   Modify the `cifar_model_parts.py` file (startLine: 2, endLine: 58) to define how your neural network model is split into parts.
    *   Create classes that inherit from `nn.Module` for each part of the model.
    *   Ensure that the `forward` method of each model part performs the correct computation.
    *   The provided example splits a CIFAR-10 model into two parts: convolutional layers (Node 0) and fully connected layers (Node 1).

3.  **Configure Nodes:**

    *   Create a `config.json` file (startLine: 1, endLine: 18) to define the network topology and node configurations.
    *   Specify the `id`, `address`, and `part_index` for each node.
    *   Set the `model_weights` path to the location of your trained model weights file (`.pth`).
    *   Define `num_parts` to match the number of model parts.
    *   `return_to_node_id` specifies which node the final result should be returned to.

    ```json
    {
      "nodes": [
        {
          "id": "node1",
          "address": "192.168.1.101:50051",
          "part_index": 0
        },
        {
          "id": "node2",
          "address": "192.168.1.120:50051",
          "part_index": 1
        }
      ],
      "model_weights": "./cifar10_model.pth",
      "num_parts": 2,
      "return_to_node_id": "node1"
    }
    ```

4.  **Prepare Model Weights:**

    *   The framework uses a single weights file (`.pth`) containing the weights for the entire model.
    *   Each node loads the full state dictionary but only uses the weights relevant to its assigned model part.
    *   Ensure that the `MODEL_PARTS_CLASSES` dictionary in `node.py` (startLine: 29, endLine: 32) maps the correct model part index to the corresponding class in `cifar_model_parts.py`.

## Running the Framework

1.  **Start Nodes:**

    Run the `node.py` script on each device, specifying the `node_id` and `config` file as command-line arguments.

    ```bash
    python node.py --node_id node1 --config ./config.json
    python node.py --node_id node2 --config ./config.json
    ```

    *   **Important:** Ensure that the addresses specified in `config.json` match the actual IP addresses and ports of the devices running the nodes.

2.  **Initiate Inference (Node 0):**

    To start the inference pipeline, run the `node.py` script on the first node (the node with `part_index` 0), providing the path to an input image using the `--input_image` flag.

    ```bash
    python node.py --node_id node1 --config ./config.json --input_image ./image.png
    ```

    *   The first node will load the image, perform its part of the computation, and forward the result to the next node.
    *   The last node will perform its computation and return the final prediction.

## Modifying Model Parts

To adapt this framework to different neural networks, you'll need to modify the `cifar_model_parts.py` file. Here's a general guide:

1.  **Define `NeuralNetwork`:** Create a class that defines your full neural network model.
2.  **Create Model Part Classes:** For each node, create a class that inherits from `nn.Module` and represents the part of the model that will run on that node.
3.  **Copy Layers:** In the `__init__` method of each model part class, copy the necessary layers from the original `NeuralNetwork` instance.
4.  **Implement `forward`:** Implement the `forward` method for each model part class to perform the correct computation. The output of one part should be suitable as input for the next part.
5.  **Update `MODEL_PARTS_CLASSES`:** Update the `MODEL_PARTS_CLASSES` dictionary in `node.py` to map the correct model part index to the corresponding class.

## Notes

*   This framework is designed for inference only. Training a distributed neural network requires a more complex setup.
*   The performance of the framework depends on the network bandwidth and the computational capabilities of each device.
*   Error handling and fault tolerance can be improved for production deployments.
*   The framework currently supports a linear pipeline topology. More complex topologies can be implemented by modifying the node communication logic.
