"""dnn: run one language model across whatever devices you have.

Nodes (python -m dnn.node) need only Python 3 + NumPy. The coordinator
(python -m dnn.run) measures every device, assigns layers to fit each one's
memory and speed, streams each node its weights, and generates text.
"""

__version__ = "0.2.0"
