import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
#make sure to: pip install qiskit_algorithms
from qiskit_machine_learning.neural_networks import EstimatorQNN

import logging

log = logging.getLogger(__name__)

# Set the seed for qiskit so randomness is controlled between runs.
algorithm_globals.random_seed = 12345

# TODO Maybe have all circuit functions be in this file?

# This is a convolution on 2-qbits. It will be used to expand to an n-qbit convolution later.
# params: ParameterVector for changeable values in circuit.

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

"""
Full convolution layer for main circuit.

num_qubits: number of qbits convolution will operate on
param_prefix: name for parameters in sub-circuit
"""
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


# This is a 2-bit pooling circuit. It will be used to expand to an n-qbit pool later
def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


# Full pooling layer for main circuit.
def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


# Main Building of QCNN circuit
# TODO change convolution/pooling layers to be auto generated off of qbits
def build_qcnn(qbits):
    log.debug("Building QCNN")
    feature_map = ZFeatureMap(qbits)

    ansatz = QuantumCircuit(qbits, name="Ansatz")


    for groups in qbits:

        num_Groups = 2
        qbits_List = [0]*qbits
        chunked_List = []
        # First Convolutional Layer
        #ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)
        #first convolutional layer will go to n_qubits
        ansatz.compose(conv_layer(8, "с1"), list(range(8)), inplace=True)
        # First Pooling Layer
        #ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        #first pooling layer will be in n of 2^n groups?
        for i in range(0, len(qbits), num_Groups):
            chunked_List.append(qbits_List[i:i+num_Groups])

        ansatz.compose(pool_layer(chunked_List[0], chunked_List[1], "p1"), list(range(qbits)), inplace=True)
        qbits = qbits / 2

    [0,1,2,3],[4,5,6,7]
        list[0]    list[1]

    [0,1],[2,3]
    list[0]   list[1]


    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    return qnn