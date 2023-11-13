import logging
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import algorithm_globals


log = logging.getLogger("Quantum_Edge_Detection")

# Set the seed for qiskit so randomness is controlled between runs.
algorithm_globals.random_seed = 12345


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
def build_qcnn(qbits):
    log.debug("Building QCNN")
    # TODO do parameterized QPIE feature map for input to circuit.
    #  It seams that qiskit may not currently support a way to do parameterized QPIE?
    #  At least not in a way that the QNN class would recognize.
    feature_map = ZFeatureMap(qbits)

    ansatz = QuantumCircuit(qbits, name="Ansatz")
    layer = 1
    i = qbits
    ansatz.compose(conv_layer(i, 'c' + str(layer)), list(range(qbits)), inplace=True)
    ansatz.compose(pool_layer(list(range(0, i // 2)), list(range(i // 2, i)), param_prefix="p" + str(layer)),
                   list(range(0, qbits)), inplace=True)
    layer += 1
    i = i // 2
    while i > 1:
        ansatz.compose(conv_layer(i, 'c' + str(layer)), list(range(qbits - i, qbits)), inplace=True)
        ansatz.compose(pool_layer(list(range(0, i // 2)), list(range(i // 2, i)), param_prefix="p" + str(layer)),
                       list(range(qbits - i, qbits)), inplace=True)
        layer += 1
        i = i // 2

    # returning the feature map and ansatz for VQC
    return feature_map, ansatz
