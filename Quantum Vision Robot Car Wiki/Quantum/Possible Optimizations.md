Below is a list of optimizations that can maybe be done to speed up and/or increase accuracy in the QED. We do not know if any of these will work, we still need to look into them.
# Boundary Error Fix
If we can get so that the [[Quantum Circuit]] will not compute the perimeter and just always put zeros there then it will save shots allowing us to one of two things.

1. Keep the same number of shots but have a much sharper edge detection
2. Lower the shots and decrease the computation time.

# Image Mapping via Pairing Function
This one I feel is the least likely to work. It hinges on weather or the pairing function can maintain locality after passing through the [[Quantum Circuit]]. 

So for each pixel it is an integer between \[0,255] and a Pairing Function can take two numbers and give one. So if we have both the H and V images and pass each pixel through the pairing function we get a single image.

If we then pass that image through the [[Quantum Circuit]] and then take the result and pass it through the inverse of the Pairing Function to get the two numbers back, the hope is that the two numbers will be the result of the H and V.

[function 1](https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way?rq=1) Use bdonlan answer

[Cantor Pairing](https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function)

# Polarized Encoding
So when we encode the circuit we add an aux q-bit, and when we decode we only grab the odd bit-strings. So only half of the q-bits are used, and I am wondering if can make use of the other half of the q-bits. 

If that is possible then in the same number of q-bits we can encode double the information, allowing us to encode the vertical image along with the horizontal in one circuit.