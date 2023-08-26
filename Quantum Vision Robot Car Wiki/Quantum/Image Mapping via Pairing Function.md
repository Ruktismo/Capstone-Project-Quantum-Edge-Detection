This one I feel is the least likely to work. It hinges on weather or the pairing function can maintain locality after passing through the [[Quantum Circuit]]. 

So for each pixel it is an integer between \[0,255] and a Pairing Function can take two numbers and give one. So if we have both the H and V images and pass each pixel through the pairing function we get a single image.

If we then pass that image through the [[Quantum Circuit]] and then take the result and pass it through the inverse of the Pairing Function to get the two numbers back, the hope is that the two numbers will be the result of the H and V.

[function 1](https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way?rq=1) Use bdonlan answer

[Cantor Pairing](https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function)
