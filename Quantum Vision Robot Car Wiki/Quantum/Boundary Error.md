The [[QED]] algorithm works well but we have found some errors with the way it processes image data. This forces us to cut out the perimeter of the image being scanned.

The main problem occurs with the Decrement Gate (DG) that detects edges, the DG detect edges by looking to the pixel on the right to see if it is part of an edge. But there is an issue, what do we do with the right edge of the image, there is no pixel to the right so to account for this so the DG just warps the image and has the last pixel look at the first.

This causes the [[Quantum Circuit]] to think that the boundary of the image is an edge. now in most examples the data is floating in the center of the image so this issue never presents itself, but when working on real images where every pixel has data this becomes a problem.

Our current solution for this to to just ignore the boundary of the image, since the majority of the data is still safe to use. But this is not sustainable for large images, or running on real hardware since as described in [[QED]] each shot of the circuit will only give one pixel, so we end up wasting many shots on detecting the boundary of the image. Requiring many more shots to get a clear edge detection.

#TODO Add examples photos of the boundary error