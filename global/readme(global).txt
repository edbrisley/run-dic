Global DIC library

The global DIC library is held apart from the main library containing the local and stereo-DIC functionality. This is done so that modifying the global-DIC code for performance improvement is simpler, before refactoring it into the main library. This library also has improved useability over the intial release of the local and stereo-DIC library.
The script 'DIC_challenge.py' contains the case study for the 2D-DIC challenge 2.0 for metrological performance evaluation.

Functionality:

- Quadrilateral shape functions, Q4 and Q8
- Single element type in mesh
- Rectangular region of interest
- Tikhonov regularization, gradient constraint
