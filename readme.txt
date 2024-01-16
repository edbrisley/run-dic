RUN-DIC is an open-source Digital Image Correlation software system implemented in Python. It may be used for high-resolution 3D surface reconstruction, 3D displacement tracking, and surface strain measurement. The library functionality currently includes:

![Strain Example](https://github.com/edbrisley/run-dic/raw/main/strain_example.png "Strain Example")

.. figure:: https://github.com/edbrisley/run-dic/raw/main/strain_example.png
   :alt: Example of surface strain measurement


Current functionality:

- local 2D solver (lucas-kanade) which utilises the state-of-the art inverse-compositional gauss-newton optimization scheme (ICGN)
- global 2D solver (finite element based) which utilises the modified Gauss-Newton minimization scheme, with Tikhonov regularization (global folder)
- strain computation via a virtual strain gauge (VSG) which utilises a local, linear polynomial fit
- depth estimation, currently using the default openCV triangulation function, but the SVD and optimal approaches are also available
- implicit camera calibration (3D calibration object, no distortion correction) (calibration folder)

In progress for next release:

- documentation
- unit tests
- automated requirements installation for conda and pip

Download all files in workspace to run strain study example
