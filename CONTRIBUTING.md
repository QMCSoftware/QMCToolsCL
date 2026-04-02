# Release

To create a release of `qmctoolscl`on PyPI:

1. go to the GitHub Actions to "PDM Build Windows" and download the `dist` artifact. Note that if you were to do `pdm build` locally you would not generate the Windows wheel which is crucial for Windows users to easily install `qmctoolscl` without having a C/C++ compiler.
2. Move the contents of the downloaded `dist/` folder into `QMCToolsCL/dist/`.
3. Open a terminal rooted at `QMCToolsCL/`
4. Run `twine check dist/*`.
5. Run `pdm publish --no-build --repository testpypi` and check the install. To clone from `testpypi`, use a command like `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ qmctoolscl==1.2.0.0.3a0`.
6. Run `pdm publish --no-build`. 