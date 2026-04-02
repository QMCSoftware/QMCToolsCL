# Release

To create a release of `qmctoolscl`on PyPI:

1. go to the GitHub Actions to "release Windows sdist bdist_wheel" and download the `dist` artifact. Note that if you were to do `python setup.py sdist` locally you would not generate the Windows wheel which is crucial for Windows users to easily install `qmctoolscl` without having a C/C++ compiler.
2. Move the contents of the downloaded `dist/` folder into `QMCToolsCL/dist/`.
3. Open a terminal rooted at `QMCToolsCL/`
4. Run `twine check dist/*`.
5. Run `twine upload --repository testpypi dist/*` and check the install.
6. Run `twine upload dist/*`. 