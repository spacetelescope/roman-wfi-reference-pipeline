"""
Placeholder for aperture correction reference file code.

This folder is set aside for Andrea Bellini's existing Fortran or C code
used to create aperture correction reference files. The code can be added
as-is and either wrapped so it can be called from Python, or used as a
reference for a later rewrite in Python.

Adding the code
---------------
1. Copy the original source files (.f, .f90, .c, .h) into this folder.
   Keep them unmodified at first so results can be checked against the
   original.
2. Add a short README noting where the code came from, who maintains it,
   how it was originally compiled and run, and any restrictions on
   sharing it.
3. Include a small sample input and its expected output, if available,
   so a wrapper or rewrite can be tested against the original.

Calling the code from Python
----------------------------
There are three common approaches, from least to most integrated:

a) Run it as a separate program
   Compile the code into an executable as usual (e.g. with gfortran or
   gcc) and call it from Python with ``subprocess.run``. This needs no
   changes to the original code and is often the quickest start.

b) Fortran via f2py
   NumPy's f2py builds a Python module directly from Fortran source:

       python -m numpy.f2py -c apcorr.f90 -m apcorr

   Then, in Python:

       import apcorr
       result = apcorr.some_subroutine(arg1, arg2)

   Arrays are passed as NumPy arrays. On Python 3.12 and later, f2py
   builds with meson, so meson and ninja must be installed.

c) C via ctypes or cffi
   Compile the C code into a shared library:

       gcc -shared -fPIC -o libapcorr.so apcorr.c

   Then load it from Python with ``ctypes.CDLL("libapcorr.so")`` and
   declare the argument and return types of each function you call.
   For larger C codebases, cffi, Cython or pybind11 are easier to
   maintain.

Packaging
---------
If the compiled code becomes a permanent part of the package, add the
build step to the project's build configuration (e.g. meson-python or
scikit-build-core in pyproject.toml) so it compiles on install, rather
than relying on prebuilt binaries committed to the repo.

Rewriting in Python
-------------------
If the code is later rewritten in Python, keep the original here until
the new version reproduces its output on the sample data.
"""