from setuptools import setup

setup(name = "blochsimu",
      version = "0.1.2",
      description = "Bloch equation simulator for e.g. NMR-related simulations",
      classifiers = [
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      url = "http://github.com/k7hoven/blochsimu",
      author = "Koos C.J. Zevenhoven",
      author_email = "koos.zevenhoven@aalto.fi",
      license = "BSD",
      packages = ["blochsimu"],
      install_requires = ["numpy"],
      zip_safe = False)

