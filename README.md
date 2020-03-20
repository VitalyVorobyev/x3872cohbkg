# x3872cohbkg

A toy model for X(3872) -> J/psi pi+ pi- with coherenct background. Invariant mass m(J/psi pi+ pi-) is considered. The X(3872) lineshape is described with a relativistic Breit-Wigner. Backgound is divided into coherennt and non-coherent parts. Coherent background is described with 1st order polynomial, while non-coherent background is described with 2nd order polynomial. Detector resolution is described with Gaussian.

The model parameters are:
 1. X(3872) mass
 2. X(3872) width
 3. fraction of coherent background
 4. phase of coherent background
 5. fraction of non-coherent background
 6. width of the Gaussian (detector resolution)
 7. polynomial coefficient for coherent background lineshape
 8 and 9. two polynomial coefficients for non-coherent background lineshape

Dependences:
 - `scipy`: `numpy` and `matplotlib`
 - `iminuit`
 
 Content:
  - `lineshapes.py` contains model definition and model parameters
  - `plots.py` contains a number of ploting functions
  - `toymc.py` runs toy MC generation
  - `fitter.py` runs toy MC generation and then fitting with different models
  - `interactive_model.py` runs interactive lineshape visualization
