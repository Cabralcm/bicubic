# Bicubic Image Interpolation

Interpolation is a method of constructing new data points within the range of a discrete set of known data points.

Linear methods of interpolation are common place in nearly every piece of technology in modern life, they play particularly important roles in [Image Scaling](https://en.wikipedia.org/wiki/Image_scaling).

<img src="https://github.com/Cabralcm/bicubic/blob/main/math/2d_images.png" alt="drawing"/>

Some methods of linear interpolation for 2D datasets (e.g. images) include:

1) Nearest Neighbor
2) Bilinear
3) Bicubic
4) Cubic Splines
5) Sinc Interpolation (Lanczos resampling)

Two dimensional linear interpolation is often used in display screens (TVs, Cell Phones, etc). A good example is when a low
resolution image is blown up on a higher resolution screen.

We will be looking at the **Bicubic Method** of Linear Interpolation. It tends yield substantially better results than Nearest Neighbour and Bilinear,
it does not require the computation of a derivative as typically performed in Cubic Spline Interpolation, and can be considered a computationally efficient approximation to Lanczos resampling.

It is important to note that **Linear Systems** have the following two (2) mathematical properties:

1) **Homogeneity**

> If x[n] = y[n], Then k*x[n] = k*y[n]

2) **Additivity** 

> If x1[n] = y1[n], and x2[n] = y2[n], Then x1[n] + x2[n] = y1[n] + y2[n]

These properties play a crucial role in making the Bicubic Interpolation Method more efficient!

# Python Script

Simply execute `main.py` to run Bicubic Interpolation. Scaling has default value of 2x.

The Python script has two methods of Bicubic Interpolation
1) **Brute Force** - Every point is individually interpolated by solving a Bicubic set of equations
2) **Column-Row Cubic Interpolation** - Each column is first interpolated using a Cubic function, following by appying a Cubic function to each row. This effectively exploits the Linear properties of the Bicubic equation and enables two easier computations to be performed, and uses the outputs of first cubic iteration as inputs to the second iteration.
 
> Note: The order does not matter. It could have been easily named "Row-Column". The main concept is that Cubic Interpolation is being applied twice. This method works because we are looking at LINEAR methods, and the properties of Homogeneity and Addivity therefore apply, which allows us to deconstruct Bicubic Interpolation into two successive iterations of Cubic Interpolation. 

The python script also computes the Mean Squared Error (MSE) between `OpenCV` method of image scaling (which is a form of image interpolation)
with the Efficient Interpolation method.

## Dependencies and Input/Output

Python dependencies:
```
Python 3.6 or newer
SciPy 1.4.1
Numpy 1.18.1
opencv-python 4.5.1.48 (older versions should be fine too)
```

The python script will read Image files from the `./Image` directory. It will save the *Bicubic* interpolated images in the `./OutputImages` directory

There are options in the `Bicubic()` constructor to automatically **show** bicubic images as they are interpolated.

# 2D Image Iterpolation Problem

In the figure below, the yellow dots are the original pixels of a low-resolution image, while the white dots (pure white, and white with stripes) are the missing pixels to be interpolated.

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/scenario.png " alt="drawing" width="400" height="400"/>
</p>

Cubic interpolation requires Four (4) data points to interpolate an unknown point. Bicubic interpolation requires Sixteen (16) data points.

## Bicubic Interpolation

The mathematical formulation the Bicubic Algorithm is as follows:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/bicubic.png " alt="drawing"/>
</p>

Where the **a** terms are the coefficients of the **Bicubic Interpolation Function**. 

The two-dimensional function `f(x,y)` interpolates the data points in a 2D plane, and approximates a single point based on surrounding 16 (x,y) data points.

## Cubic Interpolation

**Cubic interpolation** in 1 dimension can be extended to **Bicubic Interpolation** in 2 dimensions. 

Cubic interpolation for a single point can be expressed in the following polynomial:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/bicubic_sum.png " alt="drawing"/>
</p>

Likewise, the **a** terms are the coefficients of the **Cubic Interpolation Function**.

## Cubic Example

With respect to our missing pixel region, we can select 4 different known pixels, for instance:
- x0 = -1
- x1 = 0
- x2 = 1
- x3 = 2

Using the **Cubic Interpolation function**, we create 4 separate equations. Each equation corresponds to the original **Cubic Interpolation Polynomial/Function** *f(x)*.

We require 4 equations, since we are trying to solve for 4 unknowns, the 4 coefficients `a3, a2, a1, and a0`

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/bicubic_equations.png " alt="drawing"/>
</p>

To make it easier to solve this **linear system of equations**, we can convert them into matrix form.

Let's call this matrix **B**. Each row corresponds to one of the **Cubic Interpolation Functions**, and each column contains the coefficients of each of the **a** terms (from its respective Cubic Interpolation Function).

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/B_matrix_first.png " alt="drawing"/>
</p>

We can define the *vector* **Y** as the output of these 4 equations:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/y_equation.png " alt="drawing"/>
</p>

And define *vector* **a** as the coefficients for the Cubic Interpolation Function:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/vector_a.PNG " alt="drawing"/>
</p>

These equations can written as:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/y_vector.png " alt="drawing"/>
</p>

Thus, if we wish to interpolate an unknown point *x* in 1-D, using the **Cubic Interpolation Function**, *f(x)*, we can define *f(x)* in matrix form as follows:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/f_solution_matrix.png " alt="drawing"/>
</p>

For example, if want to interpolate an unknown data point, `x = 0.5`, we would have the following system of equations.

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/f_example_0_5.PNG" alt="drawing"/>
</p>

Applying matrix multiplcation, this simplifies to:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/f_example_0_5_2.PNG " alt="drawing"/>
</p>

## Cubic Summary

We have taken the Cubic Interpolation equation, which has 4 unknowns, corresponding to the **a coefficients**, `a0, a1, a2, and a3`.

We require 4 equations to solve for 4 unknowns.

We used the following known *x* datapoints to solve for the **a coefficients**: `x0 = -1, x1 = 0, x2 = 1, x3 = 2`.

After we have solved for the coefficients, we can use their weights to extrapolate an unknown point `x = 0.5`.

> Important: The neighbourhood that you choose for your KNOWN points is directly corresponds to the accuracy of the interpolation. If you choose KNOWN points that close to the UNKNOWN point(s), then you will *typically* have an more accurate result than if you use KNOWN points that are far away from your UNKNOWN points

**Example**
If you are trying to interpolate x = 0.5, it is better to use the KNOWN points: x0 = -1, x1 = 0, x2 = 1, x3 = 2

Than if you use the KNOWN points of:
x0 = -100, x1 = -50, x2 = 50, x3 = 100

This does not always hold true, it depends on the distribution of your data.

> Aside: If this wasn't clear, the value that **a coefficients** taken on, are dependent upon the input KNOWN datapoints.

# Bicubic Extension of Cubic Interpolation

For the **Bicubic Interpolation formula** our system of equations in matrix form is as follows:

<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/bicubic_interpolation.png" alt="drawing"/>
</p>

The function `f(x,y)` can be represented by *matrix* **F** for all 16 data points required to interpolate an unknown point.

**Example**
We will use the following data points to as an example:
```
x = {-1,0,1,2}

y = {-1,0,1,2}

As arguments in the function f(x,y)
```

**F** = 
<p align="center">
<img src="https://github.com/Cabralcm/bicubic/blob/main/math/bicubic_example.png" alt="drawing" style="float:left;"/>
</p>






### System of Equations
```
a00 = f(1,1)
a01 = -0.5*f(1,0) + 0.5*f(1,2)
a02 = f(1,0) - 2.5*f(1,1) + 2*f(1,2) - 0.5*f(1,3)
a03 = -0.5*f(1,0) + 1.5*f(1,1) - 1.5*f(1,2) + 0.5*f(1,3)
a10 = -0.5*f(0,1) + 0.5*f(2,1)
a11 = 0.25*f(0,0) - 0.25*f(0,2) - 0.25*f(2,0) + 0.25*f(2,2)
a12 = -.5*f(0,0) + 1.25*f(0,1) - f(0,2) + .25*f(0,3) + .5*f(2,0) - 1.25*f(2,1) + f(2,2) -.25*f(2,3)
a13 = .25*f(0,0) - .75*f(0,1) + .75*f(0,2) - .25*f(0,3) - .25*f(2,0) + .75*f(2,1) - .75*f(2,2) + .25*f(2,3)
a20 = f(0,1) - 2.5*f(1,1) + 2*f(2,1) - .5*f(3,1)
a21 = -.5*f(0,0) + .5*f(0,2) + 1.25*f(1,0) - 1.25*f(1,2) - f(2,0) + f(2,2) + .25*f(3,0) - .25*f(3,2)
a22 = f(0,0) - 2.5*f(0,1) + 2*f(0,2) - .5*f(0,3) - 2.5*f(1,0) + 6.25*f(1,1) - 5*f(1,2) + 1.25*f(1,3) + 2*f(2,0) - 5*f(2,1) + 4*f(2,2) - f(2,3) - .5*f(3,0) + 1.25*f(3,1) - f(3,2) + .25*f(3,3)
a23 = -.5*f(0,0) + 1.5*f(0,1) - 1.5*f(0,2) + .5*f(0,3) + 1.25*f(1,0) - 3.75*f(1,1) + 3.75*f(1,2) - 1.25*f(1,3) - f(2,0) + 3*f(2,1) - 3*f(2,2) + f(2,3) + .25*f(3,0) - .75*f(3,1) + .75*f(3,2) - .25*f(3,3)
a30 = -.5*f(0,1) + 1.5*f(1,1) - 1.5*f(2,1) + .5*f(3,1)
a31 = .25*f(0,0) - .25*f(0,2) - .75*f(1,0) + .75*f(1,2) + .75*f(2,0) - .75*f(2,2) - .25*f(3,0) + .25*f(3,2)
a32 = -.5*f(0,0) + 1.25*f(0,1) - f(0,2) + .25*f(0,3) + 1.5*f(1,0) - 3.75*f(1,1) + 3*f(1,2) - .75*f(1,3) - 1.5*f(2,0) + 3.75*f(2,1) - 3*f(2,2) + .75*f(2,3) + .5*f(3,0) - 1.25*f(3,1) + f(3,2) - .25*f(3,3)
a33 = .25*f(0,0) - .75*f(0,1) + .75*f(0,2) - .25*f(0,3) - .75*f(1,0) + 2.25*f(1,1) - 2.25*f(1,2) + .75*f(1,3) + .75*f(2,0) - 2.25*f(2,1) + 2.25*f(2,2) - .75*f(2,3) - .25*f(3,0) + .75*f(3,1) - .75*f(3,2) + .25*f(3,3) 
```

### General Form
```
f(x,y) = (a00 + a01 * y + a02 * y2 + a03 * y3) +
(a10 + a11 * y + a12 * y2 + a13 * y3) * x +
(a20 + a21 * y + a22 * y2 + a23 * y3) * x2 +
(a30 + a31 * y + a32 * y2 + a33 * y3) * x3
```

# Additional Resources

1) [Interpolation - Dr. Xiaolin Wu, McMaster University](https://www.ece.mcmaster.ca/~xwu/3sk3/interpolation.pdf)

2) [Paul Breeuwsma](https://www.paulinternet.nl/?page=bicubic)

3) [Computerphile - Resizing Images](https://www.youtube.com/watch?v=AqscP7rc8_M)

4) [Computerphile - Bicubic Interpolation](https://www.youtube.com/watch?v=poY_nGzEEWM)

5) [2-D Interpolation - Dr. Ruye Wang, Harvey Mudd College](http://fourier.eng.hmc.edu/e176/lectures/ch7/node7.html)

5b) [ML Course - Dr. Ruye Wang, Harvey Mudd College](http://fourier.eng.hmc.edu/e176/)

6) [Bicubic - Michael Thomas Flanagan](https://www.ee.ucl.ac.uk/~mflanaga/java/BiCubicInterpolation.html)

7) [Java Bicubic - Ken Perlin, NYU](https://mrl.cs.nyu.edu/~perlin/java/Bicubic.html)