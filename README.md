# Bicubic Image Interpolation

Interpolation is a method of constructing new data points within the range of a discrete set of known data points.

Linear methods of interpolation are common place in nearly every piece of technology in modern life, they play particularly important roles in [Image Scaling](https://en.wikipedia.org/wiki/Image_scaling).

![Interpolation_Methods](https://github.com/Cabralcm/bicubic/blob/main/math/2d_images.png)

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


<img src="https://github.com/Cabralcm/bicubic/blob/main/math/scenario.png " alt="drawing" width="400" height="400"  style="margin-left: auto;
  margin-right: auto;"/>

![Problem](https://github.com/Cabralcm/bicubic/blob/main/math/scenario.png = 250x250)

![Test Image](https://github.com/Cabralcm/bicubic/blob/main/math/bicubic_sum.png)

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
