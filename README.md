# Bicubic Image Interpolation

Interpolation is a method of constructing new data points within the range of a discrete set of known data points.

Linear methods of interpolation are common place in nearly every piece of technology in modern life.

Some methods of linear interpolation for 2D dataset (e.g. images) include:

1) Nearest Neighbor
2) Bilinear
3) Bicubic
4) Cubic Splines
5) Sinc Interpolation

Two dimensional linear interpolation is often used in display screens (TVs, Cell Phones, etc), for instance when a low
resolution image is blown up on a higher resolution screen.

We will be looking at the Bicubic Method of Linear Interpolation. It tends to be smoother than Nearest Neighbour and Bilinear,
and does not require computation of a derivative as typically performed in Cubic Spline Interpolation.

It is important to note that **Linear Systems** have the following two (2) mathematical properties:

1) **Homogeneity**

> If x[n] = y[n], Then k*x[n] = k*y[n]

2) **Additivity** 

> If x1[n] = y1[n], and x2[n] = y2[n], Then x1[n] + x2[n] = y1[n] + y2[n]

# Python Script

The Python script has two methods of Bicubic Interpolation
1) **Brute Force** - Every point is individually interpolated by solving a Bicubic set of equations
2) **Column-Row Cubic Interpolation** - Each column is first interpolated using a Cubic function, following by appying a Cubic function to each row.
> Note: The order does not matter. It could have been easily named "Row-Column". The main concept is that Cubic Interpolation is being applied twice. This method works because we are looking at LINEAR methods, and the properties of Homogeneity and Addivity therefore apply, which allows us to deconstruct Bicubic Interpolation into two successive iterations of Cubic Interpolation. 

The python script also computes the Mean Squared Error (MSE) between `OpenCV` method of image scaling (which is a form of image interpolation)
with the Efficient Interpolation method.

## Image Input/Output

The python script will read Image files from the `./Image` directory. It will save the *Bicubic* interpolated images in the `./OutputImages` directory

There are options in the `Bicubic()` constructor to automatically **show** bicubic images as they are interpolated.

# Image Iterpolation Problem



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