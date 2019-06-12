package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// newtonsRegression will use netwons method to computer linear regression on a
// two dimensional vectorspace
func newtonsRegression(xx, yy []float64, epsilon float64, show bool) (float64, float64) {
	fmt.Println(xx, yy)
	var B0 float64 = 0
	X := mat.NewDense(len(xx), 1, xx)
	y := mat.NewVecDense(len(yy), yy)
	B := mat.NewVecDense(1, nil)

	// step gradient and update gradient parameters
	dB, dB0 := gradientStep(X, y, B, B0)
	H := hessian(X, B)
	fmt.Printf("H: %v\n", mat.Formatted(H))
	B.AddVec(B, dB)
	B0 += dB0

	fmt.Printf(
		"stepping gradient B: %v\nB0: %v\n",
		mat.Formatted(B),
		B0,
	)

	fmt.Printf(
		"stepping gradient dB: %v\ndB0: %v\n",
		mat.Formatted(dB),
		dB0,
	)
	return 0, 0
}

// gradientStep will compute the change in the gradient of weights as well as
// the change in B0 given the observations, target variable, current
// gradient, and B0.
//
// j represents primarily the jth element of the vector as well as other
// corresponding jth columns observations
// i represents the ith observation of x observations
func gradientStep(x *mat.Dense, y, g mat.Vector, b0 float64) (*mat.VecDense, float64) {
	xr, xc := x.Dims()

	// create a resulting gradient step vector
	d_g := mat.VecDenseCopyOf(g)
	d_g.Zero()
	gr, gc := d_g.Dims()
	fmt.Printf("%v %v\n", gr, gc)

	// iterate over the number columns in x vector to step the gradient
	fmt.Printf("xr, xc = %d, %d\n", xr, xc)
	for j := 0; j < xc; j++ {
		for i := 0; i < xr; i++ {
			fmt.Printf("%d %d: %f\n", i, j, x.At(i, j))
			dBeta_ij := d_g.At(j, 0) + (x.At(i, j) * (y.AtVec(i) - (mat.Dot(g, x.RowView(i)) + b0)))
			d_g.SetVec(j, dBeta_ij)
		}
	}
	var db0 float64
	for i := 0; i < xr; i++ {
		db0 += y.At(i, 0) - (mat.Dot(g, x.RowView(i)) + b0)
	}

	// scale gradient and B0
	scale := -2.0 / float64(xr)
	db0 *= scale
	d_g.ScaleVec(scale, d_g)
	return d_g, db0
}

// hessian will calculate the hessian of the loss function given the gradient
// and the x observations
func hessian(x *mat.Dense, g mat.Vector) *mat.Dense {
	xr, xc := x.Dims()
	H := mat.NewDense(xc, xc, nil)
	fmt.Println(mat.Formatted(H))

	for j := 0; j < xc; j++ {
		for k := 0; k < xc; k++ {
			for i := 0; i < xr; i++ {
				dBeta_ijk := H.At(j, k) + x.At(i, j)*x.At(i, k)*g.At(0, k)
				H.Set(j, k, dBeta_ijk)
				// dBeta_ij := d_g.At(j, 0) + (x.At(i, j) * (y.AtVec(i) - (mat.Dot(g, x.RowView(i)) + b0)))
				// d_g.SetVec(j, dBeta_ij)
			}

		}
	}
	scale := 2.0 / float64(xr)
	H.Scale(scale, H)
	return H
}
