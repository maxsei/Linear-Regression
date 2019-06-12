package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// newtonsRegression will use netwons method to computer linear regression on a
// two dimensional vectorspace
func newtonsRegression(xx, yy []float64, epsilon float64, show bool) (float64, float64) {
	fmt.Println("\033[1;1H\033[2J")
	X := mat.NewDense(len(xx), 1, xx)
	y := mat.NewVecDense(len(yy), yy)
	B := mat.NewVecDense(2, nil)
	fmt.Println("Stepping the Gradient:")

	// B_t+1 =
	// B_t: [mat] -
	// fmt.Println("\u03d0\u209C\u208a\u2081\n=")
	// fmt.Printf("\u03d0\u209C:\n%v\n-\n", mat.Formatted(B))
	var magnitude float64
	// for i := 0; i < 1000; i++ {
	i := 0
	for magnitude > epsilon || i == 0 {

		// step gradient and update gradient parameters
		gradientB := gradientStep(X, y, B)
		magnitude = math.Pow(mat.Dot(gradientB, gradientB), 2)
		// gradient Beta: [mat]
		fmt.Printf("\u2362\u03d0\u209c:\n%v\n\u00B7\n", mat.Formatted(gradientB))

		H := hessian(X)
		fmt.Printf("H[X]:\n%v\n", mat.Formatted(H))

		inv_H := mat.DenseCopyOf(H)
		inv_H.Inverse(H)
		// fmt.Printf("inv_H:\n%v\n", mat.Formatted(inv_H))

		dB := mat.VecDenseCopyOf(B)
		dB.MulVec(inv_H, gradientB)
		// fmt.Printf("\u2362\u03d0\u209c\u00B7H\u207B\u00B9[X]:\n%v\n", mat.Formatted(dB))
		// dB.ScaleVec(-1, dB)
		// fmt.Printf("-\u2362\u03d0\u209c\u00B7H\u207B\u00B9[X]:\n%v\n", mat.Formatted(dB))

		B.AddVec(B, dB)
		fmt.Printf("\u03d0\u209C\u208a\u2081=\u03d0\u209C-\u2362\u03d0\u209c\u00B7H\u207B\u00B9[X]:\n%v\n", mat.Formatted(B))

		fmt.Printf("%d----------------------------------------\n", i)
		i++
	}
	return B.At(0, 0), B.At(1, 0)
}

// gradientStep will compute the change in the gradient of weights as well as
// the change in B0 given the observations, target variable, current
// gradient, and B0.
//
// j represents primarily the jth element of the vector as well as other
// corresponding jth columns observations
// i represents the ith observation of x observations
func gradientStep(X *mat.Dense, y, B *mat.VecDense) *mat.VecDense {
	xr, xc := X.Dims()
	// alias B0 and B1 - Bn vector for easy reference
	B0 := B.At(0, 0)
	Bn := B.SliceVec(0, xc)
	// fmt.Printf("Bn:\n%v\nVS\nB:\n%v\n", mat.Formatted(Bn), mat.Formatted(B))

	// instantiate a resulting gradient step vector
	dB := mat.VecDenseCopyOf(B)
	dB.Zero()

	// compute gradient with respect to B1 - Bn
	for j := 0; j < xc; j++ {
		for i := 0; i < xr; i++ {
			// fmt.Printf("%d %d: %f\n", i, j, X.At(i, j))
			dBeta_ij := Bn.At(j, 0) + (X.At(i, j) * (y.AtVec(i) - (mat.Dot(Bn, X.RowView(i)) + B0)))
			dB.SetVec(j+1, dBeta_ij)
		}
	}
	// compute gradient with respect to B0
	for i := 0; i < xr; i++ {
		dB0 := y.At(i, 0) - (mat.Dot(Bn, X.RowView(i)) + B0)
		dB.SetVec(0, dB0)
	}
	// scale gradient and B0
	scale := -2.0 / float64(xr)
	dB.ScaleVec(scale, dB)
	return dB
}

// hessian will calculate the hessian of the loss function given the gradient
// and the x observations
// func hessian(x *mat.Dense, B *mat.VecDense) *mat.Dense {
func hessian(x *mat.Dense) *mat.Dense {
	xr, xc := x.Dims()
	// alias B0 and B1 - Bn vector for easy reference
	// B0 := B.At(0, 0)
	// Bn := B.SliceVec(0, xc)
	// instantiate a resulting Hessian Matrix
	H := mat.NewDense(xc+1, xc+1, nil)
	// fmt.Println(mat.Formatted(H))

	for j := 0; j < xc+1; j++ {
		for k := 0; k < xc+1; k++ {
			// d^2 B0
			if j == 0 && k == 0 {
				H.Set(j, k, float64(xr))
				continue
			}
			for i := 0; i < xr; i++ {
				var dBeta_ijk float64
				switch {
				case j == 0:
					dBeta_ijk = H.At(j, k) + x.At(i, k-1)
				case k == 0:
					dBeta_ijk = H.At(j, k) + x.At(i, j-1)
				default:
					dBeta_ijk = H.At(j, k) + x.At(i, j-1)*x.At(i, k-1)
				}
				H.Set(j, k, dBeta_ijk)
			}

		}
	}
	scale := 2.0 / float64(xr)
	H.Scale(scale, H)
	return H
}
