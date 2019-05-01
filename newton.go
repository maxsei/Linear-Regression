package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// newtonsRegression will use netwons method to computer linear regression on a
// two dimensional vectorspace
func newtonsRegression(X, Y []float64, epsilon float64, show bool) (float64, float64) {
	b_t := mat.NewDense(2, len(X), nil)
	x := mat.NewDense(2, len(X), append(X, Y...))
	// fmt.Printf("x =\n%+v\n", mat.Formatted(x))
	// fmt.Printf("b_t =\n%+v\n", mat.Formatted(b_t))
	// xrow, xcol := x.Dims()
	// fmt.Printf("x rows: %d\tx columns: %d\n", xrow, xcol)
	g_bt := gradient(x, b_t)
	fmt.Printf("gradient of weights =\n%+v\n", mat.Formatted(g_bt))
	return 0, 0
}

// gradientGN will compute and return the gradient of a matrix containing the
// vectors that represent columns of data
func gradient(x, b mat.Matrix) mat.Matrix {
	xr, xc := x.Dims()
	g := mat.NewDense(xr, xc, nil)
	for i := 0; i < xr; i++ {
		x_n := mat.Row(nil, i, x)
		fmt.Printf("x_n = %+v\n", x_n)
	}
	// Row(dst []float64, i int, a Matrix)
	return g
}
