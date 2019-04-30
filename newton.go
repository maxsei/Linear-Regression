package main

import (
	"fmt"
	"math"
)

// correlationCoefficient returns the correlation coefficient of two equally
// sized vectors using the following formula:
//
//           nsum(xy)-sum(x)sum(y)
//    _______________________________________
//   ________________________________________
// \/ [nsum(x^2)-sum(x)^2][nsum(y^2)-sum(y)^2]
func correlationCoefficient(X, Y []float64) float64 {
	var sumX2, sumY2, sumXY, sumX, sumY float64
	for i := range X {
		sumX2 += math.Pow(X[i], 2)
		sumY2 += math.Pow(Y[i], 2)
		sumXY += X[i] * Y[i]
		sumX += X[i]
		sumY += Y[i]
	}
	n := float64(len(X))
	return (n*sumXY - sumX*sumY) /
		math.Pow((n*sumX2-math.Pow(sumX, 2))*(n*sumY2-math.Pow(sumY, 2)), .5)
}

// calcMAE calculates the mean absolute error given data points and slope
// intercept values
func calcMAE(X, Y []float64, m, b float64) (mAE float64) {
	for i := range Y {
		mAE += math.Abs(Y[i]-(m*X[i]+b)) / float64(len(Y))
	}
	return
}

// newtonsRegression will use netwons method to computer linear regression on a
// two dimensional vectorspace
func newtonsRegression(X, Y []float64, epsilon float64, show bool) (float64, float64) {
	// define m and b as well as their changes, the magnitude of those changes,
	// and the number of iterations counted
	var m, b, deltaM, deltaB, heshMagnitude, iterations float64
	// loop until magnitude is lower than epsilon except for the first iteration
	for heshMagnitude > epsilon || iterations == 0 {
		// calculate changes in m and b as well as calculating their combined magnitude
		deltaM, deltaB = stepNewton(X, Y, m, b)
		heshMagnitude = math.Pow((math.Pow(deltaM, 2) + math.Pow(deltaB, 2)), .5)
		if show {
			fmt.Printf("Iteration: %.0f\t%cm: %.8f\t%cb: %.8f\t |%cf|: %.8f\tm: %.16f\tb: %.16f\n",
				iterations, 0x0394, deltaM, 0x0394, deltaB, 0x0394, heshMagnitude, m, b)
		}
		//why are these signs different. Wolfe Conditions?
		m = m + deltaM
		b = b - deltaB
		iterations++
	}
	return m, b
}

// stepNewton computes a single iterative step of netwons methdod
func stepNewton(X, Y []float64, m, b float64) (float64, float64) {
	// dfine all partial derivates
	var fm, fb, fmm, fbb, fmb float64
	// sum up change in error with respect to m, b, m^2, and m*b
	// note that b^2 is not here because it is constant at 2.0
	for i := range X {
		fm += -X[i] * (Y[i] - (m*X[i] + b))
		fb += -(Y[i] - (m*X[i] + b))
		fmm += X[i] * X[i]
		fmb += X[i]
	}
	// multiply all partials by the two constant that is almost always
	// pulled out of the derivative equation and devide by n to normalize
	// note that the b^2 derivates is always a constant values of 2
	n := float64(len(X))
	fm = 2 * fm / n
	fb = 2 * fb / n
	fmm = fmm / n
	fbb = 2.0
	fmb = 2 * fmb / n
	// calculate the determinant of the hessian matrix and the change of m and b
	determ := fmm*fbb - math.Pow(fmb, 2)
	dm := (fm*fbb + fb*fmb) / determ
	db := (fm*fmb + fb*fmm) / determ
	return dm, db
}
