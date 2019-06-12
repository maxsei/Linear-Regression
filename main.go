package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/plot/plotter"
)

func main() {

	// build flags
	iterationsVisible := flag.Bool("v", true, "make each iteration of the newtons method process visible in stdout")
	inputFile := flag.String("i", "test.csv", "input path to csv data source where only the first two columns are read")
	outputFile := flag.String("o", "defaults to name of input data", "the name of the file that the line is outputted to. Must be *.png,*jpeg. *bmp")
	columns := flag.String("c", "0,1", "specify the columns that you want to read in from the csv in the format: row,col ")
	describe := flag.Bool("d", false, "output the first five elements of the data used and along with header information")
	epsilon := flag.Float64("e", .001, "define the value of epsilon")
	flag.Parse()

	// by default assign output file to the
	if *outputFile == "defaults to name of input data" {
		tempVal := (*inputFile)[:len(*inputFile)-4] + ".png"
		*outputFile = tempVal
	}

	// parse columns of interest
	xcol, ycol, err := parseColumns(*columns)
	if err != nil {
		log.Fatal(err)
	}

	// open data file
	f, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// read in values from csv ready
	reader := csv.NewReader(f) //*csv.Reader
	X, Y, head := readInValues(xcol, ycol, reader)
	if *describe {
		fmt.Println("\n" + describeString(X, Y, head))
	}

	fmt.Println("Starting Regression")
	m, b := newtonsRegression(X, Y, *epsilon, *iterationsVisible)
	r := correlationCoefficient(X, Y)
	pc := pearsonCoefficient(X, Y)

	fmt.Printf(
		"\nRegression Line: y = %.8fx + %.8f\n"+
			"Correlation Coefficient: %.8f\n"+
			"Pearson Coefficient: %.8f\n"+
			"MAE: %.8f\n\n",
		m, b, r, pc, calcMAE(X, Y, m, b),
	)

	// make XY pairs for original data as well data points created from the
	// regression line equation
	pts := make(plotter.XYs, len(X))
	ptsPred := make(plotter.XYs, len(X))
	for i := range X {
		pts[i].X = X[i]
		pts[i].Y = Y[i]
		ptsPred[i].X = X[i]
		ptsPred[i].Y = m*X[i] + b
	}
	plotRegression(pts, ptsPred, *outputFile)
}

// calcMAE calculates the mean absolute error given data points and slope
// intercept values
func calcMAE(X, Y []float64, m, b float64) (mAE float64) {
	for i := range Y {
		mAE += math.Abs(Y[i]-(m*X[i]+b)) / float64(len(Y))
	}
	return
}

// predict will predict the y value for the associated x values
func predict(m, x, b float64) float64 {
	return m*x + b
}

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

// pearsonCoefficient will calculate the pearson coefficient of the dataset
func pearsonCoefficient(X, Y []float64) float64 {
	var sumXMeanDiff, sumYMeanDiff float64
	var meanX, meanY, sdX, sdY float64
	for i := range X {
		meanX += X[i] / float64(len(X))
		meanY += Y[i] / float64(len(Y))
	}
	for i := range X {
		sumXMeanDiff += X[i] - meanX
		sumYMeanDiff += Y[i] - meanY
		sdX += math.Pow(X[i]-meanX, 2)
		sdY += math.Pow(Y[i]-meanY, 2)
	}
	sdX = math.Pow(sdX, .5)
	sdX = math.Pow(sdY, .5)
	return sumXMeanDiff * sumYMeanDiff / (sdX * sdY)
}
