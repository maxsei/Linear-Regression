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
	inputFile := flag.String("i", "regression_test.csv", "input path to csv data source where only the first two columns are read")
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

	fmt.Printf(
		"\nRegression Line: y = %.8fx + %.8f\n"+
			"Correlation Coefficient: %.8f\n"+
			"MAE: %.8f\n\n",
		m, b, r, calcMAE(X, Y, m, b),
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
