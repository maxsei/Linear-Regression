package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {

	// build flags
	showIterations := flag.Bool("s", true, "show each iteration of the regression process")
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

	// open data file
	f, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// parse columns of interest
	xcol, ycol, err := parseColumns(*columns)
	if err != nil {
		log.Fatal(err)
	}

	reader := csv.NewReader(f) //*csv.Reader
	X, Y, head := readInValues(xcol, ycol, reader)
	if *describe {
		fmt.Println("\n" + describeString(X, Y, head))
	}

	fmt.Println("Starting Regression")
	m, b := newtonsRegression(X, Y, *epsilon, *showIterations)
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

// parseColumns parseColumns will get the columns of interest from the columns
// string,  will return an error the column string is invalid
func parseColumns(colstr string) (int, int, error) {
	var xcol, ycol int
	colErr := fmt.Errorf("invalid columns string format: %s must be in form %s where columns are different values", colstr, "0,1")
	colSlice := strings.Split(colstr, ",")
	if len(colSlice) != 2 {
		fmt.Println("!=2")
		return -1, -1, colErr
	}
	if i, err := strconv.Atoi(colSlice[0]); err == nil {
		xcol = i
	}
	if i, err := strconv.Atoi(colSlice[1]); err == nil {
		ycol = i
	}
	if xcol == ycol {
		return -1, -1, colErr
	}
	return xcol, ycol, nil
}

// describeString will return a string that has the up to the first five entries
// of the values of interest along with header information for both columns
func describeString(X, Y []float64, head string) string {
	result := head + "\n"
	for i := 0; i < 5; i++ {
		if i >= len(X) {
			continue
		}
		result += fmt.Sprintf("%.3f\t%.3f\n", X[i], Y[i])
	}
	return result
}

// readInValues takes in a csv reader and returns a two vectors for x and y
// values respectively. The function will also skip over rows that have missing
// data for row column values that are of interest. Also returns header
// information of the csv file if their is any (headers cannot contain labels
// that can be parsed intto numeric types)
func readInValues(xcol, ycol int, reader *csv.Reader) ([]float64, []float64, string) {
	var X []float64
	var Y []float64
	var head string
	line := 1
	for {
		// read lines of csv until reaching the end
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		// handle header labels
		if line == 1 {
			line++
			if _, err := strconv.Atoi(record[xcol]); err == nil {
				head = ""
				continue
			}
			head += record[xcol] + "\t"
			if _, err := strconv.Atoi(record[ycol]); err == nil {
				head = ""
				continue
			}
			head += record[xcol]
			continue
		}
		// if there is a missing value in any row column that is being parsed
		// the loop will skip the row entirely
		XVal, err := strconv.ParseFloat(record[xcol], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}
		YVal, err := strconv.ParseFloat(record[ycol], 64)
		if err != nil {
			log.Printf("Parseing line %d failed, enexpected type\n", line)
			continue
		}
		X = append(X, XVal)
		Y = append(Y, YVal)
		line++
	}
	return X, Y, head
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

// plotRegression takes plotter.XYs pairs for bo
func plotRegression(pts plotter.XYs, linepts plotter.XYs, fname string) {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.X.Label.Text = "TV"
	p.Y.Label.Text = "Sales"
	p.Add(plotter.NewGrid())
	// Add the scatter plot points for the observations.
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.GlyphStyle.Radius = vg.Points(3)
	// Add the line plot points for the predictions.
	l, err := plotter.NewLine(linepts)
	if err != nil {
		log.Fatal(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	// Save the plot to a PNG file.
	p.Add(s, l)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, fname); err != nil {
		log.Fatal(err)
	}
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
